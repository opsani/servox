import asyncio
import contextlib
from typing import Any, Dict, List, Optional, Tuple

import autobahn.asyncio.wamp
import autobahn.asyncio.component
import devtools
import pydantic
import pytest
import servo
import txaio

class WampConfiguration(servo.BaseConfiguration):
    realm: str
    port: int

    @property
    def url(self) -> str:
        return f"ws://127.0.0.1:{self.port}"

    def to_autobahn(self) -> Dict[str, Any]:
        return {
            "transports": [
                {
                    "type": "websocket",
                    "url": self.url,
                    "endpoint": {
                        "type": "tcp",
                        "host": "localhost",
                        "port": self.port,
                    },
                    "serializers": ["json"],
                },
            ],
            "realm": self.realm,
        }

class WampChecks(servo.BaseChecks):
    config: WampConfiguration
    publisher: servo.Publisher

    @servo.check("Report aggregation")
    async def check_with_progress(self) -> Tuple[bool, str]:
        progress = servo.DurationProgress('10s')

        async for update in progress.every('0.500ms'):
            await self.publisher(servo.Message(json=dict(progress=update.progress)))

        return (True, "Fuhgeddaboudit")

class WampConnector(servo.BaseConnector):
    config: WampConfiguration
    component: Optional[autobahn.asyncio.component.Component] = None
    _router_task: asyncio.Task = pydantic.PrivateAttr()
    _server_task: asyncio.Task = pydantic.PrivateAttr()
    _session: autobahn.asyncio.wamp.Session = pydantic.PrivateAttr()

    @servo.on_event()
    async def startup(self) -> None:
        # TODO: integrate with the rest of logging
        # txaio.start_logging(level='info')
        # logger = txaio.make_logger()
        # import logging
        # logging.getLogger("txaio").addHandler(servo.logging.InterceptHandler())
        router_event = asyncio.Event()
        async def _log_output(output: str) -> None:
            router_event.set()
            servo.logger.info(output)

        self._router_task = asyncio.create_task(
            servo.stream_subprocess_shell(
                f"nexusd -realm servo -ws :{self.config.port}", stdout_callback=_log_output
            )
        )
        # FIXME: Use standard connect timeout (self.config.settings.timeouts.connect)
        await asyncio.wait_for(router_event.wait(), timeout=5.0)

        self.component = _create_server_component(self.config)

        server_event = asyncio.Event()
        @self.component.on_join
        async def _joined(session, details) -> None:
            # Get a reference to the session for publication
            self._session = session

            servo.logger.success(f"WAMP interface initialized on: {self.config.url}: {self._session} ({details})")
            server_event.set()

        # Publish from WAMP to Servo pub/sub
        @self.component.subscribe('counter')
        async def _wamp_message_received(counter: Dict[str, int]) -> None:
            async with self.publish('counter') as publisher:
                # NOTE: utilize metadata to avoid duplicating messages across the bridge
                await publisher(servo.Message(json=counter, metadata={ 'source': 'WAMP' }))

        # FIXME: Use standard connect timeout (self.config.settings.timeouts.connect)
        self._server_task = self.component.start(loop=asyncio.get_event_loop())
        await asyncio.wait_for(server_event.wait(), timeout=5.0)

        # Subscribe to all messages traversing the exchange
        @self.subscribe('*')
        async def _message_received(message: servo.Message, channel: servo.Channel) -> None:
            servo.logger.info(f"Notified of a new Message: {message}, {channel}")
            if message.metadata.get('source') == 'WAMP':
                servo.logger.warning(f"Declining to publish Message to WAMP because it originated from WAMP")
            else:
                self._session.publish(channel.name, message.json())
            servo.logger.success(f"ServoX -> WAMP: Sent Message '{message.text}' to Channel '{channel.name}'")

    @servo.on_event()
    async def shutdown(self) -> None:
        servo.logger.info(f"Cancelling WAMP tasks: {self._router_task}, {self._server_task}")
        self._router_task.cancel()
        self._server_task.cancel()

        with contextlib.suppress(asyncio.CancelledError):
            await asyncio.gather(self._router_task, self._server_task)

    @servo.on_event()
    async def check(
        self,
        matching: Optional[servo.CheckFilter] = None,
        halt_on: Optional[servo.ErrorSeverity] = servo.ErrorSeverity.critical,
    ) -> List[servo.Check]:
        channel = self.pubsub_exchange.create_channel('checks')
        publisher = self.pubsub_exchange.create_publisher(channel)
        return await WampChecks(config=self.config, publisher=publisher).run_all(matching=matching, halt_on=halt_on)

    class Config:
        arbitrary_types_allowed = True

def _create_server_component(config: WampConfiguration) -> autobahn.asyncio.component.Component:
    component = autobahn.asyncio.component.Component(**config.to_autobahn())

    @component.on_join
    async def join(session, details):
        servo.logger.success(f"Client connected: {devtools.pformat(session)} {devtools.pformat(details)}")

    @component.on_leave
    async def leave(session, details):
        servo.logger.info(f"Client disconnected: {details}")

    # NOTE: Procedures must be registered before you start the component
    @component.register(
        "servo.whats_the_magic_number",
        options=autobahn.wamp.types.RegisterOptions(details_arg='details'),
    )
    async def _whats_the_magic_number(*args, **kwargs):
        servo.logger.info(f"RPC method `servo.whats_the_magic_number` invoked: {args} (kwargs={kwargs} -> 187")
        return 187

    return component


@pytest.fixture(autouse=True)
def _set_log_level() -> None:
    servo.logging.set_level("DEBUG")

@pytest.fixture(autouse=True)
async def _cleanup_tasks() -> None:
    # Yield to run after the test case
    yield

    tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    [task.cancel() for task in tasks]
    await asyncio.gather(*tasks, return_exceptions=True)

@pytest.fixture
def wamp_config(unused_tcp_port: int) -> WampConfiguration:
    return WampConfiguration(realm='servo', port=unused_tcp_port)

@pytest.fixture
async def wamp_connector(wamp_config: WampConfiguration) -> WampConnector:
    connector = WampConnector(config=wamp_config)
    await connector.dispatch_event(servo.Events.startup)
    try:
        connector.pubsub_exchange.start()
        yield connector
    finally:
        servo.logger.info("Shutting down WampConnector...")
        await connector.dispatch_event(servo.Events.shutdown)
        servo.logger.success("WampConnector successfully shut down.")

        # Ensure the pub/sub exchange is cleaned up (handled by Servo typically)
        if connector.pubsub_exchange.running:
            await connector.pubsub_exchange.shutdown()
        else:
            connector.pubsub_exchange.clear()


async def test_servo_pubsub_to_wamp_bridging(wamp_connector: WampConnector, event_loop: asyncio.AbstractEventLoop) -> None:
    # Verify that ServoX -> WAMP pub/sub bridging works as expected
    # A ServoX connector, an Autobahn component, and a standalone ServoX publisher object are utilized

    # Set up a publisher. Will send to the ServoX exchange and be replayed onto WAMP
    event = asyncio.Event()
    publisher = wamp_connector.pubsub_exchange.create_publisher('counter')
    async def _publish_counter() -> None:
        # Wait for connection to the router
        await event.wait()

        for counter in range(1, 5):
            # Publish to the ServoX exchange
            servo.logger.critical(f"Publishing counter: {counter} ({publisher})")
            await publisher(servo.Message(json=dict(counter=counter)))
            await asyncio.sleep(0.1)

        servo.logger.success(f"Finished publising counters (counter={counter})")

    # Set up Autobahn WAMP client component
    async def main(reactor, session):
        # Release the blocking publisher
        event.set()

        # Give the pub/sub enough time to process
        await asyncio.sleep(0.5)

    client_component = _create_client_component(wamp_connector.config, main)
    counters = []

    @client_component.subscribe(
        "counter",
        options=autobahn.wamp.types.SubscribeOptions(match='prefix', details_arg='details'),
    )
    async def _record_counters(message_json: str, **kwargs):
        message = servo.Message(json=message_json)
        servo.logger.success(f"Recording Message: {message}")
        counters.append(message.json()['counter'])

    # Run the whole apparatus
    client_task = client_component.start(loop=event_loop)
    await asyncio.gather(_publish_counter(), client_task)

    assert counters == [1, 2, 3, 4]

async def test_wamp_pubsub_to_servo_bridging(wamp_connector: WampConnector, event_loop: asyncio.AbstractEventLoop) -> None:
    # Verify that WAMP -> ServoX pub/sub bridging works as expected

    # Set up a publisher. Will send to the ServoX exchange and be replayed onto WAMP
    event = asyncio.Event()

    # Set up Autobahn WAMP client component
    async def main(reactor, session):
        # Exit immediately
        ...

    # Accumulate the counter messages by subscribing to the connector exchange
    counters = []
    @wamp_connector.subscribe('counter')
    async def _record_counters(message: servo.Message, channel: servo.Channel) -> None:
        servo.logger.success(f"Recording Message: {message}")
        counters.append(message.json()['counter'])


    # Publish messages from WAMP origin
    client_component = _create_client_component(wamp_connector.config, main)
    @client_component.on_join
    async def join(session, details):
        servo.logger.critical(f"Client connected: {devtools.pformat(session)} {devtools.pformat(details)}")

        for counter in range(1, 5):
            # Publish to the WAMP
            servo.logger.critical(f"Publishing counter: {counter}")
            session.publish('counter', dict(counter=counter))

        servo.logger.success(f"Finished publising counters (counter={counter})")
        event.set()

    # Run the whole apparatus
    client_task = client_component.start(loop=event_loop)
    await asyncio.gather(event.wait(), client_task)

    assert counters == [1, 2, 3, 4]

async def test_client_to_servo_rpc(wamp_connector: WampConnector, event_loop: asyncio.AbstractEventLoop) -> None:
    # Test calling an RPC from the client component to the servo

    event = asyncio.Event()
    _session = None
    client_component = _create_client_component(wamp_connector.config)
    @client_component.on_join
    async def join(session, details):
        nonlocal _session
        _session = session
        event.set()

    client_task = client_component.start(loop=event_loop)
    await event.wait()
    result = await _session.call("servo.whats_the_magic_number")
    assert result == 187

async def test_servo_to_client_rpc(wamp_connector: WampConnector, event_loop: asyncio.AbstractEventLoop) -> None:
    # Test calling an RPC from the client component to the server

    event = asyncio.Event()
    client_component = _create_client_component(wamp_connector.config)
    @client_component.on_join
    async def join(session, details):
        event.set()

    client_task = client_component.start(loop=event_loop)
    await event.wait()
    result = await wamp_connector._session.call("client.whats_the_magic_number")
    assert result == 31337

async def test_observing_checks(wamp_connector: WampConnector, event_loop: asyncio.AbstractEventLoop) -> None:
    # Run a check that emits progress and render it as a progress bar
    from alive_progress import alive_bar

    servo.logging.set_level("CRITICAL")
    client_component = _create_client_component(wamp_connector.config)
    reports = []
    queue = asyncio.Queue()

    async def _progress_bar() -> None:
        with alive_bar(manual=True, bar='classic') as bar:
            while True:
                progress = await queue.get()
                if progress is None:
                    break

                bar(progress)
                queue.task_done()

    queue_processor = asyncio.create_task(_progress_bar())

    @client_component.subscribe(
        "checks",
        options=autobahn.wamp.types.SubscribeOptions(match='prefix', details_arg='details'),
    )
    async def _record_progress(message_json: str, **kwargs):
        message = servo.Message(json=message_json)
        progress = message.json()['progress']
        servo.logger.success(f"Recording Message: {message}")
        reports.append(progress)

        fraction = progress / 100.0
        await queue.put(fraction)

    # Run the whole apparatus
    client_task = client_component.start(loop=event_loop)
    await wamp_connector.dispatch_event(servo.Events.check)
    await client_component.stop()

def _create_client_component(config: WampConfiguration, main = None) -> autobahn.asyncio.component.Component:
    component = autobahn.asyncio.component.Component(
        **config.to_autobahn(),
        main=main,
    )

    @component.on_join
    async def join(session, details):
        servo.logger.info(f"Client connected: {devtools.pformat(session)} {devtools.pformat(details)}")

    @component.on_leave
    async def leave(session, details):
        servo.logger.info(f"Client disconnected: {devtools.pformat(details)}")

    # NOTE: Procedures must be registered before you start the component
    @component.register(
        "client.whats_the_magic_number",
        options=autobahn.wamp.types.RegisterOptions(details_arg='details'),
    )
    async def _whats_the_magic_number(*args, **kwargs):
        servo.logger.info(f"RPC method `client.whats_the_magic_number` invoked: {args} (kwargs={kwargs} -> 31337")
        return 31337

    return component
