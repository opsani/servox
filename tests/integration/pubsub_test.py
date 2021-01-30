import asyncio
import pytest
import servo
import servo.pubsub
import servo.connectors.prometheus
import servo.connectors.vegeta

from typing import Callable, AsyncIterator

pytestmark = [pytest.mark.asyncio, pytest.mark.integration]

class TestVegeta:
    @pytest.fixture
    def connector(self) -> servo.connectors.vegeta.VegetaConnector:
        config = servo.connectors.vegeta.VegetaConfiguration(
            rate="50/1s",
            target="GET http://localhost:8080",
            reporting_interval="500ms"
        )
        return servo.connectors.vegeta.VegetaConnector(config=config)

    async def test_subscribe_via_exchange_subscriber_object(self, connector) -> None:
        reports = []

        async def _callback(message, channel) -> None:
            debug("Vegeta Reported: ", message.json())
            reports.append(message.json())

        subscriber = connector.pubsub_exchange.create_subscriber("loadgen.vegeta", callback=_callback)
        connector.pubsub_exchange.start()
        measurement = await asyncio.wait_for(
            connector.measure(control=servo.Control(duration="5s")),
            timeout=7 # NOTE: Always make timeout exceed control duration
        )
        assert len(reports) > 5

    async def test_subscribe_via_exchange_context_manager(self, connector) -> None:
        connector.pubsub_exchange.start()
        reports = []

        async def _subscribe_to_vegeta() -> None:
            async with connector.subscribe("loadgen.vegeta") as subscriber:
                async for message, channel in subscriber:
                    debug("Vegeta Reported: ", message.json())
                    reports.append(message.json())

        task = asyncio.create_task(_subscribe_to_vegeta())
        await connector.measure(control=servo.Control(duration="3s"))
        task.cancel()
        assert len(reports) > 5


    async def test_subscribe_via_connector(self, connector) -> None:
        ...

    async def test_subscribe_via_servo(self, connector) -> None:
        ...
