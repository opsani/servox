from datetime import datetime
from importlib.metadata import version 
from typing import Any, Dict, List, Optional

import backoff
import httpx
from pydantic import AnyHttpUrl, BaseModel, SecretStr, validator, Field

import servo
from servo import metadata, License, Maturity, Duration
from servo.events import EventContext, EventResult, Preposition, get_event, event_handler

try:
    __version__ = version("servo-webhooks")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"

# TODO: Move into the events core
# TODO: Event.from_str() ??
def event_from_str(event_id: str) -> EventContext:
    preposition, event_name = event_id.split(":", 1)
    return EventContext(
        preposition=preposition, 
        event=get_event(event_name)
    )

class BackoffConfig(BaseModel):
    """
    The BackoffConfig class provides configuration for backoff and retry provided
    by the backoff library.
    """
    max_time: Duration = '3m'
    max_tries: int = 12

class Result(BaseModel):
    connector: str
    value: Any

class WebhookRequestBody(BaseModel):
    event: str
    created_at: datetime
    results: List[Result]

class Webhook(BaseModel):
    name: Optional[str] = Field(
        description="A unique name identifying the webhook.",
    ),
    description: Optional[str] = Field(
        description="Optional free-form text describing the context or purpose of the webhook.",
    ),
    events: List[EventContext] = Field(
        description="A list of events that the webhook is listening for.",
    ),
    url: AnyHttpUrl = Field(
        description="An HTTP, HTTPS, or HTTP/2 endpoint listening for webhooks event requests.",
    ),
    secret: SecretStr = Field(
        description="A secret string value used to produce an HMAC digest for verifying webhook authenticity.",
    ),
    headers: Dict[str, str] = Field(
        {},
        description="A dictionary of supplemental HTTP headers to include in webhook requests.",
    )
    backoff: BackoffConfig = BackoffConfig()

    # TODO: Move into the events core
    @validator("events", pre=True)
    @classmethod
    def validate_events(cls, value) -> List[EventContext]:
        if isinstance(value, str):
            return [event_from_str(value)]
        elif isinstance(value, (list, set, tuple)):
            events = []
            for e in value:
                if isinstance(e, EventContext):
                    events.append(e)
                elif isinstance(e, str):
                    events.append(event_from_str(e))
                else:
                    raise ValueError(f"invalid value of type '{e.__class__}'")
            return events
            
        return value

class Configuration(servo.BaseConfiguration):
    webhooks: List[Webhook]

    @classmethod
    def generate(cls, **kwargs) -> "Configuration":
        return cls(
            [
                Webhook(
                    name="My Webhook",
                    description="Listens for after:measure events and sends an email",
                    events=["after:measure"],
                    url="https://example.com/",
                    secret="s3cr3t!"
                )
            ],
            description="Add webhooks for the events you want to listen to.",
            **kwargs,
        )

    # Allow the webhooks to be root configuration
    def __init__(self, webhooks: List[Webhook], *args, **kwargs) -> None:
        super().__init__(webhooks=webhooks, *args, **kwargs)

@metadata(
    name="Webhooks",
    description="Dispatch servo events via HTTP webhooks",
    version=__version__,
    homepage="https://github.com/opsani/servo-webhooks",
    license=License.APACHE2,
    maturity=Maturity.EXPERIMENTAL,
)
class Connector(servo.Connector):
    config: Configuration

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._register_event_handlers()
    
    def _register_event_handlers(self) -> None:
        for webhook in self.config.webhooks:
            debug("webhook", webhook)
            for event in webhook.events:
                debug(event)
                if event.preposition == Preposition.BEFORE:
                    self._add_before_event_webhook_handler(webhook, event)
                elif event.preposition == Preposition.AFTER:
                    self._add_after_event_webhook_handler(webhook, event)
                else:
                    raise ValueError(f"Unsupported Preposition value given for webhook: '{event.preposition}'")
                # debug(event.preposition)
                # fn = event_handler(event.event.name, event.preposition)(fn)
                # debug(fn, event.preposition)
                # handler = fn.__event_handler__
                # debug(handler, event.preposition, handler.preposition)
                # self.__class__.__event_handlers__.append(handler)
    def _add_before_event_webhook_handler(self, webhook: Webhook, event: EventContext) -> None:
        async def __before_handler(self) -> None:
            headers = {
                "Content-Type": "application/json",
            }
            async with httpx.AsyncClient(headers=headers) as client:
                response = await client.post(webhook.url, data=dict(foo="bar"))
                success = (response.status_code == httpx.codes.OK)

        self.add_event_handler(event.event, event.preposition, __before_handler)
        
    def _add_after_event_webhook_handler(self, webhook: Webhook, event: EventContext) -> None:
        async def __after_handler(self, results: List[EventResult]) -> None:
            print(results)
            headers = {
                "Content-Type": "application/json",
            }

            outbound_results = []
            for result in results:
                outbound_results.append(
                    Result(
                        connector=result.connector.name,
                        value=result.value
                    )
                )
            body = WebhookRequestBody(
                event=str(event),
                created_at=datetime.now(),
                results=outbound_results
            )
            # class Result(BaseModel):
            #     connector: str
            #     value: Any

            # class WebhookRequestBody(BaseModel):
            #     event: str
            #     created_at: datetime
            #     results: List[Result]

            json_body = body.json()
            debug(json_body)
            async with httpx.AsyncClient(headers=headers) as client:
                response = await client.post(webhook.url, data=dict(foo="bar"))
                success = (response.status_code == httpx.codes.OK)

        self.add_event_handler(event.event, event.preposition, __after_handler)

class CLI(servo.cli.ConnectorCLI):
    pass
    # add, remove, test
