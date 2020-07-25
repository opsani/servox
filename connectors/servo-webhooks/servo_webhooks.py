from datetime import datetime
from importlib.metadata import version 
from typing import Any, Dict, List, Optional, Union

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
    )
    description: Optional[str] = Field(
        description="Optional free-form text describing the context or purpose of the webhook.",
    )
    events: List[EventContext] = Field(
        description="A list of events that the webhook is listening for.",
    )
    url: AnyHttpUrl = Field(
        description="An HTTP, HTTPS, or HTTP/2 endpoint listening for webhooks event requests.",
    )
    secret: SecretStr = Field(
        description="A secret string value used to produce an HMAC digest for verifying webhook authenticity.",
    )
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
            if event_context := EventContext.from_str(value):
                return [event_context]
            raise ValueError(f"Invalid value for events")
        elif isinstance(value, (list, set, tuple)):
            events = []
            for e in value:
                event = EventContext.from_str(e)
                if not event:
                    raise ValueError(f"Invalid value for events")
                events.append(event)
        return events

class Configuration(servo.BaseConfiguration):
    webhooks: List[Webhook] = []

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
    name: str = "servo-webhooks"
    __default_name__ = "servo-webhooks"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._register_event_handlers()
    
    def _register_event_handlers(self) -> None:
        for webhook in self.config.webhooks:
            for event in webhook.events:
                if event.preposition == Preposition.BEFORE:
                    self._add_before_event_webhook_handler(webhook, event)
                elif event.preposition == Preposition.AFTER:
                    self._add_after_event_webhook_handler(webhook, event)
                else:
                    raise ValueError(f"Unsupported Preposition value given for webhook: '{event.preposition}'")

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
        async def __after_handler(self, results: List[EventResult], **kwargs) -> None:
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

            json_body = body.json()
            debug(json_body)
            async with httpx.AsyncClient(headers=headers) as client:
                response = await client.post(webhook.url, data=dict(foo="bar"))
                success = (response.status_code == httpx.codes.OK)

        self.add_event_handler(event.event, event.preposition, __after_handler)


Connector.__default_name__ = "servo-webhooks"

class CLI(servo.cli.ConnectorCLI):
    pass
    # add, remove, test
