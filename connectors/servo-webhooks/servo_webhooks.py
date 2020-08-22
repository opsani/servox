from __future__ import annotations
import hmac
import hashlib
from datetime import datetime
from importlib.metadata import version 
from typing import Any, Dict, List, Optional, Union, Sequence

import backoff
import httpx
from pydantic import AnyHttpUrl, BaseModel, Field, SecretStr, validator

import servo
from servo import metadata, License, Maturity, Duration
from servo.events import (
    EventContext, 
    EventResult, 
    Preposition, 
    get_event, 
    event_handler, 
    validate_event_contexts
)


try:
    __version__ = version("servo-webhooks")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"


SUCCESS_STATUS_CODES = (
    httpx.codes.OK, 
    httpx.codes.CREATED, 
    httpx.codes.ACCEPTED, 
    httpx.codes.NO_CONTENT, 
    httpx.codes.ALREADY_REPORTED
)
CONTENT_TYPE = "application/vnd.opsani.servo-webhooks+json"

class BackoffConfig(BaseModel):
    """
    The BackoffConfig class provides configuration for backoff and retry provided
    by the backoff library.
    """
    max_time: Duration = '3m'
    max_tries: int = 12


class Webhook(servo.BaseConfiguration):
    name: Optional[str] = Field(
        description="A unique name identifying the webhook.",
    )
    description: Optional[str] = Field(
        description="Optional free-form text describing the context or purpose of the webhook.",
    )
    events: List[str] = Field(
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

    # Map strings from config into EventContext objects
    _validate_events = validator("events", pre=True, allow_reuse=True)(validate_event_contexts)

class WebhooksConfiguration(servo.AbstractBaseConfiguration):
    __root__: List[Webhook] = []

    @classmethod
    def generate(cls, **kwargs) -> "WebhooksConfiguration":
        return cls(
            __root__=[
                Webhook(
                    name="My Webhook",
                    description="Listens for after:measure events and sends an email",
                    events=["after:measure"],
                    url="https://example.com/",
                    secret="s3cr3t!"
                )
            ],
            **kwargs,
        )
    
    @property
    def webhooks(self) -> List[Webhook]:
        """
        Convenience method for retrieving the root type as a list of webhooks.
        """
        return self.__root__


class Result(BaseModel):
    """Models an EventResult webhook representation"""
    connector: str
    value: Any


class RequestBody(BaseModel):
    """Models the JSON body of a webhook request containing event results"""
    event: str
    created_at: datetime
    results: List[Result]


@metadata(
    name=("servo-webhooks", "webhooks"),
    description="Dispatch servo events via HTTP webhooks",
    version=__version__,
    homepage="https://github.com/opsani/servo-webhooks",
    license=License.APACHE2,
    maturity=Maturity.EXPERIMENTAL,
)
class WebhooksConnector(servo.BaseConnector):
    config: WebhooksConfiguration

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._register_event_handlers()
    
    def _register_event_handlers(self) -> None:
        for webhook in self.config.webhooks:
            for event_name in webhook.events:
                event = EventContext.from_str(event_name)
                if not event:
                    raise ValueError(f"invalid webhook event '{event_name}'")
                if event.preposition == Preposition.BEFORE:
                    self._add_before_event_webhook_handler(webhook, event)
                elif event.preposition == Preposition.AFTER:
                    self._add_after_event_webhook_handler(webhook, event)
                else:
                    raise ValueError(f"Unsupported Preposition value given for webhook: '{event.preposition}'")

    def _add_before_event_webhook_handler(self, webhook: Webhook, event: EventContext) -> None:
        async def __before_handler(self) -> None:
            headers = {**webhook.headers, **{ "Content-Type": CONTENT_TYPE }}
            async with httpx.AsyncClient(headers=headers) as client:
                response = await client.post(webhook.url, data=dict(foo="bar"))
                success = (response.status_code == httpx.codes.OK)

        self.add_event_handler(event.event, event.preposition, __before_handler)
        
    def _add_after_event_webhook_handler(self, webhook: Webhook, event: EventContext) -> None:
        async def __after_handler(self, results: List[EventResult], **kwargs) -> None:
            headers = {**webhook.headers, **{ "Content-Type": CONTENT_TYPE }}

            outbound_results = []
            for result in results:
                outbound_results.append(
                    Result(
                        connector=result.connector.name,
                        value=result.value
                    )
                )
            body = RequestBody(
                event=str(event),
                created_at=datetime.now(),
                results=outbound_results
            )
            
            json_body = body.json()
            headers["X-Servo-Signature"] = self._signature_for_webhook_body(webhook, json_body)
            async with httpx.AsyncClient(headers=headers) as client:
                response = await client.post(webhook.url, data=json_body, headers=headers)
                success = (response.status_code in SUCCESS_STATUS_CODES)
                if success:                    
                    self.logger.success(f"posted webhook for '{event}' event to '{webhook.url}' ({response.status_code} {response.reason_phrase})")
                else:
                    self.logger.error(f"failed posted webhook for '{event}' event to '{webhook.url}' ({response.status_code} {response.reason_phrase}): {response.text}")

        self.add_event_handler(event.event, event.preposition, __after_handler)

    def _signature_for_webhook_body(self, webhook: Webhook, body: str) -> str:
        secret_bytes = webhook.secret.get_secret_value().encode()
        return str(hmac.new(secret_bytes, body.encode(), hashlib.sha1).hexdigest())

class CLI(servo.cli.ConnectorCLI):
    pass
    # add, remove, test
