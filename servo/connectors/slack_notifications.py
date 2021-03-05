"""Provides ready to roll integration with slack web api and webhooks from minimal configuration
"""
from __future__ import annotations

from typing import List, Optional, Union

import abc
import backoff
import pydantic

from aiohttp.client_exceptions import ClientError

from slack_sdk.errors import SlackApiError
from slack_sdk.web.async_client import AsyncWebClient
from slack_sdk.webhook.async_client import AsyncWebhookClient

import servo
from servo.events import EventContext

class SlackBlockText(pydantic.BaseModel):
    type: str = "mrkdwn"
    text: str

class SlackBlock(pydantic.BaseModel):
    type: str = "section"
    text: SlackBlockText

    @pydantic.validator("text", pre=True)
    def coerce_str_to_block_text(cls, v):
        if isinstance(v, str):
            return SlackBlockText(text=v)
        return v

class BaseSlackNotifier(servo.BaseConfiguration, abc.ABC):
    name: Optional[str] = None
    """A unique name identifying the notifier.
    """

    events: List[str]
    """A list of events that the notifier is listening for.
    """

    @abc.abstractmethod
    def init_client(self) -> None:
        ...

    @abc.abstractmethod
    async def send(self, blocks: List[str], text: str) -> None:
        ...

    _client: Union[AsyncWebClient, AsyncWebhookClient] = pydantic.PrivateAttr(None)


class SlackIncomingWebhookNotifier(BaseSlackNotifier):
    """Configuration for sending slack notifications by Incoming Webhook
    """

    url: pydantic.SecretStr
    """A Slack Incoming Webhook URL used to post messages
    """

    def init_client(self) -> None:
        self._client = AsyncWebhookClient(url=self.url.get_secret_value())

    async def send(self, blocks: List[str], text: str) -> None:
        return await self._client.send(blocks=blocks, text=text)


class SlackWebApiNotifier(BaseSlackNotifier):
    """Configuration for sending slack notifications by Web API
    """

    channel_id: str
    """ID of slack channel to which notifications should be posted
    """

    bot_token: pydantic.SecretStr
    """Slack App bot user's OAuth Access Token
    """

    # TODO? ability to delete previous message (best effort) or scan target channel for previous messages to remove and/or update

    def init_client(self) -> None:
        self._client = AsyncWebClient(token=self.bot_token.get_secret_value())

    async def send(self, blocks: List[str], text: str) -> None:
        return await self._client.chat_postMessage(
            blocks=blocks,
            channel=self.channel_id,
            text=text,
        )


class SlackNotificationsConfiguration(servo.AbstractBaseConfiguration):
    __root__: List[Union[SlackWebApiNotifier, SlackIncomingWebhookNotifier]] = []

    @classmethod
    def generate(cls, **kwargs) -> "SlackNotificationsConfiguration":
        return cls(
            __root__=[
                SlackIncomingWebhookNotifier(
                    name="Adjust Notify",
                    description="Listens for adjust events and notifies a slack channel on before start and on completion",
                    events=["adjust"],
                    url="https://hooks.slack.com/services/T00000000/B00000000/XXXXXXXXXXXXXXXXXXXXXXXX",
                )
            ]
        )

    @property
    def slack_notifiers(self):
        """
        Convenience method for retrieving the root type as a list of slack notifiers.
        """
        return self.__root__

def fatal_slack_error(e: SlackApiError):
    return e.response["error"] != "rate_limit"

@backoff.on_exception(backoff.expo, SlackApiError, max_tries=3, giveup=fatal_slack_error)
async def _send_with_backoff(notifier: BaseSlackNotifier, blocks: List[SlackBlock], text: str):
    return await notifier.send(blocks=blocks, text=text)

@servo.metadata(
    description="Slack notifications integration connector",
    version="0.0.1",
    homepage="https://github.com/opsani/servo-slack-notifications",
    license=servo.License.apache2,
    maturity=servo.Maturity.experimental,
)
class SlackNotificationsConnector(servo.BaseConnector):
    config: SlackNotificationsConfiguration

    @servo.on_event()
    async def attach(self, servo_: servo.Servo) -> None:
        for notifier in self.config.slack_notifiers:
            # Ensure we are ready to talk to the Slack API
            notifier.init_client()

            for event_name in notifier.events:
                event = EventContext.from_str(event_name)
                if not event:
                    raise ValueError(f"invalid notification event '{event_name}'")
                if event.preposition == servo.Preposition.on:
                    event.preposition = servo.Preposition.all

                if event.preposition & servo.Preposition.before:
                    self._add_before_event_slack_notifier(notifier, event)
                if event.preposition & servo.Preposition.after:
                    self._add_after_event_slack_notifier(notifier, event)
                if not event.preposition & servo.Preposition.all:
                    raise ValueError(f"Unsupported Preposition value given for webhook: '{event.preposition}'")

    def _add_before_event_slack_notifier(self, notifier: BaseSlackNotifier, event: EventContext) -> None:
        block_text = text = f"Servo started the {event.event.name} event!"
        if self.optimizer:
            console_url = f"https://console.opsani.com/accounts/{self.optimizer.org_domain}/applications/{self.optimizer.app_name}"
            block_text = f"{block_text} Check out its progress on the <{console_url}|Opsani Dashboard>"

        async def __before_handler(self) -> None:
            blocks = [SlackBlock(text=block_text).dict()] 
            try:
                await _send_with_backoff(notifier, blocks, text)
            except (SlackApiError, ClientError) as e:
                self.logger.opt(exception=e).trace(f"Unable to send before {event.event.name} event notification to slack")

        self.add_event_handler(event.event, servo.Preposition.before, __before_handler)

    def _add_after_event_slack_notifier(self, notifier: BaseSlackNotifier, event: EventContext) -> None:
        if self.optimizer:
            console_url = f"https://console.opsani.com/accounts/{self.optimizer.org_domain}/applications/{self.optimizer.app_name}"

        async def __after_handler(self, results: List[servo.EventResult]) -> None:
            block_text = text = f"{len(results)} Servo connectors completed the {event.event.name} event!"
            errors = list(filter(lambda r: isinstance(r.value, Exception), results))
            if errors:
                block_text = f"{block_text} {len(errors)} connectors had an error result." # TODO: emoji
                if console_url:
                    block_text = f"{block_text} Check out the <{console_url}/logs|Opsani Console Logs> for more information"

            blocks = [SlackBlock(text=block_text).dict()] 
            try:
                await _send_with_backoff(notifier, blocks, text)
            except (SlackApiError, ClientError) as e:
                self.logger.opt(exception=e).trace(f"Unable to send after {event.event.name} event notification to slack")

        self.add_event_handler(event.event, servo.Preposition.after, __after_handler)
