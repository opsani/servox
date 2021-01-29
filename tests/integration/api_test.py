import pathlib
import dotenv
import pytest
import pydantic
import servo
import mechanism
import hypothesis
import json

from typing import Any, Dict, List, Optional
# Let's torture OCO!

# pytestmark = [pytest.mark.asyncio, pytest.mark.integration, pytest.mark.opsani_dotenv(skippable=True)] # TODO: maybe opsani_optimizer? pytest.mark.optimizer(skippable=True)
# pytest.mark.requires_optimizer(skippable=True), pytest.mark.optimizer('dev.opsani.com/blah', token)... pytest.mark.optimizer(skippable=True, dotenv=False)

# pytestmark = [pytest.mark.asyncio, pytest.mark.integration, pytest.mark.optimizer(skippable=True)] # TODO: maybe opsani_optimizer? pytest.mark.optimizer(skippable=True)
import abc
from typing import Tuple


class BaseErrorResponse(pydantic.BaseModel, abc.ABC):
    """Base class for representing errors to clients.

    Error response objects provide a canonical representation of runtime
    error conditions for HTTP transport. Responses are expected to provide
    clear, human readable information and metadata for programmatic processing.
    """
    error: Exception = pydantic.Field(..., description='The error that was raised.')
    type: str = pydantic.Field(None, description='The type of error that was raised.')
    message: str = pydantic.Field(None, description='A human readable description of the error.')
    metadata: Dict[str, str] = pydantic.Field({}, description='Supplementary metadata about the error context.')
    status_code: int = pydantic.Field(..., description='The HTTP status code for error response.')

    @pydantic.validator('type', always=True, pre=True)
    def _set_type(cls, _, values: Dict[str, Any]) -> str:
        return values.get('error').__class__.__name__

    @pydantic.validator('message', always=True, pre=True)
    def _set_default_message(cls, message: Optional[str], values: Dict[str, Any]) -> str:
        return message or str(values.get('error'))

    def json(self) -> str:
        """Return a JSON string representation of the error response content.

        The `status_code` and `error` fields are excluded from the representation returned.
        """
        return super().json(exclude={'status_code', 'error'}, indent=2)

    def http(self) -> Tuple[int, Dict[str, str], str]:
        """Return a tuple of the status code, headers, and content body for HTTP transport.

        Content is serialized as JSON with a content type header of "application/json".

        This is a convenience method supporting integration of error responses with arbitrary
        HTTP servers.
        """
        return (self.status_code, { "content-type": "application/json" }, self.json())

    class Config:
        allow_mutation = False
        arbitrary_types_allowed = True
        allow_population_by_field_name = True
        json_encoders = {
            Exception: lambda v: v.__class__.__name__,
        }


class RequestErrorResponse(BaseErrorResponse):
    """A generic response for arbitrary request errors.

    Request errors are represented as HTTP responses in the 4xx status code range
    with a default value of `400 (Bad Request)`.
    """
    status_code: pydantic.conint(ge=400, lt=500, strict=True) = pydantic.Field(
        400,
        description='The HTTP status code for error response.'
    )


class FieldErrorDescriptor(pydantic.BaseModel):
    """A description of a specific validation error of a model field."""
    type: str = pydantic.Field(
        description='The type of validation failure affecting the field.'
    )
    message: str = pydantic.Field(
        alias='msg',
        description='A description of the validation failure.'
    )
    attributes: Tuple[str] = pydantic.Field(
        alias='loc',
        description='The model attributes that are in an invalid state.'
    )


class ModelValidationErrorResponse(BaseErrorResponse):
    """A response describing a data model validation error.

    Validation errors are represented as HTTP responses with the `422 (Unprocessable Entity)`
    status code. A model validation error may contain an arbitrary number of underlying
    field errors, describing the invalid state of one or more attributes of the model.
    """
    error: pydantic.ValidationError
    status_code: int = pydantic.Field(422, const=True, description='HTTP Status Code 422 (Unprocessable entity)')
    model: str = pydantic.Field(..., description='Name of the model type that has failed validation.')
    errors: List[FieldErrorDescriptor] = pydantic.Field(..., description='List of underlying validation errors for model fields.')

    @pydantic.validator('message', always=True, pre=True)
    def _set_default_message(cls, message: Optional[str], values: Dict[str, Any]) -> str:
        if not message:
            error = values.get('error')
            if isinstance(error, pydantic.ValidationError):
                message = f'{error.model.__name__} failed validation ({len(error.errors())} field errors)'

        return message

    @pydantic.root_validator(pre=True)
    def _parse_error(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        error = values.get('error')
        if isinstance(error, pydantic.ValidationError):
            values['model'] = error.model.__name__
            values['errors'] = error.errors()

        return values


class ServerErrorResponse(BaseErrorResponse):
    """A generic response for arbitrary server errors.

    Server errors are represented as HTTP responses in the 5xx status code range
    with a default value of `500 (Internal Server Error)`.
    """
    status_code: pydantic.conint(ge=500, lt=600, strict=True) = 500

# TODO: parametrize this: no message, override message... model, message

# @pytest.parametrize(
#     [
#         ('model', 'exception', 'args')
#     ]
# )

def test_set_default_message() -> None:
    error = RuntimeError('wtf?')
    assert str(error) == 'wtf?'
    response = RequestErrorResponse(error=error)
    assert response.message == 'wtf?'

def test_set_explicit_message() -> None:
    error = RuntimeError('wtf?')
    response = RequestErrorResponse(error=error, message="This is the message")
    assert response.message == 'This is the message'


# @pytest.parametrize(
#     [
#         ('model', 'args', 'http')
#     ]
# )

def test_error_serialization() -> None:
    response = RequestErrorResponse(error=RuntimeError('Something bad happened.'))
    try:
        servo.RangeSetting()
    except pydantic.ValidationError as error:
        response = ModelValidationErrorResponse(error=error)
        status_code, headers, content = response.http()
        assert status_code == 422
        assert headers['content-type'] == 'application/json'
        assert content == (
            '{\n'
            '  "type": "ValidationError",\n'
            '  "message": "RangeSetting failed validation (4 field errors)",\n'
            '  "metadata": {},\n'
            '  "model": "RangeSetting",\n'
            '  "errors": [\n'
            '    {\n'
            '      "type": "value_error.missing",\n'
            '      "message": "field required",\n'
            '      "attributes": [\n'
            '        "name"\n'
            '      ]\n'
            '    },\n'
            '    {\n'
            '      "type": "value_error.missing",\n'
            '      "message": "field required",\n'
            '      "attributes": [\n'
            '        "min"\n'
            '      ]\n'
            '    },\n'
            '    {\n'
            '      "type": "value_error.missing",\n'
            '      "message": "field required",\n'
            '      "attributes": [\n'
            '        "max"\n'
            '      ]\n'
            '    },\n'
            '    {\n'
            '      "type": "value_error.missing",\n'
            '      "message": "field required",\n'
            '      "attributes": [\n'
            '        "step"\n'
            '      ]\n'
            '    }\n'
            '  ]\n'
            '}'
        )

class APITestClient(pydantic.BaseModel, servo.api.Mixin):
    optimizer: servo.Optimizer

    @property
    def api_client_options(self) -> Dict[str, Any]:
        return {
            "base_url": self.optimizer.api_url,
            "headers": {
                "Authorization": f"Bearer {self.optimizer.token}",
                "User-Agent": servo.api.USER_AGENT,
                "Content-Type": "application/json",
            },
            "timeout": None
        }


@pytest.fixture
def client(mechanism: mechanism.Fixture) -> APITestClient:
    servo.logging.set_level("TRACE")
    return APITestClient(
        optimizer=mechanism.dotenv.optimizer
    )


async def test_describe_valid(client: APITestClient) -> None:
    cmd_response = await client._post_event(servo.api.Events.hello, dict(agent=servo.api.USER_AGENT))
    debug("CMD repsonse: ", cmd_response)

    cmd_response = await client._post_event(servo.api.Events.whats_next, None)
    debug("CMD repsonse: ", cmd_response)

    description = servo.Description(
        metrics=[
            servo.Metric("throughput", servo.Unit.requests_per_minute)
        ],
        components=[
            servo.Component(
                "main",
                settings=[
                    servo.CPU(
                        min=1,
                        max=5,
                        value=1
                    ),
                ]
            )
        ]
    )

    status = servo.api.Status.ok(descriptor=description.__opsani_repr__())
    cmd_response = await client._post_event(servo.api.Events.describe, status.dict())
    debug("CMD repsonse: ", cmd_response)
    # Try without name, try without value, try without, empty metrics, empty components... component with empty settings... out of order...
    ...

async def test_describe_with_empty_metrics(client: APITestClient) -> None:
    cmd_response = await client._post_event(servo.api.Events.hello, dict(agent=servo.api.USER_AGENT))
    debug("CMD repsonse: ", cmd_response)

    cmd_response = await client._post_event(servo.api.Events.whats_next, None)
    debug("CMD repsonse: ", cmd_response)

    description = servo.Description(
        metrics=[],
        components=[
            servo.Component(
                "main",
                settings=[
                    servo.CPU(
                        min=1,
                        max=5,
                        value=1
                    ),
                ]
            )
        ]
    )

    status = servo.api.Status.ok(descriptor=description.__opsani_repr__())
    cmd_response = await client._post_event(servo.api.Events.describe, status.dict())
    debug("CMD repsonse: ", cmd_response)
    # Try without name, try without value, try without, empty metrics, empty components... component with empty settings... out of order...
    ...

async def test_describe_with_empty_components_and_metrics(client: APITestClient) -> None:
    # TODO: describe without requests/limits

    cmd_response = await client._post_event(servo.api.Events.hello, dict(agent=servo.api.USER_AGENT))
    debug("CMD repsonse: ", cmd_response)

    cmd_response = await client._post_event(servo.api.Events.whats_next, None)
    debug("CMD repsonse: ", cmd_response)

    # self.logger.info(f"What's Next? => {cmd_response.command}")
    description = servo.Description(
        metrics=[],
        components=[]
    )

    status = servo.api.Status.ok(descriptor=description.__opsani_repr__())
    cmd_response = await client._post_event(servo.api.Events.describe, status.dict())
    debug("CMD repsonse: ", cmd_response)
    # Try without name, try without value, try without, empty metrics, empty components... component with empty settings... out of order...
    ...

async def test_describe_without_descriptor(client: APITestClient) -> None:
    # TODO: This should not be a 200 status code
    status = await client._post_event(servo.api.Events.describe, None)
    assert status.status == servo.api.OptimizerStatuses.invalid
    assert status.reason == "describe event without 'descriptor' in the data"

async def test_describe_with_null_metrics(client: APITestClient) -> None:
    # {
    #     'status': <OptimizerStatuses.ok: 'ok'>,
    #     'message': None,
    #     'reason': 'success',
    #     'descriptor': {
    #         'application': {
    #             'components': {},
    #         },
    #         'measurement': {
    #             'metrics': {},
    #         },
    #     },
    # }
    description = servo.Description(
        metrics=[],
        components=[]
    )
    status = servo.api.Status.ok(descriptor=description.__opsani_repr__())
    status_dict = status.dict()
    debug("status_dict=", status_dict) # TODO: Could I use exclude for this?
    status_dict['descriptor']['measurement']['metrics'] = None
    status = await client._post_event(servo.api.Events.describe, status_dict)
    assert status.status == servo.api.OptimizerStatuses.invalid
    assert status.reason == (
        '1 validation error for ServoDescriptor\n'
        'measurement -> metrics\n'
        '  none is not an allowed value (type=type_error.none.not_allowed)'
    )
    # Try without name, try without value, try without, empty metrics, empty components... component with empty settings... out of order...
    ...

# TODO: Start one connection then show the weird ass read timeout. The second one connects and blocks, eventually getting a sleep. You don't know why
# TODO: It will eventually timeout -- the default timeout of 5 seconds is probably too low.
# Something is wrong but you don't know what it is.


async def test_describe_component_with_no_settings(client: APITestClient) -> None:
    description = servo.Description(
        metrics=[],
        components=[
            servo.Component(
                "main",
                settings=[
                    # servo.CPU(
                    #     min=1,
                    #     max=5
                    # ),
                    # servo.Memory(
                    #     min=0.25,
                    #     max=8.0,
                    #     step=0.125
                    # ),
                    # servo.Replicas(
                    #     min=1,
                    #     max=10
                    # )
                ]
            )
        ]
    )
    status = servo.api.Status.ok(descriptor=description.__opsani_repr__())
    status = await client._post_event(servo.api.Events.describe, status.dict())
    assert status.status == servo.api.OptimizerStatuses.invalid
    assert status.reason == (
        '1 validation error for ServoDescriptor\n'
        'measurement -> metrics\n'
        '  none is not an allowed value (type=type_error.none.not_allowed)'
    )

async def test_describe_component_setting_incomplete(client: APITestClient) -> None:
    response = await client._post_event(servo.api.Events.hello, dict(agent=servo.api.USER_AGENT))
    debug("HELLO repsonse: ", response)

    cmd_response = await client._post_event(servo.api.Events.whats_next, None)
    debug("WHATS_NEXT repsonse: ", cmd_response)

    # NOTE: This is returning a 400 Bad Request
#     ❯ curl -X POST -H "host: api.opsani.com" -H "accept: */*" -H "accept-encoding: gzip, deflate" -H "connection: keep-alive" -H "user-agent: github.com/opsani/servox" -H "authorization: Bearer 452361d3-48e0-41df-acec-0fe2be826cb8" -H "content-type: application/json" -H "content-length: 294" -d '{"event": "DESCRIPTION", "param": {"status": "ok", "message": null, "reason": "success", "descriptor": {"application": {"components": {"fake-app": {"settings": {"cpu": {"type": "range", "pinned": false, "value": 1.0, "min": 1.0, "max": 5.0, "step": 0.125}}}}}, "measurement": {"metrics": {}}}}}' https://api.opsani.com/accounts/dev.opsani.com/applications/leviathan-wakes/servo
# {"status": "400 Bad Request", "message": "must be real number, not NoneType", "version": "18.1.2"}
    description = servo.Description(
        metrics=[],
        components=[
            servo.Component(
                "fake-app",
                settings=[
                    servo.CPU(
                        min=1,
                        max=5,
                        # value=1
                    ),
                ]
            )
        ]
    )
    status = servo.api.Status.ok(descriptor=description.__opsani_repr__())
    status = await client._post_event(servo.api.Events.describe, status.dict())
    assert status.status == servo.api.OptimizerStatuses.invalid
    assert status.reason == (
        '1 validation error for ServoDescriptor\n'
        'measurement -> metrics\n'
        '  none is not an allowed value (type=type_error.none.not_allowed)'
    )


# TODO: This now throwing a 500
# curl -X POST -H "host: api.opsani.com" -H "accept: */*" -H "accept-encoding: gzip, deflate" -H "connection: keep-alive" -H "user-agent: github.com/opsani/servox" -H "authorization: Bearer 452361d3-48e0-41df-acec-0fe2be826cb8" -H "content-type: application/json" -H "content-length: 66" -d '{"event": "HELLO", "param": {"agent": "github.com/opsani/servox"}}' https://api.opsani.com/accounts/dev.opsani.com/applications/leviathan-wakes/servo




# This series of steps will wedge OCO:
# Say HELLO: curl -X POST -H "host: api.opsani.com" -H "accept: */*" -H "accept-encoding: gzip, deflate" -H "connection: keep-alive" -H "user-agent: github.com/opsani/servox" -H "authorization: Bearer 452361d3-48e0-41df-acec-0fe2be826cb8" -H "content-type: application/json" -H "content-length: 66" -d '{"event": "HELLO", "param": {"agent": "github.com/opsani/servox"}}' https://api.opsani.com/accounts/dev.opsani.com/applications/leviathan-wakes/servo
# Ask What's Next?: curl -X POST -H "host: api.opsani.com" -H "accept: */*" -H "accept-encoding: gzip, deflate" -H "connection: keep-alive" -H "user-agent: github.com/opsani/servox" -H "authorization: Bearer 452361d3-48e0-41df-acec-0fe2be826cb8" -H "content-type: application/json" -H "content-length: 38" -d '{"event": "WHATS_NEXT", "param": null}' https://api.opsani.com/accounts/dev.opsani.com/applications/leviathan-wakes/servo
# Describe with a null value: curl -X POST -H "host: api.opsani.com" -H "accept: */*" -H "accept-encoding: gzip, deflate" -H "connection: keep-alive" -H "user-agent: github.com/opsani/servox" -H "authorization: Bearer 452361d3-48e0-41df-acec-0fe2be826cb8" -H "content-type: application/json" -H "content-length: 294" -d '{"event": "DESCRIPTION", "param": {"status": "ok", "message": null, "reason": "success", "descriptor": {"application": {"components": {"fake-app": {"settings": {"cpu": {"type": "range", "pinned": false, "value": null, "min": 1.0, "max": 5.0, "step": 0.125}}}}}, "measurement": {"metrics": {}}}}}' https://api.opsani.com/accounts/dev.opsani.com/applications/leviathan-wakes/servo
# Describe with a complete value: curl -X POST -H "host: api.opsani.com" -H "accept: */*" -H "accept-encoding: gzip, deflate" -H "connection: keep-alive" -H "user-agent: github.com/opsani/servox" -H "authorization: Bearer 452361d3-48e0-41df-acec-0fe2be826cb8" -H "content-type: application/json" -H "content-length: 294" -d '{"event": "DESCRIPTION", "param": {"status": "ok", "message": null, "reason": "success", "descriptor": {"application": {"components": {"fake-app": {"settings": {"cpu": {"type": "range", "pinned": false, "value": 1.0, "min": 1.0, "max": 5.0, "step": 0.125}}}}}, "measurement": {"metrics": {}}}}}' https://api.opsani.com/accounts/dev.opsani.com/applications/leviathan-wakes/servo

# curl -X POST -H "host: api.opsani.com" -H "accept: */*" -H "accept-encoding: gzip, deflate" -H "connection: keep-alive" -H "user-agent: github.com/opsani/servox" -H "authorization: Bearer 452361d3-48e0-41df-acec-0fe2be826cb8" -H "content-type: application/json" -H "content-length: 294" -d '{"event": "DESCRIPTION", "param": {"status": "ok", "message": null, "reason": "success", "descriptor": {"application": {"components": {"fake-app": {"settings": {"cpu": {"type": "range", "pinned": false, "value": 1.0, "min": 1.0, "max": 5.0, "step": 0.125}}}}}, "measurement": {"metrics": {}}}}}' https://api.opsani.com/accounts/dev.opsani.com/applications/leviathan-wakes/servo
