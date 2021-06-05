import datetime
import functools
from typing import Any, Dict, List, Literal, Optional, Type, Union

import httpx
import pydantic

import servo

from .models import *

API_PATH = "/api/v1"

def _rstrip_slash(cls, base_url):
    return base_url.rstrip("/")

class Client(pydantic.BaseModel):
    """A high level interface for interacting with the Prometheus HTTP API.

    The client supports instant and range queries and retrieving the targets.
    Requests and responses are serialized through an object model to make working
    with Prometheus fast and ergonomic.

    For details about the Prometheus HTTP API see: https://prometheus.io/docs/prometheus/latest/querying/api/

    ### Attributes:
        base_url: The base URL for connecting to Prometheus.
    """
    base_url: pydantic.AnyHttpUrl
    _normalize_base_url = pydantic.validator('base_url', allow_reuse=True)(_rstrip_slash)

    @property
    def url(self) -> str:
        """Return the full URL for accessing the Prometheus API."""
        return f"{self.base_url}{API_PATH}"

    async def query(
        self,
        promql: Union[str, PrometheusMetric],
        time: Optional[datetime.datetime] = None,
        *,
        timeout: Optional[servo.DurationDescriptor] = None,
        method: Literal['GET', 'POST'] = 'GET'
    ) -> BaseResponse:
        """Send an instant query to Prometheus for evaluation and return the response.

        Instant queries return the result of a query at a moment in time.
        https://prometheus.io/docs/prometheus/latest/querying/api/#instant-queries

        ### Args:
            promql: A PromQL query string or PrometheusMetric object to query for.
            time: An optional time to evaluate the query at. When `None`, evaluate
                the query expression at the time it was received.
            timeout: Evaluation timeout for the query.
        """
        if isinstance(promql, PrometheusMetric):
            response_type = functools.partial(MetricResponse, metric=promql)
            promql_ = promql.build_query()
        elif isinstance(promql, str):
            response_type = BaseResponse
            promql_ = promql
        else:
            raise TypeError(f"cannot query for type: '{promql.__class__.__name__}'")

        query = InstantQuery(
            query=promql_,
            time=time,
        )
        return await self.send_request(method, query, response_type)

    async def query_range(
        self,
        promql: Union[str, PrometheusMetric],
        start: datetime.datetime,
        end: datetime.datetime,
        step: servo.Duration = None,
        *,
        timeout: Optional[servo.DurationDescriptor] = None,
        method: Literal['GET', 'POST'] = 'GET'
    ) -> BaseResponse:
        """Send a range query to Prometheus for evaluation and return the response."""
        if isinstance(promql, PrometheusMetric):
            promql_ = promql.build_query()
            step_ = step or promql.step
            response_type = functools.partial(MetricResponse, metric=promql)
        elif isinstance(promql, str):
            promql_ = promql
            step_ = step
            response_type = BaseResponse
        else:
            raise TypeError(f"cannot query for type: '{promql.__class__.__name__}'")

        query = RangeQuery(
            query=promql_,
            start=start,
            end=end,
            step=step_,
            timeout=timeout,
        )
        return await self.send_request(method, query, response_type)

    async def list_targets(self, state: Optional[TargetsStateFilter] = None) -> TargetsResponse:
        """List the targets discovered by Prometheus.

            ### Args:
                state: Optionally filter by active or dropped target state.
        """
        return await self.send_request("GET", TargetsRequest(state=state), TargetsResponse)

    async def check_is_metric_absent(self, queryable: Union[str, PrometheusMetric]) -> bool:
        """Check if the metric referenced in a Prometheus expression is absent."""
        query = (
            f"absent({queryable.query})" if isinstance(queryable, PrometheusMetric)
            else f"absent({queryable})"
        )
        response = await self.query(query)
        servo.logger.debug(f"Absent metric introspection returned {query}: {response}")
        if response.data:
            if response.data.result_type != servo.connectors.prometheus.ResultType.vector:
                raise TypeError(f"expected a vector result but found {response.data.result_type}")
            if len(response.data) != 1:
                raise ValueError(f"expected a single result vector but found {len(response.data)}")
            result = next(iter(response.data))
            return int(result.value[1]) == 1

        else:
            servo.logger.info(f"Metric '{query}' is present in Prometheus but returned an empty result set")
            return False

    async def send_request(
        self,
        method: Literal['GET', 'POST'],
        request: QueryRequest,
        response_type: Type[BaseResponse] = BaseResponse
    ) -> BaseResponse:
        """Send a request to the Prometheus HTTP API and return the response.

        ### Args:
            method: The HTTP method to use when sending the request.
            request: An object describing a request to the Prometheus HTTP API.
            response_type: The type of object to parse the response into. Must be `Response` or a subclass thereof.
        """
        servo.logger.trace(
            f"Sending request to Prometheus HTTP API (`{request}`): {method} {request.endpoint}"
        )
        async with httpx.AsyncClient(base_url=self.url) as client:
            try:
                kwargs = (
                    dict(params=request.params) if method == 'GET'
                    else dict(data=request.params)
                )
                http_request = client.build_request(method, request.endpoint, **kwargs)
                http_response = await client.send(http_request)
                http_response.raise_for_status()
                return response_type(request=request, **http_response.json())
            except (
                httpx.HTTPError,
                httpx.ReadTimeout,
                httpx.ConnectError,
            ) as error:
                servo.logger.trace(
                    f"HTTP error encountered during GET {request.url}: {error}"
                )
                raise
