import abc
import datetime
import enum
import itertools
import operator
import re
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

import httpx
import pydantic

import servo

class AbsentMetricPolicy(str, enum.Enum):
    """An enumeration of behaviors for handling absent metrics.

    Absent metrics do not exist in Prometheus at query evaluation time.
    Metrics can be absent because no target scraped has reported a value
    or due to previously reported data being lost or purged.

    ### Members:
        ignore: Silently ignore the absent metric and continue processing.
        zero: Return a vector of zero for the absent metric.
        warn: Log a warning message when an absent metric is encountered.
        fail: Raise a runtime exception when an absent metric is encountered.
    """
    ignore = "ignore"
    zero = "zero"
    warn = "warn"
    fail = "fail"

class PrometheusMetric(servo.Metric):
    """A metric that can be measured by querying Prometheus.

    ### Attributes:
        query: A PromQL query string that returns the value of the target metric.
            For details on PromQL, see https://prometheus.io/docs/prometheus/latest/querying/basics/.
        step: The time resolution offset between data points in a range query.
            The number of data points within the query result is equal to the duration between the
            start and end times divided by the step. May be a numeric value or Golang duration string.
        absent: The behavior to apply when the queried metric is absent.
        eager: The duration to observe the metric and eagerly return a measurement if it does not change.
            Defaults to `None`, disabling eager measurements.
    """
    query: str = None
    step: servo.Duration = "1m"
    absent: AbsentMetricPolicy = AbsentMetricPolicy.ignore
    eager: Optional[servo.Duration] = None

    def build_query(self) -> str:
        """Build and return a complete Prometheus query string.

        The current implementation handles appending the zero vector suffix.
        """
        if self.absent == AbsentMetricPolicy.zero:
            return self.query + " or on() vector(0)"
        return self.query

    @property
    def escaped_query(self) -> str:
        return re.sub(r"\{(.*?)\}", r"{{\1}}", self.query)

    def __check__(self) -> servo.Check:
        """Return a Check representation of the metric."""
        return servo.Check(
            name=f"Check {self.name}",
            description=f'Run Prometheus query "{self.query}"',
        )


class ActiveTarget(pydantic.BaseModel):
    """An active endpoint exporting metrics being scraped by Prometheus.

    ### Attributes:
        pool: The group of related targets that the target belongs to.
        url: The URL that metrics were scraped from.
        global_url: An externally accessible URL to the target.
        health: Health status ("up", "down", or "unknown").
        labels: The set of labels after relabelling has occurred.
        discovered_labels: The unmodified set of labels discovered during service
            discovery before relabelling has occurred.
        last_scraped_at: Time that the target was last scraped by Prometheus.
        last_scrape_duration: The amount of time that the last scrape took to complete.
        last_error: The last error that occurred during scraping (if any).
    """
    pool: str = pydantic.Field(..., alias='scrapePool')
    url: str = pydantic.Field(..., alias='scrapeUrl')
    global_url: str = pydantic.Field(..., alias='globalUrl')
    health: Literal['up', 'down', 'unknown']
    labels: Optional[Dict[str, str]]
    discovered_labels: Optional[Dict[str, str]] = pydantic.Field(..., alias='discoveredLabels')
    last_scraped_at: Optional[datetime.datetime] = pydantic.Field(..., alias='lastScrape')
    last_scrape_duration: Optional[servo.Duration] = pydantic.Field(..., alias='lastScrapeDuration')
    last_error: Optional[str] = pydantic.Field(..., alias='lastError')

    def is_healthy(self) -> bool:
        """Return True if the target is healthy."""
        return self.health == 'up'


class DroppedTarget(pydantic.BaseModel):
    """A target that was discovered by Prometheus and then dropped.

    Dropped targets only have the `discovered_labels` attribute.

    ### Attributes:
        discovered_labels: The set of labels discovered about the dropped target.
    """
    discovered_labels: Optional[Dict[str, str]] = pydantic.Field(..., alias='discoveredLabels')


class BaseRequest(pydantic.BaseModel, abc.ABC):
    """Abstract base class for Prometheus HTTP API request types.

    The base class handles serialization of parameters for subclasses.
    The `param_attrs` attribute must return a sequence of attribute names
    that are to be serialized into request parameters.
    """
    endpoint: str
    param_attrs: Sequence[str]

    @property
    def params(self) -> Dict[str, str]:
        """Return the dictionary of parameters for the query request.

        The values serialized as parameters is determined by the sequence
            of attribute names returned by param_attrs attribute.
        """
        def _param_for_attr(attr: str) -> Optional[Tuple[str, str]]:
            value = getattr(self, attr)
            if not value:
                return None
            elif isinstance(value, datetime.datetime):
                value = value.timestamp()
            return (attr, str(value))

        return dict(filter(None, map(_param_for_attr, self.param_attrs)))

    @property
    def url(self) -> httpx.URL:
        """The relative URL for sending the request as an HTTP GET."""
        return httpx.URL(self.endpoint, params=self.params)


class TargetsStateFilter(str, enum.Enum):
    """An enumeration of states to filter Prometheus targets by."""
    active = 'active'
    dropped = 'dropped'
    any = 'any'


class TargetsRequest(BaseRequest):
    """A request for retrieving targets from the Prometheus HTTP API.

    ### Attributes:
        endpoint: A constant value of `/targets`.
        state: Target state to optionally filter by. One of
            `active`, `dropped`, or `any`.
    """
    endpoint: str = pydantic.Field("/targets", const=True)
    state: Optional[TargetsStateFilter] = None
    param_attrs: Tuple[str] = pydantic.Field(('state', ), const=True)


class QueryRequest(BaseRequest, abc.ABC):
    """Abstract base class for Prometheus query types.

    ### Attributes:
        query: A PromQL string to be evaluated by Prometheus.
        timeout: The maximum amount of time to consume while evaluating the query.
    """
    query: str
    timeout: Optional[servo.Duration]


class InstantQuery(QueryRequest):
    """A query that returns a vector result of a metric at a moment in time.

    ### Attributes:
        endpoint: Constant value of `/query`.
        time: An optional time value to evaluate the query at. `None` defers to the
            server current time when the query is evaluated.
    """
    endpoint: str = pydantic.Field("/query", const=True)
    param_attrs: Tuple[str] = pydantic.Field(('query', 'time', 'timeout'), const=True)
    time: Optional[datetime.datetime] = None


class RangeQuery(QueryRequest):
    """A query that returns a matrix result of a metric across a series of moments in time.

    ### Attributes:
        endpoint: Constant value of `/query_range`.
        start: Start time of the time range to query.
        end: End time of the time range to query.
        step: Time interval between data points within the queried time range to return data
            points for. A step of '5m' would return a measurement for the metric every five
            minutes across the queried time range, determining the number of data points returned.
    """
    endpoint: str = pydantic.Field("/query_range", const=True)
    param_attrs: Tuple[str] = pydantic.Field(('query', 'start', 'end', 'step', 'timeout'), const=True)
    start: datetime.datetime
    end: datetime.datetime
    step: servo.Duration

    @pydantic.validator("step", pre=True, always=True)
    @classmethod
    def _default_step_from_metric(cls, step, values) -> str:
        if step is None:
            if metric := values.get("metric"):
                return metric.step

        return step

    @pydantic.validator("end")
    @classmethod
    def _validate_range(cls, end, values) -> dict:
        assert end > values["start"], "start time must be earlier than end time"
        return end


class ResultType(str, enum.Enum):
    """Types of results returned for Prometheus queries.

    See https://prometheus.io/docs/prometheus/latest/querying/api/#expression-query-result-formats

    ### Members:
        matrix: A result consisting of an array of objects with metric and values attributes
            describing a time series of scalar measurements.
        vector: A result consisting of an array of objects with metric and value attributes
            describing a single scalar measurement.
        scalar: A result consisting of a time and a numeric value encoded as a string.
        string: A result consisting of a time and a string value.
    """
    matrix = "matrix"
    vector = "vector"
    scalar = "scalar"
    string = "string"


Scalar = Tuple[datetime.datetime, float]
String = Tuple[datetime.datetime, str]


class BaseVector(abc.ABC, pydantic.BaseModel):
    """Abstract base class for Prometheus vector types.

    Vectors are sized and iterable to enable processing response data
    without conditional handling.

    Subclasses must implement `__len__` and `__iter__`.

    Attributes:
        metric: The Prometheus metric name and labels.
    """
    metric: Dict[str, str]

    @abc.abstractmethod
    def __len__(self) -> int:
        ...

    @abc.abstractmethod
    def __iter__(self) -> Scalar:
        ...


class InstantVector(BaseVector):
    """A vector modeling the value of a metric at a moment in time.

    Instant vectors are returned as the result of instant queries in `Response` objects
    with a `result_type` of `vector`.

    ### Attributes:
        value: The time and value of the metric measured.
    """
    value: Scalar

    def __len__(self) -> int:
        return 1

    def __iter__(self) -> Scalar:
        return iter((self.value, ))


class RangeVector(BaseVector):
    """A vector modeling the value of a metric across a series of moments in time.

    Range vectors are returned as the result of range queries in `Response` objects
    with a `result_type` of `matrix`.

    ### Attributes:
        values: A sequence of time and value pairs of the metric across the measured range.
    """
    values: List[Scalar]

    def __len__(self) -> int:
        return len(self.values)

    def __iter__(self) -> Scalar:
        return iter(self.values)


class Status(str, enum.Enum):
    """Prometheus HTTP API response statuses.

    See https://prometheus.io/docs/prometheus/latest/querying/api/#format-overview
    """
    success = "success"
    error = "error"


class Error(pydantic.BaseModel):
    """An error returned in a response from the Prometheus HTTP API.

    ### Attributes:
        type: The type of error that occurred.
        message: A textual description of the error.
    """
    type: str = pydantic.Field(..., alias='errorType')
    message: str = pydantic.Field(..., alias='error')


class QueryData(pydantic.BaseModel):
    """The data component of a response from a query endpoint of the Prometheus HTTP API.

    QueryData is an envelope enclosing the result payload of an evaluated query.
    QueryData objects are sized, sequenced collections of results. Scalar and string results
    are presented as single item collections.

    Attributes:
        result_type: The type of result returned by the query.
        result: The query result. The type is polymorphic based on the result type.
    """
    result_type: ResultType = pydantic.Field(..., alias='resultType')
    result: Union[List[InstantVector], List[RangeVector], Scalar, String]

    def __len__(self) -> int:
        if self.is_vector:
            return len(self.result)
        elif self.is_value:
            return 1
        else:
            raise TypeError(f"unknown data type '{self.result_type}'")

    def __iter__(self):
        if self.is_vector:
            return iter(self.result)
        elif self.is_value:
            return iter((self.result, ))
        else:
            raise TypeError(f"unknown data type '{self.result_type}'")

    def __getitem__(self, index: int):
        return list(iter(self))[index]

    @property
    def is_vector(self) -> bool:
        """Returns True when the result is a vector or matrix."""
        return self.result_type in (servo.connectors.prometheus.ResultType.vector, servo.connectors.prometheus.ResultType.matrix)

    @property
    def is_value(self) -> bool:
        """Returns True when the result is a scalar or string."""
        return self.result_type in (servo.connectors.prometheus.ResultType.scalar, servo.connectors.prometheus.ResultType.string)

class TargetData(pydantic.BaseModel):
    """The data component of a response from the targets endpoint of the Prometheus HTTP API.

    ### Attributes:
        active_targets: The active targets being scraped by Prometheus.
        dropped_targets: Targets that were previously active but are no longer being scraped.
    """
    active_targets: Optional[List[ActiveTarget]] = pydantic.Field(alias="activeTargets")
    dropped_targets: Optional[List[DroppedTarget]] = pydantic.Field(alias="droppedTargets")

    def __len__(self) -> int:
        return len(list(iter(self)))

    def __iter__(self):
        return itertools.chain(self.active_targets or [], self.dropped_targets or [])


Data = Union[QueryData, TargetData]


class BaseResponse(pydantic.BaseModel, abc.ABC):
    """Abstract base class for responses returned by the Prometheus HTTP API.

    All successfully processed Prometheus HTTP API responses contain
    `status` and `data` fields. The response format is documented at:
    https://prometheus.io/docs/prometheus/latest/querying/api/#format-overview

    ### Attributes:
        request: The request that triggered the response.
        status: Whether or not the query was successfully evaluated.
        data: The data payload returned in response to the request.
        error: A description of the error triggering query failure, if any.
        warnings: A list of warnings returned during query evaluation, if any.
    """
    request: BaseRequest
    status: Status
    data: Data
    error: Optional[Error]
    warnings: Optional[List[str]]

    @pydantic.root_validator(pre=True)
    def _parse_error(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        if error := dict(filter(lambda item: item[0].startswith("error"), values.items())):
            values["error"] = error
        return values

    def raise_for_error(self) -> None:
        """Raise an error if the request was not successful."""
        if self.status == Status.error:
            raise RuntimeError(f"Prometheus query request failed with error '{self.error.type}': {self.error.message}")


class TargetsResponse(BaseResponse):
    """A response returned by the targets endpoint of the Prometheus HTTP API.

    `TargetsResponse` objects are sized and iterable. Be aware that the sizing and iteration are
    cumulative of the active and dropped targets collections. Utilize the `active` and `dropped`
    attributes to focus on a particular collection of targets.
    """
    data: TargetData

    def __len__(self) -> int:
        return self.data.__len__()

    def __iter__(self):
        return self.data.__iter__()

    @property
    def active(self) -> List[ActiveTarget]:
        """The active targets in the response data."""
        return self.data.active_targets

    @property
    def dropped(self) -> List[DroppedTarget]:
        """The dropped targets in the response data."""
        return self.data.active_targets


class MetricResponse(BaseResponse):
    """A Prometheus HTTP API response for a servo metric.

    ### Attributes:
        metric: The metric that was queried for.
    """
    metric: PrometheusMetric

    def results(self) -> Optional[List[servo.Reading]]:
        """Return `DataPoint` or `TimeSeries` representations of the query results.

        Response data containing vector and matrix results are serialized into
        `TimeSeries` objects. Scalar and string results are serialized into `DataPoint`.
        """
        if self.status == Status.error:
            return None
        elif not self.data:
            return []

        results_ = []
        for result in self.data:
            if self.data.is_vector:
                results_.append(
                    self._time_series_from_vector(result)
                )
            elif self.data.is_value:
                results_.append(
                    servo.DataPoint(self.metric, **result)
                )
            else:
                raise TypeError(f"unknown Result type '{result.__class__.name}' encountered")

        return results_

    def _time_series_from_vector(self, vector: BaseVector) -> servo.TimeSeries:
        instance = vector.metric.get("instance")
        job = vector.metric.get("job")
        annotation = " ".join(
            map(lambda m: "=".join(m), sorted(vector.metric.items(), key=operator.itemgetter(0)))
        )
        return servo.TimeSeries(
            self.metric,
            list(map(lambda v: servo.DataPoint(self.metric, *v), iter(vector))),
            id=f"{{instance={instance},job={job}}}",
            annotation=annotation,
        )
