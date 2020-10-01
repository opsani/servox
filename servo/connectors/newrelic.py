from __future__ import annotations
import asyncio
from collections import defaultdict
from dateutil import parser as date_parser
from functools import reduce
from datetime import datetime, timedelta
from typing import Awaitable, Callable, Dict, Iterable, List, Optional, Set, Tuple

import httpx
import httpcore._exceptions
from pydantic import BaseModel, AnyHttpUrl, validator

from servo import (
    BaseChecks,
    BaseConfiguration,
    BaseConnector,
    Control,
    Check,
    CheckHandler,
    Description,
    Duration,
    Filter,
    HaltOnFailed,
    License,
    Maturity,
    Measurement,
    Metric,
    Unit,
    check,
    metadata,
    on_event,
    TimeSeries,
    DurationProgress,
)
from servo.checks import create_checks_from_iterable, multicheck
from servo.utilities import join_to_series


DEFAULT_BASE_URL = 'https://api.newrelic.com'
API_PATH = "/v2"

# TODO: how are secrets handled?
NEWRELIC_ACCOUNT_ID = str(open('/run/secrets/optune_newrelic_account_id').read()).strip()
NEWRELIC_APM_API_KEY = str(open('/run/secrets/optune_newrelic_apm_api_key').read()).strip()
NEWRELIC_APM_APP_ID = str(open('/run/secrets/optune_newrelic_apm_app_id').read()).strip()

class NewrelicMetric(Metric):
    """NewrelicMetric objects describe metrics that can be measure from the Newrelic apm api."""    
    
    fetch_name: str
    """The name of the APM metric containing the values for this Metric

    For details, see the [TODO better Newrelic resource](https://docs.newrelic.com/docs/apis/rest-api-v2/application-examples-v2/average-response-time-examples-v2) documentation.
    """

    values_selector: str
    """
    Values selector  of resultant time series
    """

    def __init__(self, name: str, unit: Unit, fetch_name: str, values_selector: str) -> None:
        super().__init__(name=name, unit=unit, fetch_name=fetch_name, values_selector=values_selector, **kwargs)

    def __check__(self) -> Check:
        return Check(
            name=f"Check {self.name}",
            description=f"Run Newrelic get \"{self.fetch_name}: {self.values_selector}\""
        ) # TODO Checker class

# TODO: what is this used for?
# class NewrelicTarget():
#     pass # TODO

class NewrelicConfiguration(BaseConfiguration):
    """NewrelicConfiguration objects describe how NewrelicConnector objects
capture measurements from the Newrelic metrics server. 
    """

    base_url: AnyHttpUrl = DEFAULT_BASE_URL
    """The base URL for accessing the Newrelic metrics API.

    The URL must point to the root of the Newrelic deployment. Resource paths
    are computed as necessary for API requests.
    """

    metrics: List[NewrelicMetric]
    """The metrics to measure from Newrelic.

    Metrics must include a valid fetch object and values selector.
    """

    step: Duration = "1m"
    """The resolution of the metrics to be fetched.
    
    The step resolution determines the number of data points captured across a
    query range.
    """

    # targets: Optional[List[NewrelicTarget]] TODO: what is this used for?
    """An optional set of Newrelic target descriptors that are expected to be
    scraped by the Newrelic instance being queried.
    """

    @classmethod # TODO
    def generate(cls, **kwargs) -> "NewrelicConfiguration":
        """Generates a default configuration for capturing measurements from the
        Newrelic metrics server.

        Returns:
            A default configuration for NewrelicConnector objects.
        """
        return cls(
            description="Update the base_url and metrics to match your Newrelic configuration",
            metrics=[
                NewrelicMetric(
                    name="throughput",
                    unit=Unit.REQUESTS_PER_SECOND,
                    fetch_name="HttpDispatcher",
                    values_selector="requests_per_minute"
                ),
                NewrelicMetric(
                    "error_rate", Unit.COUNT, "Errors/all", "error_count"
                ),
            ],
            **kwargs,
        )
    
    @validator("base_url", allow_reuse=True)
    @classmethod
    def rstrip_base_url(cls, base_url):
        return base_url.rstrip("/")

    @property
    def api_url(self) -> str:
        return f"{self.base_url}{API_PATH}"

class NewrelicRequest(BaseModel):
    base_url: AnyHttpUrl
    fetches: Dict[str, Set[str]]
    start: datetime
    end: datetime
    step: Duration

    @property
    def params(self) -> Dict[str, str]:
        return { 
            'names[]': '+'.join(self.fetches.keys()), # TODO can these be lists and httpx will extrapolate?
            'values[]': '+'.join(reduce(lambda x, y: x+y, self.fetches.values())),
            'from': self.start.isoformat(),
            'to': self.end.isoformat(),
            'period': self.step.total_seconds(),
            'summarize': False,
            'raw': True,
        }

class NewrelicChecks():
    pass # TODO

@metadata(
    description="Newrelic Connector for Opsani",
    version="0.0.1",
    homepage="https://github.com/opsani/newrelic-connector",
    license=License.APACHE2,
    maturity=Maturity.EXPERIMENTAL,
)
class NewrelicConnector(BaseConnector):
    """NewrelicConnector objects enable servo assemblies to capture
    measurements from the [Newrelic](https://newrelic.com/) metrics server.
    """
    config: NewrelicConfiguration

    # TODO
    # @on_event()
    # async def check(self,
    #     filter_: Optional[Filter] = None, 
    #     halt_on: HaltOnFailed = HaltOnFailed.requirement
    # ) -> List[Check]:
    #     """Checks that the configuration is valid and the connector can capture        
    #     measurements from Newrelic.

    #     Checks are implemented in the NewrelicChecks class.

    #     Args:
    #         filter_ (Optional[Filter], optional): A filter for limiting the
    #             checks that are run. Defaults to None.
    #         halt_on (HaltOnFailed, optional): When to halt running checks.
    #             Defaults to HaltOnFailed.requirement.

    #     Returns:
    #         List[Check]: A list of check objects that report the outcomes of the            
    #             checks that were run.
    #     """        
    #     return await NewrelicChecks.run(self.config, filter_, halt_on=halt_on)

    @on_event()
    def describe(self) -> Description:
        """Describes the current state of Metrics measured by querying Newrelic.

        Returns:
            Description: An object describing the current state of metrics
                queried from Newrelic.
        """
        return Description(metrics=self.config.metrics)

    @property
    @on_event()
    def metrics(self) -> List[Metric]:
        """Returns the list of Metrics measured through Newrelic queries.

        Returns:
            List[Metric]: The list of metrics to be queried.
        """
        return self.config.metrics

    @on_event()
    async def measure(
        self, *, metrics: List[str] = None, control: Control = Control()
    ) -> Measurement:
        """Queries Newrelic for metrics as time series values and returns a
        Measurement object that aggregates the readings for processing by the
        optimizer.

        Args:
            metrics (List[str], optional): A list of the metric names to measure. 
                When None, all configured metrics are measured. Defaults to None.
            control (Control, optional): A control descriptor that describes how            
                the measurement is to be captured. Defaults to Control().

        Returns:
            Measurement: An object that aggregates the state of the metrics
            queried from Newrelic.
        """
        if metrics:
            metrics__ = list(filter(lambda m: m.name in metrics, self.metrics))
        else:
            metrics__ = self.metrics
        measuring_names = list(map(lambda m: m.name, metrics__))
        self.logger.info(f"Starting measurement of {len(metrics__)} metrics: {join_to_series(measuring_names)}")

        start = datetime.now() + control.warmup
        end = start + control.duration

        sleep_duration = Duration(control.warmup + control.duration)
        self.logger.info(
            f"Waiting {sleep_duration} during metrics collection ({control.warmup} warmup + {control.duration} duration)..."
        )

        progress = DurationProgress(sleep_duration)
        notifier = lambda p: self.logger.info(p.annotate(f"waiting {sleep_duration} during metrics collection...", False), progress=p.progress)
        await progress.watch(notifier)
        self.logger.info(f"Done waiting {sleep_duration} for metrics collection, resuming optimization.")

        # Fetch the measurements
        self.logger.info(f"Querying Newrelic for {len(metrics__)} metrics...")
        readings = await self._query_newrelic(metrics__, start, end)
        measurement = Measurement(readings=readings)
        return measurement

    async def _query_newrelic(
        self, metrics: List[NewrelicMetric], start: datetime, end: datetime
    ) -> List[TimeSeries]:
        f_names_to_ms: Dict[str, List[NewrelicMetric]] = defaultdict(list)
        fetches: Dict[str, Set[str]] = defaultdict(set)
        for m in metrics:
            f_names_to_ms[m.fetch_name].append(m)
            fetches[m.fetch_name].add(m.values_selector)

        newrelic_request = NewrelicRequest(base_url=self.config.api_url, fetches=fetches, start=start, end=end, step=self.config.step)

        self.logger.trace(f"Getting Newrelic instance ids: {newrelic_request.url}")
        instance_ids = []
        # TODO
        
        self.logger.trace(f"Querying Newrelic: {newrelic_request.url}")
        readings = []
        # TODO asyncio gather this instead
        for i in instance_ids:
            api_path = '/applications/{app_id}/instances/{instance_id}/metrics/data.json'.format(NEWRELIC_APM_APP_ID, i)
            self.logger.trace(f"Querying Newrelic for instance: {i}")
            async with self.api_client() as client:
                try:
                    response = await client.get(self.config.api_url + api_path, params=newrelic_request.params)
                    response.raise_for_status()
                except (httpx.HTTPError, httpcore._exceptions.ReadTimeout, httpcore._exceptions.ConnectError) as error:                
                    self.logger.trace(f"HTTP error encountered during GET {newrelic_request.url}: {error}")
                    raise

            data = response.json()
            self.logger.trace(f"Got response data for instance {i}: {data}")


            if "status" not in data or data["status"] != "success":
                raise RuntimeError(f"instance {i} fetch unsuccessful", data)

            for fetched_m in data.get('metric_data', {}).get('metrics', []):
                m_readings: Dict[NewrelicMetric, List[Tuple[datetime, Numeric]]] = defaultdict(list)
                for ts in fetched_m.get('timeslices', []):
                    for m in f_names_to_ms[fetched_m['name']]:
                        m_readings[m].append((date_parser.parse(ts['from']), ts['values'][m.values_selector]))
                for m in f_names_to_ms[fetched_m['name']]:
                    readings.append(
                        TimeSeries(
                            metric=m,
                            values=m_readings[m],
                            id=i,
                            metadata=dict(instance=i)
                        )
                    )
        return readings

