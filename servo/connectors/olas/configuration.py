from typing import Any, Dict, List, Optional
from pydantic import (
    AnyHttpUrl,
    BaseModel,
    root_validator,
    StrictStr,
    validator,
)

import servo


class ScaleTargetRef(BaseModel):
    apiVersion: Optional[StrictStr]
    kind: Optional[StrictStr]
    service: Optional[StrictStr]      # match deployment by the selector in this service object
    name: Optional[StrictStr]         # only take effect if service is not specified.
    namespace: StrictStr
    exclude: Optional[StrictStr]      # deployment label selector that will be excluded from scaling

    @root_validator(skip_on_failure=True)
    def _validate_custom_scale_target(cls, values: Dict[str, str]) -> Dict[str, str]:
        kind = values.get('kind')
        if not kind or kind == 'Deployment':
            return values

        api = values.get('apiVersion')
        if not api:
            raise ValueError(f"ScaleTargetRef.apiVersion not specified for kind {kind}")
        if len(api.split('/')) != 2:
            raise ValueError(f"ScaleTargetRef.apiVersion {api} should contain exactly one '/'")
        return values

    @root_validator(skip_on_failure=True)
    def _validate_scale_target_exists(cls, values: Dict[str, str]) -> Dict[str, str]:
        if not values.get('service') and not values.get('name'):
            raise ValueError("At least one of 'service' and 'name' in ScaleTargetRef must exist."
                             + " 'name' takes effect if 'service' is not specified.")
        return values


class Config(BaseModel):
    fastpathWindow: int
    waveLength: int
    resolution: int
    mode: StrictStr
    enablePrediction: bool
    coolDown: int
    tolerance: float
    sloWindow: int
    warmUpDelay: int
    envoySideCar: bool
    envoyContour: bool
    disableHPA: bool
    costFormula: Optional[str]

    @validator('resolution')
    def _validate_resolution_value(cls, v: int) -> int:
        if v < 1:
            raise ValueError("'resolution' should be an integer and no less than 1 (denotes 1 minutes).")
        return v


class Query(BaseModel):
    name: str
    query: str


class PrometheusConfig(BaseModel):
    url: AnyHttpUrl
    rq_time_histogram_quantile: int

    @validator('rq_time_histogram_quantile')
    def _validate_quantile_value(cls, v: int) -> int:
        if v < 0 or v > 100:
            raise ValueError("'rq_time_histogram_quantile' denotes quantile that should be between 0 and 100.")
        return v


class Plugin(BaseModel):
    plugin: str
    config: PrometheusConfig
    metrics: List[Query]

    @validator('metrics')
    def _validate_metrics(cls, v: Dict[str, Any], values: Dict[str, Any]) -> Dict[str, Any]:
        names = [q.name for q in v]
        if len(set(names)) != len(names):
            raise ValueError("Metrics with identical names are not allowed")

        source = values.get('plugin')
        if source != 'prometheus':
            return v

        # mandatory metrics for Prometheus plugin
        for n in ['rq_total', 'rq_time_bucket', 'rq_error']:
            if n not in names:
                raise ValueError(f"Metric '{n}' is mandatory for prometheus plugin")
        return v


class Value(BaseModel):
    name: str
    value: float


class Objectives(BaseModel):
    minReplicas: int
    maxReplicas: int
    minScaleCPU: float
    maxCost: float
    metrics: List[Value]
    maxCPU: float
    maxMem: Optional[str]

    @validator('metrics')
    def _validate_metrics_configuration(cls, metrics: List[Value]) -> List[Value]:
        names = [m.name for m in metrics]
        if len(set(names)) != len(names):
            raise ValueError("metrics key contains duplicate metric names")
        if 'rq_time' not in names:
            raise ValueError("'rq_time' must be configured under metrics key")
        if 'cpu' not in names:
            raise ValueError("'cpu' must be configured under metrics key")
        return metrics


class OLASConfiguration(servo.BaseConfiguration):
    '''
    Configuration of the OLAS connector
    '''
    scaleTargetRef: ScaleTargetRef
    config: Config
    objectives: Objectives
    metricSource: List[Plugin]

    @validator('metricSource')
    def _validate_metric_source_has_prometheus(cls, sources: List[Plugin]) -> List[Plugin]:
        if any(s.plugin == 'prometheus' for s in sources):
            return sources
        raise ValueError("'prometheus' plugin must exist in metricSource")

    @classmethod
    def generate(cls, **kwargs) -> "OLASConfiguration":
        '''
        Generate a default configuration for OLASConnector
        '''
        return cls(
            scaleTargetRef=ScaleTargetRef(
                service='foo',
                namespace='default'
            ),
            config=Config(
                fastpathWindow=5,
                waveLength=1440,          # one day in minutes
                resolution=10,            # 10 minutes
                mode='cpu_predict',
                enablePrediction=True,
                coolDown=300,             # 5 minutes
                tolerance=0.1,
                sloWindow=2,
                warmUpDelay=0,
                envoySideCar=False,
                envoyContour=False,
                disableHPA=False,
            ),
            objectives=Objectives(
                minReplicas=1,
                maxReplicas=0,            # 0 denotes no replicas limit
                minScaleCPU=20.0,         # percentage
                maxCPU=0.0,               # 0 denotes no CPU cap
                maxCost=0.0,              # 0 denotes no cost constrain
                metrics=[
                    Value(
                        name='rq_time',   # rq_time uses rq_time_bucket and rq_time_histogram_quantile to calculate a histogram quantile
                        value=60          # more than 60 ms denotes SLO violation
                    ),
                    Value(
                        name='cpu',       # cpu is an internal defined metric collected from Kubernetes Metrics Server
                        value=90          # more than 90% average CPU usage denotes SLO violation
                    ),
                ],
            ),
            metricSource=[
                Plugin(
                    plugin='prometheus',
                    config=PrometheusConfig(
                        url='http://localhost:9090',
                        rq_time_histogram_quantile=90  # use p90 on rq_time_bucket
                    ),
                    metrics=[
                        Query(
                            name='rq_total',           # rq_total expects result is a matrix
                            query='envoy_cluster_upstream_rq_total{envoy_cluster_name="{{ envoy_cluster_name }}"}[90s]'
                        ),
                        Query(
                            name='rq_time_bucket',     # rq_time_bucket expects result is a matrix
                            query='envoy_cluster_upstream_rq_time_bucket{envoy_cluster_name="{{ envoy_cluster_name }}"}[90s]'
                        ),
                        Query(
                            name='rq_error',           # rq_error expects result is a vector
                            query='sum(rate(envoy_cluster_upstream_rq_xx{envoy_cluster_name="{{ envoy_cluster_name }}",envoy_response_code_class!="2"}[2m]))'
                        ),
                    ]
                )
            ]
        )
