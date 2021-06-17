from typing import List, Dict, Optional
from pydantic import BaseModel, validator

from servo.connectors.olas.utility import str_to_epoch, TIME_STAMP_FORMAT


__all__ = [
    "Message",
    "TimeSeries",
    "PredictionResult",
    "Id",
    "PodModelWithId",
    "TrainPodModel",
    "Allocation",
    "NodeType",
    "NodeMetrics",
    "PodMetrics",
    "Metrics",
]


def timestamp_validator(ts: str) -> str:
    try:
        str_to_epoch(ts)
        return ts
    except ValueError:
        raise ValueError(f"Time string '{ts}' does not conform to ISO format '{TIME_STAMP_FORMAT}'.")


class Message(BaseModel):
    ts: str
    msg: str

    _formatted_timestamp = validator('ts', allow_reuse=True)(timestamp_validator)


class TimeSeries(BaseModel):
    ts: str
    value: float

    _formatted_timestamp = validator('ts', allow_reuse=True)(timestamp_validator)


class Prediction(BaseModel):
    value: float
    error: str


class PredictionResult(BaseModel):
    cpu: Optional[Prediction]
    rate: Optional[Prediction]


class Id(BaseModel):
    id: str


class PodModelWithId(BaseModel):
    model_id: str
    model: str

    _formatted_timestamp = validator('model_id', allow_reuse=True)(timestamp_validator)


class TrainPodModel(BaseModel):
    start: str
    end: str

    _formatted_timestamp = validator('start', 'end', allow_reuse=True)(timestamp_validator)


class Allocation(BaseModel):
    cpu_request: float = 0
    cpu_limit: float = 0
    memory_request: float = 0
    memory_limit: float = 0


class NodeType(BaseModel):
    name: str
    cpu_allocatable: float
    memory_allocatable: float
    cpu_capacity: float
    memory_capacity: float


class Usage(BaseModel):
    cpu: float
    memory: float


class NodeMetrics(BaseModel):
    name: str
    nodetype: int = 0
    usage: Usage
    allocation: Allocation


class PodMetrics(BaseModel):
    name: str
    allocation: int = 0
    node: int = 0
    metrics: Dict[str, float]


class Metrics(BaseModel):
    ts: str
    metrics: Dict[str, float] = {}
    nodetypes: List[NodeType] = []
    allocations: List[Allocation] = []
    nodes: List[NodeMetrics] = []
    pods: List[PodMetrics] = []
    excluded_pods: List[PodMetrics] = []
    deployments: List[str] = []
    excluded_deployments: List[str] = []
    replicas: int = 0

    _formatted_timestamp = validator('ts', allow_reuse=True)(timestamp_validator)
