from typing import List, Dict, Optional
from pydantic import BaseModel

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


class Message(BaseModel):
    ts: float
    msg: str


class TimeSeries(BaseModel):
    ts: float
    value: float


class Prediction(BaseModel):
    value: float
    error: str


class PredictionResult(BaseModel):
    cpu: Optional[Prediction]
    rate: Optional[Prediction]


class Id(BaseModel):
    id: float


class PodModelWithId(BaseModel):
    model_id: float
    model: str


class TrainPodModel(BaseModel):
    start: float
    end: float


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
    ts: float
    metrics: Dict[str, float] = {}
    nodetypes: List[NodeType] = []
    allocations: List[Allocation] = []
    nodes: List[NodeMetrics] = []
    pods: List[PodMetrics] = []
    excluded_pods: List[PodMetrics] = []
    deployments: List[str] = []
    excluded_deployments: List[str] = []
    replicas: int = 0
