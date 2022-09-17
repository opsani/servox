from typing import Mapping, Optional, Union

from kubernetes_asyncio.client import (
    V1Container,
    V1Pod,
    V1StatefulSet,
    V1Deployment,
    V1PodTemplateSpec,
)


def dict_to_string(mapping: Mapping[str, str]) -> str:
    # https://stackoverflow.com/a/17888002
    return ",".join(["=".join((k, v)) for k, v in mapping.items()])


# NOTE this method may need to become async if other workload types need their container object deserialized
#   by the kubernetes_asyncio.client
def find_container(
    workload: Union[V1Pod, V1PodTemplateSpec, V1StatefulSet, V1Deployment], name: str
) -> Optional[V1Container]:
    containers: list[V1Container]
    if isinstance(workload, (V1Pod, V1PodTemplateSpec)):
        containers = workload.spec.containers
    else:
        containers = workload.spec.template.spec.containers

    return next(iter((c for c in containers if c.name == name)), None)
