from contextlib import asynccontextmanager
import copy
from typing import Any, AsyncIterator, Optional

from kubernetes_asyncio.client import (
    ApiClient,
    AppsV1Api,
    V1Deployment,
    V1DeploymentCondition,
    V1ObjectMeta,
    V1Pod,
    V1PodTemplateSpec,
    V1ReplicaSet,
)
from servo.connectors.kubernetes_helpers.pod import PodHelper

from servo.errors import ConnectorError, AdjustmentFailedError, AdjustmentRejectedError
from servo.logging import logger
from .base_workload import BaseKubernetesWorkloadHelper
from .replicaset import ReplicasetHelper
from .util import dict_to_string


class DeploymentHelper(BaseKubernetesWorkloadHelper):
    @classmethod
    @asynccontextmanager
    async def api_client(cls) -> AsyncIterator[AppsV1Api]:
        async with ApiClient() as api:
            yield AppsV1Api(api)

    @classmethod
    async def read(cls, name: str, namespace: str) -> V1Deployment:
        logger.debug(f'reading deployment "{name}" in namespace "{namespace}"')
        async with cls.api_client() as api:
            return await api.read_namespaced_deployment(name=name, namespace=namespace)

    @classmethod
    async def patch(
        cls,
        workload: V1Deployment,
        api_client_default_headers: dict[str, str] = {
            "content-type": "application/strategic-merge-patch+json"
        },
    ) -> V1Deployment:
        async with cls.api_client() as api_client:
            # TODO: move up to baser class helper method
            for k, v in (api_client_default_headers or {}).items():
                api_client.api_client.set_default_header(k, v)

            return await api_client.patch_namespaced_deployment(
                name=workload.metadata.name,
                namespace=workload.metadata.namespace,
                body=workload,
            )

    @classmethod
    @asynccontextmanager
    async def watch_args(cls, workload: V1Deployment) -> AsyncIterator[dict[str, Any]]:
        async with cls.api_client() as api:
            metadata: V1ObjectMeta = workload.metadata
            watch_args = {"func": api.list_namespaced_deployment}
            watch_args["namespace"] = metadata.namespace
            watch_args["label_selector"] = dict_to_string(metadata.labels)
            watch_args["field_selector"] = dict_to_string(
                {"metadata.name": metadata.name}
            )
            yield watch_args

    @classmethod
    def check_conditions(cls, workload: V1Deployment) -> None:
        conditions: list[V1DeploymentCondition] = workload.status.conditions
        for condition in conditions:
            if condition.type == "Available":
                if condition.status == "True":
                    # If we hit on this and have not raised yet we are good to go
                    break
                elif condition.status in ("False", "Unknown"):
                    # Condition has not yet been met, log status and continue monitoring
                    logger.debug(
                        f"Condition({condition.type}).status == '{condition.status}' ({condition.reason}): {condition.message}"
                    )
                else:
                    raise AdjustmentFailedError(
                        f"encountered unexpected Condition status '{condition.status}'"
                    )

            elif condition.type == "ReplicaFailure":
                # TODO/FIXME Can't do RCA without getting the ReplicaSet
                raise AdjustmentRejectedError(
                    f"ReplicaFailure: message='{condition.status.message}', reason='{condition.status.reason}'",
                    reason="start-failed",
                )

            elif condition.type == "Progressing":
                if condition.status in ("True", "Unknown"):
                    # Still working
                    logger.debug(
                        f"{workload.kind} update is progressing: {condition}",
                    )
                    break
                elif condition.status == "False":
                    raise AdjustmentRejectedError(
                        f"ProgressionFailure: message='{condition.status.message}', reason='{condition.status.reason}'",
                        reason="start-failed",
                    )
                else:
                    raise AdjustmentFailedError(
                        f"unknown {workload.kind} status condition: {condition}"
                    )

    @classmethod
    async def get_latest_pods(cls, workload: V1Deployment) -> list[V1Pod]:
        latest_replicaset: V1ReplicaSet = await cls.get_latest_replicaset(workload)
        # NOTE Can skip checking owner references due to Deployment setting
        # pod-template-hash on its ReplicaSets
        return await PodHelper.list_pods_with_labels(
            workload.metadata.name, latest_replicaset.spec.selector.matchlabels
        )

    @classmethod
    async def get_latest_replicaset(cls, workload: V1Deployment) -> V1ReplicaSet:
        rs_list = ReplicasetHelper.list_replicasets_with_labels(
            workload.metadata.namespace, workload.spec.selector.match_labels
        )
        # Verify all returned RS have this deployment as an owner
        rs_list = [
            rs
            for rs in rs_list.items
            if rs.metadata.owner_references
            and any(
                ownRef.kind == "Deployment" and ownRef.uid == workload.metadata.uid
                for ownRef in rs.metadata.owner_references
            )
        ]
        if not rs_list:
            raise ConnectorError(
                f'Unable to locate replicaset(s) for deployment "{workload.metadata.name}"'
            )
        if missing_revision_rsets := list(
            filter(
                lambda rs: "deployment.kubernetes.io/revision"
                not in rs.metadata.annotations,
                rs_list,
            )
        ):
            raise ConnectorError(
                f'Unable to determine latest replicaset for deployment "{workload.metadata.name}" due to missing revision'
                f' annotation in replicaset(s) "{", ".join(list(map(lambda rs: rs.metadata.name, missing_revision_rsets)))}"'
            )
        return sorted(
            rs_list,
            key=lambda rs: int(
                rs.metadata.annotations["deployment.kubernetes.io/revision"]
            ),
            reverse=True,
        )[0]

    # NOTE this method may need to become async if other workload types need their spec.template deserialized
    #   by the kubernetes_asyncio.client
    @classmethod
    def get_pod_template_spec_copy(cls, workload: V1Deployment) -> V1PodTemplateSpec:
        """Return a deep copy of the pod template spec. Eg. for creation of a tuning pod"""
        return copy.deepcopy(workload.spec.template)

    @classmethod
    async def inject_sidecar(
        cls,
        workload: V1Deployment,
        name: str,
        image: str,
        *,
        service: Optional[str] = None,
        port: Optional[int] = None,
        index: Optional[int] = None,
        service_port: int = 9980,
    ) -> None:
        """
        Injects an Envoy sidecar into a target Deployment that proxies a service
        or literal TCP port, generating scrapeable metrics usable for optimization.

        The service or port argument must be provided to define how traffic is proxied
        between the Envoy sidecar and the container responsible for fulfilling the request.

        Args:
            name: The name of the sidecar to inject.
            image: The container image for the sidecar container.
            deployment: Name of the target Deployment to inject the sidecar into.
            service: Name of the service to proxy. Envoy will accept ingress traffic
                on the service port and reverse proxy requests back to the original
                target container.
            port: The name or number of a port within the Deployment to wrap the proxy around.
            index: The index at which to insert the sidecar container. When `None`, the sidecar is appended.
            service_port: The port to receive ingress traffic from an upstream service.
        """
        # TODO
        raise NotImplementedError("TODO")
        if not (service or port):
            raise ValueError(f"a service or port must be given")

        if isinstance(port, str) and port.isdigit():
            port = int(port)

        # check for a port conflict
        container_ports = list(
            itertools.chain(*map(operator.attrgetter("ports"), self.containers))
        )
        if service_port in list(
            map(operator.attrgetter("container_port"), container_ports)
        ):
            raise ValueError(
                f"Port conflict: {self.__class__.__name__} '{self.name}' already exposes port {service_port} through an existing container"
            )

        # lookup the port on the target service
        if service:
            try:
                service_obj = await Service.read(service, self.namespace)
            except kubernetes_asyncio.client.exceptions.ApiException as error:
                if error.status == 404:
                    raise ValueError(f"Unknown Service '{service}'") from error
                else:
                    raise error
            if not port:
                port_count = len(service_obj.obj.spec.ports)
                if port_count == 0:
                    raise ValueError(
                        f"Target Service '{service}' does not expose any ports"
                    )
                elif port_count > 1:
                    raise ValueError(
                        f"Target Service '{service}' exposes multiple ports -- target port must be specified"
                    )
                port_obj = service_obj.obj.spec.ports[0]
            else:
                if isinstance(port, int):
                    port_obj = next(
                        filter(lambda p: p.port == port, service_obj.obj.spec.ports),
                        None,
                    )
                elif isinstance(port, str):
                    port_obj = next(
                        filter(lambda p: p.name == port, service_obj.obj.spec.ports),
                        None,
                    )
                else:
                    raise TypeError(
                        f"Unable to resolve port value of type {port.__class__} (port={port})"
                    )

                if not port_obj:
                    raise ValueError(
                        f"Port '{port}' does not exist in the Service '{service}'"
                    )

            # resolve symbolic name in the service target port to a concrete container port
            if isinstance(port_obj.target_port, str):
                container_port_obj = next(
                    filter(lambda p: p.name == port_obj.target_port, container_ports),
                    None,
                )
                if not container_port_obj:
                    raise ValueError(
                        f"Port '{port_obj.target_port}' could not be resolved to a destination container port"
                    )

                container_port = container_port_obj.container_port
            else:
                container_port = port_obj.target_port

        else:
            # find the container port
            container_port_obj = next(
                filter(lambda p: p.container_port == port, container_ports), None
            )
            if not container_port_obj:
                raise ValueError(
                    f"Port '{port}' could not be resolved to a destination container port"
                )

            container_port = container_port_obj.container_port

        # build the sidecar container
        container = kubernetes_asyncio.client.V1Container(
            name=name,
            image=image,
            image_pull_policy="IfNotPresent",
            resources=kubernetes_asyncio.client.V1ResourceRequirements(
                requests={"cpu": "125m", "memory": "128Mi"},
                limits={"cpu": "250m", "memory": "256Mi"},
            ),
            env=[
                kubernetes_asyncio.client.V1EnvVar(
                    name="OPSANI_ENVOY_PROXY_SERVICE_PORT", value=str(service_port)
                ),
                kubernetes_asyncio.client.V1EnvVar(
                    name="OPSANI_ENVOY_PROXIED_CONTAINER_PORT",
                    value=str(container_port),
                ),
                kubernetes_asyncio.client.V1EnvVar(
                    name="OPSANI_ENVOY_PROXY_METRICS_PORT", value="9901"
                ),
            ],
            ports=[
                kubernetes_asyncio.client.V1ContainerPort(
                    name="opsani-proxy", container_port=service_port
                ),
                kubernetes_asyncio.client.V1ContainerPort(
                    name="opsani-metrics", container_port=9901
                ),
            ],
        )

        # add the sidecar to the Deployment
        if index is None:
            self.obj.spec.template.spec.containers.append(container)
        else:
            self.obj.spec.template.spec.containers.insert(index, container)

        # patch the deployment
        await self.patch()


# Run a dummy instantiation to detect missing ABC implementations
DeploymentHelper()
