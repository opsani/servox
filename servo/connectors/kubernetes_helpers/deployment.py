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
from .base import BaseKubernetesWorkloadHelper
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


# Run a dummy instantiation to detect missing ABC implementations
DeploymentHelper()
