# Copyright 2022 Cisco Systems, Inc. and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import contextlib
from functools import partial
from typing import Any, AsyncIterator, cast, Optional, Union

from kubernetes_asyncio.client import (
    ApiClient,
    ApiException,
    CoreV1Api,
    V1ContainerStatus,
    V1ObjectMeta,
    V1Pod,
    V1PodList,
    V1PodCondition,
    V1PodStatus,
    V1Status,
)

from servo.errors import AdjustmentFailedError, AdjustmentRejectedError, EventError
from servo.logging import logger
from servo.types.api import Adjustment
from servo.types.kubernetes import ContainerLogOptions
from .base import BaseKubernetesHelper
from .util import dict_to_selector

# FIXME should be coming from servo.types.telemetry which does not exist (yet)
ONE_MiB = 1048576


class PodHelper(BaseKubernetesHelper):
    @classmethod
    @contextlib.asynccontextmanager
    async def api_client(cls) -> AsyncIterator[CoreV1Api]:
        async with ApiClient() as api:
            yield CoreV1Api(api)

    @classmethod
    async def create(cls, workload: V1Pod):
        metadata: V1ObjectMeta = workload.metadata
        logger.info(
            f'creating pod "{metadata.name}" in namespace "{metadata.namespace}"'
        )
        async with cls.api_client() as api:
            return await api.create_namespaced_pod(
                namespace=metadata.namespace, body=workload
            )

    @classmethod
    async def read(cls, name: str, namespace: str) -> V1Pod:
        logger.debug(f'reading pod "{name}" in namespace "{namespace}"')
        async with cls.api_client() as api:
            return await api.read_namespaced_pod(name=name, namespace=namespace)

    @classmethod
    async def delete(cls, pod: V1Pod) -> None:
        metadata: V1ObjectMeta = pod.metadata
        logger.debug(
            f'deleting pod "{metadata.name}" in namespace "{metadata.namespace}"'
        )
        async with cls.api_client() as api:
            return await api.delete_namespaced_pod(
                name=metadata.name, namespace=metadata.namespace
            )

    @classmethod
    @contextlib.asynccontextmanager
    async def watch_args(cls, pod: V1Pod) -> AsyncIterator[dict[str, Any]]:
        async with cls.api_client() as api:
            metadata: V1ObjectMeta = pod.metadata
            watch_args = {"func": api.list_namespaced_pod}
            watch_args["namespace"] = metadata.namespace
            watch_args["label_selector"] = dict_to_selector(metadata.labels)
            watch_args["field_selector"] = dict_to_selector(
                {"metadata.name": metadata.name}
            )
            yield watch_args

    @classmethod
    def is_ready(cls, pod: V1Pod, event_type: Optional[str] = None) -> bool:
        # implementation derived from official go client
        # https://github.com/kubernetes/kubernetes/blob/096dafe757f897a9d1d9f6160451813062eec063/test/utils/conditions.go#L33
        status: V1PodStatus = pod.status
        logger.trace(f"current pod status is {status}")
        if status is None:
            return False

        phase = status.phase
        logger.debug(f"current pod phase is {phase}")
        if phase != "Running":
            return False

        conditions: list[V1PodCondition] = status.conditions or []
        logger.debug(f"checking status conditions {conditions}")
        ready_condition = next(iter((c for c in conditions if c.type == "Ready")), None)
        if ready_condition and ready_condition.status == "True":
            return True

        logger.debug(f"unable to find ready=true, continuing to wait...")
        return False

    @classmethod
    async def list_pods_with_labels(
        cls, namespace: str, match_labels: dict[str, str]
    ) -> list[V1Pod]:
        async with cls.api_client() as api:
            pod_list: V1PodList = await api.list_namespaced_pod(
                namespace=namespace, label_selector=dict_to_selector(match_labels)
            )
            return pod_list.items or []

    @classmethod
    def get_restart_count(cls, pod: V1Pod, container_name: Optional[str] = None) -> int:
        """Return restart count for all containers by default or a specific container if the optional container_name
        is specified
        """
        if pod.status is None or pod.status.container_statuses is None:
            return 0

        total = 0
        for container_status in pod.status.container_statuses:
            if container_status.name == container_name:
                return container_status.restart_count

            total += container_status.restart_count

        if container_name:
            raise RuntimeError(
                f"Unable to determine container status for {container_name} from pod {pod}"
            )

        return total

    @classmethod
    async def raise_for_status(
        cls,
        workload: V1Pod,
        adjustments: list[Adjustment],
        include_container_logs=False,
    ) -> None:
        """Raise an exception if the Pod status is not not ready."""
        # NOTE: operate off of current state, assuming you have checked is_ready()
        status: V1PodStatus = workload.status
        logger.trace(f"current pod status is {status}")

        if not status.conditions:
            raise EventError(f"Pod is not running: {workload.metadata.name}")

        logger.debug(f"checking container statuses: {status.container_statuses}")
        if status.container_statuses:
            for cont_stat in cast(list[V1ContainerStatus], status.container_statuses):
                if (
                    cont_stat.state
                    and cont_stat.state.waiting
                    and cont_stat.state.waiting.reason
                    in ["ImagePullBackOff", "ErrImagePull"]
                ):
                    raise AdjustmentFailedError(
                        f"Container image pull failure detected in container {cont_stat.name}",
                        reason="image-pull-failed",
                    )

        restarted_container_statuses: list[V1ContainerStatus] = [
            cont_stat
            for cont_stat in cast(list[V1ContainerStatus], status.container_statuses)
            or []
            if cont_stat.restart_count > 0
        ]
        if restarted_container_statuses:
            container_messages = [
                (
                    f"{cont_stat.name} x{cont_stat.restart_count}"
                    # TODO enable logs config on per container basis
                    f"container logs {'DISABLED' if not include_container_logs else await cls.get_logs_for_container(workload, cont_stat.name)}"
                )
                for cont_stat in restarted_container_statuses
            ]
            raise AdjustmentRejectedError(
                # NOTE: cant use f-string with newline (backslash) insertion
                (
                    f"Tuning optimization {workload.metadata.name} crash restart detected on container(s): "
                    + ", \n".join(container_messages)
                ),
                reason="unstable",
            )

        logger.debug(f"checking status conditions {status.conditions}")
        for cond in cast(list[V1PodCondition], status.conditions):
            if cond.reason == "Unschedulable":
                # FIXME: The servo rejected error should be raised further out. This should be a generic scheduling error
                unschedulable_adjustments = [
                    a for a in adjustments if a.setting_name in cond.message
                ]
                raise AdjustmentRejectedError(
                    f"Requested adjustment(s) ({', '.join(map(str, unschedulable_adjustments))}) cannot be scheduled due to \"{cond.message}\"",
                    reason="unschedulable",
                )

            if cond.type == "Ready" and cond.status == "False":
                rejection_message = cond.message
                if include_container_logs and cond.reason == "ContainersNotReady":
                    unready_container_statuses = [
                        cont_stat
                        for cont_stat in cast(
                            list[V1ContainerStatus], status.container_statuses
                        )
                        or []
                        if not cont_stat.ready
                    ]
                    container_logs = [
                        await cls.get_logs_for_container(workload, cs.name)
                        for cs in unready_container_statuses
                    ]
                    # NOTE: cant insert newline (backslash) into f-string brackets
                    rejection_message = (
                        f"{rejection_message} container logs "
                        + "\n\n--- \n\n".join(container_logs)
                    )
                raise AdjustmentRejectedError(
                    f"(reason {cond.reason}) {rejection_message}", reason="start-failed"
                )

        # Catchall
        logger.error(
            f"unable to determine type of error to raise for pod {workload.metadata.name} status: {status}"
        )
        raise EventError(f"Unknown Pod status for '{workload.metadata.name}': {status}")

    @classmethod
    async def get_logs_for_container(
        cls,
        pod: V1Pod,
        container_name: str,
        limit_bytes: int = ONE_MiB,
        logs_selector: ContainerLogOptions = ContainerLogOptions.both,
    ) -> list[str]:
        """
        Get container logs from the current pod for the container's whose statuses are provided in the list

        Args:
            container_statuses (list[V1ContainerStatus]): The name of the Container.
            limit_bytes (int): Maximum bytes to provide per log (NOTE: this will be 2x per container )
            logs_selector (ContainerLogOptions): "previous", "current", or "both"

        Returns:
            list[str]: List of logs per container in the same order as the list of container_statuses
        """
        read_logs_partial = partial(
            cls.try_get_container_single_log,
            pod=pod,
            container_name=container_name,
            limit_bytes=limit_bytes,
        )
        if logs_selector == ContainerLogOptions.both:
            return (
                f"previous (crash):\n {await read_logs_partial(previous=True)} \n\n--- \n\n"
                f"current (latest):\n {await read_logs_partial(previous=False)}"
            )
        else:
            return await read_logs_partial(
                previous=(logs_selector == ContainerLogOptions.previous)
            )

    @classmethod
    async def try_get_container_single_log(
        cls,
        pod: V1Pod,
        container_name: str,
        limit_bytes: int = ONE_MiB,
        previous=False,
    ) -> str:
        """Get log for a container run while handling common error cases (eg. Not Found)"""
        async with cls.api_client() as api:
            try:
                return await api.read_namespaced_pod_log(
                    name=pod.metadata.name,
                    namespace=pod.metadata.namespace,
                    container=container_name,
                    limit_bytes=limit_bytes,
                    previous=previous,
                )
            except ApiException as ae:
                if ae.status == 400:
                    ae.data = ae.body
                    status: V1Status = api.api_client.deserialize(ae, "V1Status")
                    if (status.message or "").endswith("not found"):
                        return "Logs not found"

                raise


# Run a dummy instantiation to detect missing ABC implementations
PodHelper()
