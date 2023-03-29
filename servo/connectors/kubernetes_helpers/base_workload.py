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

import abc
from collections import defaultdict
from typing import Optional, Union

from kubernetes_asyncio.client import (
    V1Deployment,
    V1StatefulSet,
    V1Pod,
    V1ContainerStatus,
    V1PodCondition,
)
from .base import BaseKubernetesHelper
from .pod import PodHelper

from servo.errors import AdjustmentFailedError, AdjustmentRejectedError
from servo.logging import logger
from servo.types.api import Adjustment


class BaseKubernetesWorkloadHelper(BaseKubernetesHelper):
    @classmethod
    @abc.abstractmethod
    def check_conditions(cls, workload: Union[V1Deployment, V1StatefulSet]) -> None:
        ...

    @classmethod
    @abc.abstractmethod
    async def get_latest_pods(
        cls, workload: Union[V1Deployment, V1StatefulSet]
    ) -> list[V1Pod]:
        ...

    @classmethod
    def is_ready(
        cls,
        workload: Union[V1Deployment, V1StatefulSet],
        event_type: Optional[str] = None,
    ):
        if event_type == "ERROR":
            # NOTE: Never seen this in action but apparently its part of k8s https://github.com/kubernetes/kubernetes/blob/6e0de20fbb4c127d2e45c7a22347c08545fc7a86/staging/src/k8s.io/apimachinery/pkg/watch/watch.go#L48
            raise AdjustmentRejectedError(str(workload), reason="start-failed")

        # TODO other rejection checks

        if workload.metadata.generation != workload.status.observed_generation:
            logger.debug(
                f"status observed generation ({workload.status.observed_generation}) does not match"
                f" metadata generation ({workload.metadata.generation}), returning is_ready=false"
            )
            return False

        # Fast fail on undesirable conditions
        # NOTE this check only applies to Deployments (see https://github.com/kubernetes/kubernetes/issues/79606)
        # TODO/FIXME we should really be checking the (latest) pods in order to get the most accurate info on deployment failures
        #   test conditions include FailedScheduling, FailedCreate (error looking up service account),
        if workload.status.conditions:
            cls.check_conditions(workload)

        # NOTE this field is N/A for StatefulSets unless the MaxUnavailableStatefulSet flag is enabled
        if unavailable_count := getattr(workload.status, "unavailable_replicas", 0):
            logger.debug(
                f"found {unavailable_count} unavailable replicas, returning is_ready=false"
            )
            return False

        desired_replicas = workload.spec.replicas
        logger.debug(
            f"Comparing desired replicas ({desired_replicas}) against current status replica counts: {workload.status}"
        )

        # Verify all scale ups and scale downs have completed
        replica_counts: list[int] = [
            workload.status.replicas,  # NOTE this includes replicas from previous versions allowing to wait for scaledowns without returning too early
            workload.status.ready_replicas,
            workload.status.updated_replicas,
        ]
        # NOTE: available counts is not always present on StatefulSets, assumedly due to the
        #   beta status of minReadySeconds https://kubernetes.io/docs/concepts/workloads/controllers/statefulset/#minimum-ready-seconds
        if (
            available_replicas := getattr(workload.status, "available_replicas", None)
        ) is not None:
            replica_counts.append(available_replicas)
        if replica_counts.count(desired_replicas) == len(replica_counts):
            # We are done: all the counts match. Stop the watch and return
            logger.debug(f"{workload.kind} '{workload.metadata.name}' is ready")
            return True

        logger.debug("Replica counts out of alignment, returning is_ready=false")
        return False

    @classmethod
    async def get_restart_count(
        cls, workload: Union[V1Deployment, V1StatefulSet]
    ) -> int:
        count = 0
        for pod in await cls.get_latest_pods(workload):
            count += PodHelper.get_restart_count(pod)

        return count

    @classmethod
    async def raise_for_status(
        cls,
        workload: Union[V1Deployment, V1StatefulSet],
        adjustments: list[Adjustment],
        include_container_logs=False,
    ) -> None:
        # NOTE: operate off of current state, assuming you have checked is_ready()
        status = workload.status
        logger.trace(f"current {workload.kind} status is {status}")
        if status is None:
            raise RuntimeError(f"No such {workload.kind}: {workload.metadata.name}")

        if not status.conditions:
            raise RuntimeError(
                f"{workload.kind} is not running: {workload.metadata.name}"
            )

        # Check for failure conditions
        # NOTE this check only applies to Deployments (see https://github.com/kubernetes/kubernetes/issues/79606)
        if status.conditions:
            cls.check_conditions(workload)

        await cls.raise_for_failed_pod_adjustments(
            workload=workload,
            adjustments=adjustments,
            include_container_logs=include_container_logs,
        )

        # Catchall
        logger.trace(
            f"unable to map {workload.kind} status to exception. workload: {workload}"
        )
        raise RuntimeError(
            f"Unknown {workload.kind} status for '{workload.metadata.name}': {status}"
        )

    @classmethod
    async def raise_for_failed_pod_adjustments(
        cls,
        workload: Union[V1Deployment, V1StatefulSet],
        adjustments: list[Adjustment],
        include_container_logs=False,
    ):
        pods = await cls.get_latest_pods(workload=workload)
        logger.trace(f"latest pod(s) status {list(map(lambda p: p.status, pods))}")
        unschedulable_pods = [
            pod
            for pod in pods
            if pod.status.conditions
            and any(cond.reason == "Unschedulable" for cond in pod.status.conditions)
        ]
        if unschedulable_pods:
            pod_messages = []
            for pod in unschedulable_pods:
                cond_msgs = []
                for unschedulable_condition in filter(
                    lambda cond: cond.reason == "Unschedulable",
                    pod.status.conditions,
                ):
                    unschedulable_adjustments = list(
                        filter(
                            lambda a: a.setting_name in unschedulable_condition.message,
                            adjustments,
                        )
                    )
                    cond_msgs.append(
                        f"Requested adjustment(s) ({', '.join(map(str, unschedulable_adjustments))}) cannot be scheduled due to \"{unschedulable_condition.message}\""
                    )
                pod_messages.append(f"{pod.metadata.name} - {'; '.join(cond_msgs)}")

            raise AdjustmentRejectedError(
                f"{len(unschedulable_pods)} pod(s) could not be scheduled for {workload.kind} {workload.metadata.name}: {', '.join(pod_messages)}",
                reason="unschedulable",
            )

        image_pull_failed_pods = [
            pod
            for pod in pods
            if pod.status.container_statuses
            and any(
                cont_stat.state
                and cont_stat.state.waiting
                and cont_stat.state.waiting.reason
                in ["ImagePullBackOff", "ErrImagePull"]
                for cont_stat in pod.status.container_statuses
            )
        ]
        if image_pull_failed_pods:
            raise AdjustmentFailedError(
                f"Container image pull failure detected on {len(image_pull_failed_pods)} pods: {', '.join(map(lambda pod: pod.metadata.name, pods))}",
                reason="image-pull-failed",
            )

        restarted_pods_container_statuses: list[tuple[V1Pod, V1ContainerStatus]] = [
            (pod, cont_stat)
            for pod in pods
            for cont_stat in (pod.status.container_statuses or [])
            if cont_stat.restart_count > 0
        ]
        if restarted_pods_container_statuses:
            pod_to_counts: dict[str, list] = defaultdict(list)
            for pod, cont_stat in restarted_pods_container_statuses:
                # TODO config to enable logs on per container basis
                log_portion = ""
                if include_container_logs:
                    log_portion = f" container logs {await PodHelper.get_logs_for_container(pod, cont_stat.name)}"
                pod_to_counts[pod.metadata.name].append(
                    f"{cont_stat.name} x{cont_stat.restart_count}{log_portion}"
                )

            pod_message = ", ".join(
                [f"{key} - {'; '.join(val)}" for key, val in pod_to_counts.items()]
            )
            raise AdjustmentRejectedError(
                f"{workload.kind} {workload.metadata.name} pod(s) crash restart detected: {pod_message}",
                reason="unstable",
            )

        # Unready pod catchall
        unready_pod_conds: list[tuple[V1Pod, V1PodCondition]] = [
            (pod, cond)
            for pod in pods
            for cond in (pod.status.conditions or [])
            if cond.type == "Ready" and cond.status == "False"
        ]
        if unready_pod_conds:
            pod_messages = []
            for pod, cond in unready_pod_conds:
                pod_message = (
                    f"{pod.metadata.name} - (reason {cond.reason}) {cond.message}"
                )

                if include_container_logs and cond.reason == "ContainersNotReady":
                    unready_container_statuses: list[V1ContainerStatus] = [
                        cont_stat
                        for cont_stat in pod.status.container_statuses or []
                        if not cont_stat.ready
                    ]
                    container_logs = [
                        f"Container {cont_stat.name}:\n{await PodHelper.get_logs_for_container(pod, cont_stat.name)}"
                        for cont_stat in unready_container_statuses
                    ]
                    # NOTE: cant insert newline (backslash) into f-string brackets
                    pod_message = (
                        f"{pod_message} container logs "
                        + "\n\n--- \n\n".join(container_logs)
                    )

                pod_messages.append(pod_message)

            raise AdjustmentRejectedError(
                f"Found {len(unready_pod_conds)} unready pod(s) for deployment {pod.metadata.name}: {', '.join(pod_messages)}",
                reason="start-failed",
            )
