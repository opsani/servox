"""Optimize services and applications deployed on Kubernetes with Opsani.
"""
from __future__ import annotations, print_function

import abc
import asyncio
import collections
import contextlib
import copy
import datetime
import decimal
import enum
import functools
import itertools
import json
import operator
import os
import pathlib
import re
from typing import (
    Any,
    Callable,
    ClassVar,
    Collection,
    Dict,
    Generator,
    Iterable,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Type,
    Union,
    cast,
    get_type_hints,
    runtime_checkable,
)

import backoff
import kubernetes_asyncio
import kubernetes_asyncio.client.models
import kubernetes_asyncio.client
import kubernetes_asyncio.client.exceptions
import kubernetes_asyncio.watch
import pydantic

import servo


class Condition(servo.logging.Mixin):
    """A Condition is a convenience wrapper around a function and its arguments
    which allows the function to be called at a later time.

    The function is called in the ``check`` method, which resolves the result to
    a boolean value, thus the condition function should return a boolean or
    something that ultimately resolves to a Truthy or Falsey value.

    Args:
        name: The name of the condition to make it easier to identify.
        fn: The condition function that will be checked.
        *args: Any arguments for the condition function.
        **kwargs: Any keyword arguments for the condition function.

    Attributes:
        name (str): The name of the Condition.
        fn (callable): The condition function that will be checked.
        args (tuple): Arguments for the checking function.
        kwargs (dict): Keyword arguments for the checking function.
        last_check (bool): Holds the state of the last condition check.

    Raises:
        ValueError: The given ``fn`` is not callable.
    """

    def __init__(self, name: str, fn: Callable, *args, **kwargs) -> None: # noqa: D107
        if not callable(fn):
            raise ValueError("The Condition function must be callable")

        self.name = name
        self.fn = fn
        self.args = args
        self.kwargs = kwargs

        # last check holds the state of the last check.
        self.last_check = False

    def __str__(self) -> str:
        return f"<Condition (name: {self.name}, met: {self.last_check})>"

    def __repr__(self) -> str:
        return self.__str__()

    async def check(self) -> bool:
        """Check that the condition was met.

        Returns:
            True if the condition was met; False otherwise.
        """
        if asyncio.iscoroutinefunction(self.fn):
            self.last_check = bool(await self.fn(*self.args, **self.kwargs))
        else:
            self.last_check = bool(self.fn(*self.args, **self.kwargs))
        return self.last_check


async def wait_for_condition(
    condition: Condition,
    interval: servo.DurationDescriptor = 0.05,
    fail_on_api_error: bool = True,
) -> None:
    """Wait for a condition to be met.

    Args:
        condition: The Condition to wait for.
        timeout: The maximum time to wait, in seconds, for the condition to be met.
            If unspecified, this function will wait indefinitely. If specified and
            the timeout is met or exceeded, a TimeoutError will be raised.
        interval: The time, in seconds, to wait before re-checking the condition.
        fail_on_api_error: Fail the condition checks if a Kubernetes API error is
            incurred. An API error can be raised for a number of reasons, including
            a Pod being restarted and temporarily unavailable. Disabling this will
            cause those errors to be ignored, allowing the check to continue until
            timeout or resolution. (default: True).

    Raises:
        TimeoutError: The specified timeout was exceeded.
    """
    servo.logger.debug(f"waiting for condition: {condition}")

    started_at = datetime.datetime.now()
    duration = servo.Duration(interval)
    async def _wait_for_condition() -> None:
        servo.logger.debug(f"wait for condition: {condition}")
        while True:
            try:
                servo.logger.trace(f"checking condition {condition}")
                if await condition.check():
                    servo.logger.trace(f"condition passed: {condition}")
                    break

                # if the condition is not met, sleep for the interval
                # to re-check later
                servo.logger.trace(f"sleeping for {duration}")
                await asyncio.sleep(duration.total_seconds())

            except asyncio.CancelledError:
                servo.logger.trace(f"wait for condition cancelled: {condition}")
                raise

            except kubernetes_asyncio.client.exceptions.ApiException as e:
                servo.logger.warning(f"encountered API exception while waiting: {e}")
                if fail_on_api_error:
                    raise

    task = asyncio.create_task(_wait_for_condition())
    try:
        await task
    except asyncio.CancelledError:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

        raise
    finally:
        servo.logger.debug(f"wait completed (total={servo.Duration.since(started_at)}) {condition}")


class Resource(str, enum.Enum):
    memory = "memory"
    cpu = "cpu"

    @classmethod
    def values(cls) -> List[str]:
        """
        Return a list of strings that identifies all resource values.
        """
        return list(map(lambda rsrc: rsrc.value, cls.__members__.values()))

class ResourceRequirement(enum.Enum):
    """
    The ResourceRequirement enumeration determines how optimization values are submitted to the
    Kubernetes scheduler to allocate core compute resources. Requests establish the lower bounds
    of the CPU and memory necessary for an application to execute while Limits define the upper
    bounds for resources that can be consumed by a given Pod. The Opsani engine can determine
    optimal values for these settings by identifying performant, low cost configurations that meet
    target SLOs and/or maximizing performance while identifying the point of diminishing returns
    on further resourcing.
    """

    request = 'request'
    limit = 'limit'

    @property
    def resources_key(self) -> str:
        """
        Return a string value for accessing resource requirements within a Kubernetes Container representation.
        """
        if self == ResourceRequirement.request:
            return "requests"
        elif self == ResourceRequirement.limit:
            return "limits"
        else:
            raise NotImplementedError(
                f'missing resources_key implementation for resource requirement "{self}"'
            )


@runtime_checkable
class KubernetesObj(Protocol):
    """
    KubernetesObj is a protocol that defines the common attributes
    of objects retrieved from the Kubernetes API.
    """

    @property
    def api_version(self) -> str:
        ...

    @property
    def kind(self) -> str:
        ...

    @property
    def metadata(self) -> kubernetes_asyncio.client.V1ObjectMeta:
        ...


class KubernetesModel(abc.ABC, servo.logging.Mixin):
    """
    KubernetesModel is an abstract base class for Servo connector
    models that wrap Kubernetes API objects.

    This base class provides common functionality and common object
    properties for all API wrappers. It also defines the following
    abstract methods which all subclasses must implement:

      - ``create``: create the resource on the cluster
      - ``patch``: partially update the resource on the cluster
      - ``delete``: remove the resource from the cluster
      - ``refresh``: refresh the underlying object model
      - ``is_ready``: check if the object is in the ready state

    Args:
         api_object: The underlying Kubernetes API object.

    Attributes:
        obj: The underlying Kubernetes API object.
    """

    obj: KubernetesObj
    """The underlying Kubernetes API object. Subclasses must update
    the type hint to reflect the type that they are wrapping.
    """

    api_clients: ClassVar[Dict[str, Type]]
    """A mapping of all the supported api clients for the API
    object type. Various resources can have multiple versions,
    e.g. "apps/v1", "apps/v1beta1", etc. The preferred version
    for each resource type should be defined under the "preferred"
    key. The preferred API client will be used when the apiVersion
    is not specified for the resource.
    """

    def __init__(self, obj, **kwargs) -> None: # noqa: D107
        self.obj = obj
        self._logger = servo.logger

    def __str__(self) -> str:
        return str(self.obj)

    def __repr__(self) -> str:
        return self.__str__()

    @classmethod
    def obj_type(cls) -> Type:
        """The type of the underlying Kubernetes API object."""
        return get_type_hints(cls)["obj"]

    @property
    def api_version(self) -> str:
        """The API version of the Kubernetes object (`obj.apiVersion``)."""
        return self.obj.api_version

    @property
    def name(self) -> str:
        """The name of the Kubernetes object (``obj.metadata.name``)."""
        return cast(str, self.obj.metadata.name)

    @name.setter
    def name(self, name: str):
        """Set the name of the Kubernetes object (``obj.metadata.name``)."""
        self.obj.metadata.name = name

    @property
    def namespace(self) -> str:
        """The namespace of the Kubernetes object (``obj.metadata.namespace``)."""
        return cast(str, self.obj.metadata.namespace)

    @namespace.setter
    def namespace(self, namespace: str):
        """Set the namespace of the Kubernetes object (``obj.metadata.namespace``)."""
        self.obj.metadata.namespace = namespace

    @contextlib.asynccontextmanager
    async def api_client(self) -> Generator[Any, None, None]:
        """The API client for the Kubernetes object. This is determined
        by the ``apiVersion`` of the object configuration.

        Raises:
            ValueError: The API version is not supported.
        """
        c = self.api_clients.get(self.api_version)
        # If we didn't find the client in the api_clients dict, use the
        # preferred version.
        if c is None:
            self.logger.debug(
                f"unknown API version ({self.api_version}) for {self.__class__.__name__}, falling back to preferred version"
            )
            c = self.api_clients.get("preferred")
            if c is None:
                raise ValueError(
                    "unknown version specified and no preferred version "
                    f"defined for resource ({self.api_version})"
                )
        # If we did find it, initialize that client version.
        async with kubernetes_asyncio.client.api_client.ApiClient() as api:
            yield c(api)

    @classmethod
    @contextlib.asynccontextmanager
    async def preferred_client(cls) -> Generator[Any, None, None]:
        """The preferred API client type for the Kubernetes object. This is defined in the
        ``api_clients`` class member dict for each object.

        Raises:
             ValueError: No preferred client is defined for the object.
        """
        c = cls.api_clients.get("preferred")
        if c is None:
            raise ValueError(
                f"no preferred api client defined for object {cls.__name__}",
            )
        async with kubernetes_asyncio.client.api_client.ApiClient() as api:
            yield c(api)

    @abc.abstractclassmethod
    async def read(cls, name: str, namespace: str) -> "KubernetesModel":
        """Read the underlying Kubernetes resource from the cluster and
        return a model instance.

        Args:
            name: The name of the resource to read.
            namespace: The namespace to read the resource from.
        """

    @abc.abstractmethod
    async def create(self, namespace: str = None) -> None:
        """Create the underlying Kubernetes resource in the cluster
        under the given namespace.

        Args:
            namespace: The namespace to create the resource under.
                If no namespace is provided, it will use the instance's
                namespace member, which is set when the object is created
                via the kubernetes_asyncio.client
        """

    @abc.abstractmethod
    async def patch(self) -> None:
        """Partially update the underlying Kubernetes resource in the cluster."""

    @abc.abstractmethod
    async def delete(self, options:kubernetes_asyncio.client.V1DeleteOptions) -> kubernetes_asyncio.client.V1Status:
        """Delete the underlying Kubernetes resource from the cluster.

        This method expects the resource to have been loaded or otherwise
        assigned a namespace already. If it has not, the namespace will need
        to be set manually.

        Args:
            options: Options for resource deletion.
        """

    @abc.abstractmethod
    async def refresh(self) -> None:
        """Refresh the local state (``obj``) of the underlying Kubernetes resource."""

    @abc.abstractmethod
    async def is_ready(self) -> bool:
        """Check if the resource is in the ready state.

        It is up to the wrapper subclass to define what "ready" means for
        that particular resource.

        Returns:
            True if in the ready state; False otherwise.
        """

    async def wait_until_ready(
        self,
        interval: servo.DurationDescriptor = 1,
        fail_on_api_error: bool = False,
    ) -> None:
        """Wait until the resource is in the ready state.

        Args:
            timeout: The maximum time to wait, in seconds, for the resource
                to reach the ready state. If unspecified, this will wait
                indefinitely. If specified and the timeout is met or exceeded,
                a TimeoutError will be raised.
            interval: The time, in seconds, to wait before re-checking if the
                object is ready.
            fail_on_api_error: Fail if an API error is raised. An API error can
                be raised for a number of reasons, such as 'resource not found',
                which could be the case when a resource is just being started or
                restarted. When waiting for readiness we generally do not want to
                fail on these conditions.

        Raises:
             TimeoutError: The specified timeout was exceeded.
        """
        ready_condition = Condition(
            "api object ready",
            self.is_ready,
        )

        task = asyncio.create_task(
            wait_for_condition(
                condition=ready_condition,
                interval=interval,
                fail_on_api_error=fail_on_api_error,
            )
        )
        try:
            await task
        except asyncio.CancelledError:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
            raise

    async def wait_until_deleted(
        self,
        interval: servo.DurationDescriptor = 1
    ) -> None:
        """Wait until the resource is deleted from the cluster.

        Args:
            timeout: The maximum time to wait, in seconds, for the resource to
                be deleted from the cluster. If unspecified, this will wait
                indefinitely. If specified and the timeout is met or exceeded,
                a TimeoutError will be raised.
            interval: The time, in seconds, to wait before re-checking if the
                object has been deleted.

        Raises:
            TimeoutError: The specified timeout was exceeded.
        """

        async def deleted_fn():
            try:
                await self.refresh()
            except kubernetes_asyncio.client.exceptions.ApiException as e:
                # If we can no longer find the deployment, it is deleted.
                # If we get any other exception, raise it.
                if e.status == 404 and e.reason == "Not Found":
                    return True
                else:
                    self.logger.error("error refreshing object state")
                    raise e
            else:
                # The object was still found, so it has not been deleted
                return False

        delete_condition = Condition("api object deleted", deleted_fn)

        task = asyncio.create_task(
            wait_for_condition(
                condition=delete_condition,
                interval=interval,
            )
        )

        try:
            await task
        except asyncio.CancelledError:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
            raise

    async def raise_for_status(self) -> None:
        """Raise an exception if in an unhealthy state."""
        self.logger.warning(f"raise_for_status not implemented on {self.__class__.__name__}")

class Namespace(KubernetesModel):
    """Kubetest wrapper around a Kubernetes `Namespace`_ API Object.

    The actual ``kubernetes.client.V1Namespace`` instance that this
    wraps can be accessed via the ``obj`` instance member.

    This wrapper provides some convenient functionality around the
    API Object and provides some state management for the `Namespace`_.

    .. _Namespace:
        https://kubernetes.io/docs/reference/generated/kubernetes-api/v1.18/#namespace-v1-core
    """

    obj:kubernetes_asyncio.client.V1Namespace
    api_clients: ClassVar[Dict[str, Type]] = {
        "preferred":kubernetes_asyncio.client.CoreV1Api,
        "v1":kubernetes_asyncio.client.CoreV1Api,
    }

    @classmethod
    def new(cls, name: str) -> "Namespace":
        """Create a new Namespace with object backing.

        Args:
            name: The name of the new Namespace.

        Returns:
            A new Namespace instance.
        """
        return cls(
            obj=kubernetes_asyncio.client.V1Namespace(
                api_version="v1", metadata=kubernetes_asyncio.client.V1ObjectMeta(name=name)
            )
        )

    @classmethod
    async def read(cls, name: str) -> "Namespace":
        """Read a Namespace from the Kubernetes API.

        Args:
            name: The name of the Namespace to read.

        Returns:
            A hydrated Namespace instance.
        """
        namespace = cls.new(name)
        await namespace.refresh()
        return namespace

    async def create(self, name: str = None) -> None:
        """Create the Namespace under the given name.

        Args:
            name: The name to create the Namespace under. If the
                name is not provided, it will be assumed to already be
                in the underlying object spec. If it is not, namespace
                operations will fail.
        """
        if name is not None:
            self.name = name

        self.logger.info(f'creating namespace "{self.name}"')

        async with self.api_client() as api_client:
            self.obj = await api_client.create_namespace(
                body=self.obj,
            )

    async def patch(self) -> None:
        """
        TODO: Add docs....
        """
        async with self.api_client() as api_client:
            await api_client.patch_namespace(
                name=self.name,
                body=self.obj,
            )

    async def delete(self, options:kubernetes_asyncio.client.V1DeleteOptions = None) -> kubernetes_asyncio.client.V1Status:
        """Delete the Namespace.

        Args:
             options: Options for Namespace deletion.

        Returns:
            The status of the delete operation.
        """
        if options is None:
            options =kubernetes_asyncio.client.V1DeleteOptions()

        self.logger.info(f'deleting namespace "{self.name}"')
        self.logger.debug(f"delete options: {options}")

        async with self.api_client() as api_client:
            return await api_client.delete_namespace(
                name=self.name,
                body=options,
            )

    async def refresh(self) -> None:
        """Refresh the underlying Kubernetes Namespace resource."""
        async with self.api_client() as api_client:
            self.obj = await api_client.read_namespace(
                name=self.name,
            )

    async def is_ready(self) -> bool:
        """Check if the Namespace is in the ready state.

        Returns:
            True if in the ready state; False otherwise.
        """
        await self.refresh()

        status = self.obj.status
        if status is None:
            return False

        return status.phase.lower() == "active"


_DEFAULT_SENTINEL = object()


class Container(servo.logging.Mixin):
    """Kubetest wrapper around a Kubernetes `Container`_ API Object.

    The actual ``kubernetes.client.V1Container`` instance that this
    wraps can be accessed via the ``obj`` instance member.

    This wrapper provides some convenient functionality around the
    API Object and provides some state management for the `Container`_.

    This wrapper does **NOT** subclass the ``objects.ApiObject`` like other
    object wrappers because it is not intended to be created or
    managed from manifest file. It is merely meant to wrap the
    Container spec for a Pod to make Container-targeted actions
    easier.

    .. _Container:
        https://kubernetes.io/docs/reference/generated/kubernetes-api/v1.18/#container-v1-core
    """

    def __init__(self, api_object, pod) -> None: # noqa: D107
        self.obj = api_object
        self.pod = pod

    @property
    def name(self) -> str:
        return self.obj.name

    @property
    def image(self) -> str:
        """
        Returns the container image name from the underlying container object.
        """
        return self.obj.image

    async def get_restart_count(self) -> int:
        """Get the number of times the Container has been restarted.

        Returns:
            The number of times the Container has been restarted.
        """
        container_name = self.obj.name
        pod_status = await self.pod.get_status()

        # If there are no container status, the container hasn't started
        # yet, so there cannot be any restarts.
        if pod_status.container_statuses is None:
            return 0

        for status in pod_status.container_statuses:
            if status.name == container_name:
                return status.restart_count

        raise RuntimeError(f"Unable to determine container status for {container_name}")

    @property
    def resources(self) -> kubernetes_asyncio.client.V1ResourceRequirements:
        """
        Return the resource requirements for the Container.

        Returns:
            The Container resource requirements.
        """
        return self.obj.resources

    @resources.setter
    def resources(self, resources: kubernetes_asyncio.client.V1ResourceRequirements) -> None:
        """
        Set the resource requirements for the Container.

        Args:
            resources: The resource requirements to set.
        """
        self.obj.resources = resources

    def get_resource_requirements(self, name: str) -> Dict[ResourceRequirement, Optional[str]]:
        """Return a dictionary mapping resource requirements to values for a given resource (e.g., cpu or memory).

        This method is safe to call for containers that do not define any resource requirements (e.g., the `resources` property is None).

        Requirements that are not defined for the named resource are returned as None. For example, a container
        that defines CPU requests but does not define limits would return a dict with a `None` value for
        the `ResourceRequirement.limit` key.

        Args:
            name: The name of the resource to set the requirements of (e.g., "cpu" or "memory").

        Returns:
            A dictionary mapping ResourceRequirement enum members to optional string values.
        """
        resources: kubernetes_asyncio.client.V1ResourceRequirements = getattr(self, 'resources', kubernetes_asyncio.client.V1ResourceRequirements())
        requirements = {}
        for requirement in ResourceRequirement:
            # Get the 'requests' or 'limits' nested structure
            requirement_subdict = getattr(resources, requirement.resources_key, {})
            if requirement_subdict:
                requirements[requirement] = requirement_subdict.get(name)
            else:
                requirements[requirement] = None

        return requirements

    def set_resource_requirements(self, name: str, requirements: Dict[ResourceRequirement, Optional[str]]) -> None:
        """Sets resource requirements on the container for the values in the given dictionary.

        If no resources have been defined yet, a resources model is provisioned.
        If no requirements have been defined for the given resource name, a requirements dictionary is defined.
        Values of None are removed from the target requirements.
        ResourceRequirement keys that are not present in the dict are not modified.

        Args:
            name: The name of the resource to set the requirements of (e.g., "cpu" or "memory").
            requirements: A dict mapping requirements to target values (e.g., `{ResourceRequirement.request: '500m', ResourceRequirement.limit: '2000m'})
        """
        resources: kubernetes_asyncio.client.V1ResourceRequirements = copy.copy(
            getattr(self, 'resources', kubernetes_asyncio.client.V1ResourceRequirements())
        )

        for requirement, value in requirements.items():
            resource_to_values = getattr(resources, requirement.resources_key, {})
            if not resource_to_values:
                resource_to_values = {}

            if value is not None:
                # NOTE: Coerce to string as values are headed into Kubernetes resource model
                resource_to_values[name] = str(value)
            else:
                resource_to_values.pop(name, None)
            setattr(resources, requirement.resources_key, resource_to_values)

        self.resources = resources

    @property
    def ports(self) -> List[kubernetes_asyncio.client.V1ContainerPort]:
        """
        Return the ports for the Container.

        Returns:
            The Container ports.
        """
        return self.obj.ports or []

    def __str__(self) -> str:
        return str(self.obj)

    def __repr__(self) -> str:
        return self.__str__()


class Pod(KubernetesModel):
    """Wrapper around a Kubernetes `Pod`_ API Object.

    The actual ``kubernetes.client.V1Pod`` instance that this
    wraps can be accessed via the ``obj`` instance member.

    This wrapper provides some convenient functionality around the
    API Object and provides some state management for the `Pod`_.

    .. _Pod:
        https://kubernetes.io/docs/reference/generated/kubernetes-api/v1.18/#pod-v1-core
    """

    obj:kubernetes_asyncio.client.V1Pod

    api_clients: ClassVar[Dict[str, Type]] = {
        "preferred":kubernetes_asyncio.client.CoreV1Api,
        "v1":kubernetes_asyncio.client.CoreV1Api,
    }

    @classmethod
    async def read(cls, name: str, namespace: str) -> "Pod":
        """Read the Pod from the cluster under the given namespace.

        Args:
            name: The name of the Pod to read.
            namespace: The namespace to read the Pod from.
        """
        servo.logger.debug(f'reading pod "{name}" in namespace "{namespace}"')

        async with cls.preferred_client() as api_client:
            obj = await api_client.read_namespaced_pod_status(name, namespace)
            return Pod(obj)

    async def create(self, namespace: str = None) -> None:
        """Create the Pod under the given namespace.

        Args:
            namespace: The namespace to create the Pod under.
                If the Pod was loaded via the kubetest client, the
                namespace will already be set, so it is not needed
                here. Otherwise, the namespace will need to be provided.
        """
        if namespace is None:
            namespace = self.namespace

        self.logger.info(f'creating pod "{self.name}" in namespace "{namespace}"')

        async with self.preferred_client() as api_client:
            self.obj = await api_client.create_namespaced_pod(
                namespace=namespace,
                body=self.obj,
            )

    async def patch(self) -> None:
        """
        Patches a Pod, applying spec changes to the cluster.
        """
        self.logger.info(f'patching pod "{self.name}"')
        async with self.api_client() as api_client:
            api_client.api_client.set_default_header('content-type', 'application/strategic-merge-patch+json')
            await api_client.patch_namespaced_pod(
                name=self.name,
                namespace=self.namespace,
                body=self.obj,
            )

    async def delete(self, options:kubernetes_asyncio.client.V1DeleteOptions = None) ->kubernetes_asyncio.client.V1Status:
        """Delete the Pod.

        This method expects the Pod to have been loaded or otherwise
        assigned a namespace already. If it has not, the namespace will
        need to be set manually.

        Args:
            options: Options for Pod deletion.

        Return:
            The status of the delete operation.
        """
        if options is None:
            options =kubernetes_asyncio.client.V1DeleteOptions()

        self.logger.info(f'deleting pod "{self.name}"')
        self.logger.trace(f"delete options: {options}")

        async with self.api_client() as api_client:
            return await api_client.delete_namespaced_pod(
                name=self.name,
                namespace=self.namespace,
                body=options,
            )

    async def refresh(self) -> None:
        """Refresh the underlying Kubernetes Pod resource."""
        async with self.api_client() as api_client:
            self.obj = await api_client.read_namespaced_pod_status(
                name=self.name,
                namespace=self.namespace,
            )

    async def is_ready(self) -> bool:
        """Check if the Pod is in the ready state.

        Returns:
            True if in the ready state; False otherwise.
        """
        self.logger.trace("refreshing pod status to check is_ready")
        await self.refresh()

        # if there is no status, the pod is definitely not ready
        status = self.obj.status
        self.logger.trace(f"current pod status is {status}")
        if status is None:
            return False

        # check the pod phase to make sure it is running. a pod in
        # the 'failed' or 'success' state will no longer be running,
        # so we only care if the pod is in the 'running' state.
        status.phase
        self.logger.trace(f"current pod phase is {status}")
        if not status.conditions:
            return False

        self.logger.trace(f"checking status conditions {status.conditions}")
        for cond in status.conditions:
            if cond.reason == "Unschedulable":
                return False

            # we only care about the condition type 'ready'
            if cond.type.lower() != "ready":
                continue

            # check that the readiness condition is True
            return cond.status.lower() == "true"

        # Catchall
        self.logger.trace(f"unable to find ready=true, continuing to wait...")
        return False

    async def raise_for_status(self, adjustments: List[servo.Adjustment]) -> None:
        """Raise an exception if the Pod status is not not ready."""
        # NOTE: operate off of current state, assuming you have checked is_ready()
        status = self.obj.status
        self.logger.trace(f"current pod status is {status}")
        if status is None:
            raise RuntimeError(f'No such pod: {self.name}')

        # check the pod phase to make sure it is running. a pod in
        # the 'failed' or 'success' state will no longer be running,
        # so we only care if the pod is in the 'running' state.
        # phase = status.phase
        if not status.conditions:
            raise RuntimeError(f'Pod is not running: {self.name}')

        self.logger.trace(f"checking container statuses: {status.container_statuses}")
        if status.container_statuses:
            for cont_stat in status.container_statuses:
                if cont_stat.state and cont_stat.state.waiting and cont_stat.state.waiting.reason in ["ImagePullBackOff", "ErrImagePull"]:
                    raise servo.AdjustmentFailedError("Container image pull failure detected", reason="image-pull-failed")

        restarted_container_statuses = list(filter(lambda cont_stat: cont_stat.restart_count > 0, (status.container_statuses or [])))
        if restarted_container_statuses:
            container_messages = list(map(lambda cont_stat: f"{cont_stat.name} x{cont_stat.restart_count}", restarted_container_statuses))
            raise servo.AdjustmentRejectedError(
                f"Tuning optimization {self.name} crash restart detected on container(s): {', '.join(container_messages)}",
                reason="unstable"
            )

        self.logger.trace(f"checking status conditions {status.conditions}")
        for cond in status.conditions:
            if cond.reason == "Unschedulable":
                # FIXME: The servo rejected error should be raised further out. This should be a generic scheduling error
                unschedulable_adjustments = list(filter(lambda a: a.setting_name in cond.message, adjustments))
                raise servo.AdjustmentRejectedError(
                    f"Requested adjustment(s) ({', '.join(map(str, unschedulable_adjustments))}) cannot be scheduled due to \"{cond.message}\"",
                    reason="unschedulable"
                )

            if cond.type == "Ready" and cond.status == "False":
                raise servo.AdjustmentRejectedError(f"(reason {cond.reason}) {cond.message}", reason="start-failed")

            # we only care about the condition type 'ready'
            if cond.type.lower() != "ready":
                continue

            # check that the readiness condition is True
            if cond.status.lower() == "true":
                return

        # Catchall
        self.logger.trace(f"unable to find ready=true, continuing to wait...")
        raise RuntimeError(f"Unknown Pod status for '{self.name}': {status}")

    async def get_status(self) ->kubernetes_asyncio.client.V1PodStatus:
        """Get the status of the Pod.

        Returns:
            The status of the Pod.
        """
        # first, refresh the pod state to ensure latest status
        await self.refresh()

        # return the status of the pod
        return cast(kubernetes_asyncio.client.V1PodStatus, self.obj.status)

    @property
    def containers(self) -> List[Container]:
        """
        Return a list of Container objects from the underlying pod template spec.
        """
        return list(map(lambda c: Container(c, self), self.obj.spec.containers))

    async def get_containers(self) -> List[Container]:
        """Get the Pod's containers.

        Returns:
            A list of containers that belong to the Pod.
        """
        self.logger.debug(f'getting containers for pod "{self.name}"')
        await self.refresh()

        return self.containers

    def get_container(self, name: str) -> Union[Container, None]:
        """Get a container in the Pod by name.

        Args:
            name (str): The name of the Container.

        Returns:
            Container: The Pod's Container with the matching name. If
            no container with the given name is found, ``None`` is returned.
        """
        return next(filter(lambda c: c.name == name, self.containers), None)

    async def get_restart_count(self) -> int:
        """Get the total number of Container restarts for the Pod.

        Returns:
            The total number of Container restarts.
        """
        status = await self.get_status()
        if status.container_statuses is None:
            return 0

        total = 0
        for container_status in status.container_statuses:
            total += container_status.restart_count

        return total

    async def containers_started(self) -> bool:
        """Check if the Pod's Containers have all started.

        Returns:
            True if all Containers have started; False otherwise.
        """
        # start the flag as true - we will check the state and set
        # this to False if any container is not yet running.
        containers_started = True

        status = await self.get_status()
        if status.container_statuses is not None:
            for container_status in status.container_statuses:
                if container_status.state is not None:
                    if container_status.state.running is not None:
                        if container_status.state.running.started_at is not None:
                            # The container is started, so move on to check the
                            # next container
                            continue
                # If we get here, then the container has not started.
                containers_started = containers_started and False
                break

        return containers_started

    def uid(self) -> str:
        """
        Gets the UID for the Pod.

        UID is the unique in time and space value for this object. It is typically generated by the server on successful creation of a resource and is not allowed to change on PUT operations.  Populated by the system. Read-only. More info: http://kubernetes.io/docs/user-guide/identifiers#uids  # noqa: E501
        """
        return self.obj.metadata.uid


class Service(KubernetesModel):
    """Kubetest wrapper around a Kubernetes `Service`_ API Object.

    The actual ``kubernetes.client.V1Service`` instance that this
    wraps can be accessed via the ``obj`` instance member.

    This wrapper provides some convenient functionality around the
    API Object and provides some state management for the `Service`_.

    .. _Service:
        https://kubernetes.io/docs/reference/generated/kubernetes-api/v1.18/#service-v1-core
    """

    obj: kubernetes_asyncio.client.V1Service

    api_clients: ClassVar[Dict[str, Type]] = {
        'preferred': kubernetes_asyncio.client.CoreV1Api,
        'v1': kubernetes_asyncio.client.CoreV1Api,
    }

    @classmethod
    async def read(cls, name: str, namespace: str) -> "Service":
        """Read the Service from the cluster under the given namespace.

        Args:
            name: The name of the Service to read.
            namespace: The namespace to read the Service from.
        """
        servo.logger.trace(f'reading service "{name}" in namespace "{namespace}"')

        async with cls.preferred_client() as api_client:
            obj = await api_client.read_namespaced_service(name, namespace)
            servo.logger.trace("service: ", obj)
            return Service(obj)

    async def create(self, namespace: str = None) -> None:
        """Creates the Service under the given namespace.

        Args:
            namespace: The namespace to create the Service under.
                If the Service was loaded via the kubetest client, the
                namespace will already be set, so it is not needed here.
                Otherwise, the namespace will need to be provided.
        """
        if namespace is None:
            namespace = self.namespace

        self.logger.info(f'creating service "{self.name}" in namespace "{self.namespace}"')

        async with self.api_client() as api_client:
            self.obj = await api_client.create_namespaced_service(
                namespace=namespace,
                body=self.obj,
            )

    async def patch(self) -> None:
        """
        TODO: Add docs....
        """
        async with self.api_client() as api_client:
            api_client.api_client.set_default_header('content-type', 'application/strategic-merge-patch+json')
            await api_client.patch_namespaced_service(
                name=self.name,
                namespace=self.namespace,
                body=self.obj,
            )

    async def delete(self, options: kubernetes_asyncio.client.V1DeleteOptions = None) -> kubernetes_asyncio.client.V1Status:
        """Deletes the Service.

        This method expects the Service to have been loaded or otherwise
        assigned a namespace already. If it has not, the namespace will need
        to be set manually.

        Args:
            options: Options for Service deletion.

        Returns:
            The status of the delete operation.
        """
        if options is None:
            options = kubernetes_asyncio.client.V1DeleteOptions()

        self.logger.info(f'deleting service "{self.name}"')
        self.logger.debug(f'delete options: {options}')

        async with self.api_client() as api_client:
            return await api_client.delete_namespaced_service(
                name=self.name,
                namespace=self.namespace,
                body=options,
            )

    async def refresh(self) -> None:
        """Refresh the underlying Kubernetes Service resource."""
        async with self.api_client() as api_client:
            self.obj = await api_client.read_namespaced_service(
                name=self.name,
                namespace=self.namespace,
            )

    async def is_ready(self) -> bool:
        """Check if the Service is in the ready state.

        The readiness state is not clearly available from the Service
        status, so to see whether or not the Service is ready this
        will check whether the endpoints of the Service are ready.

        This comes with the caveat that in order for a Service to
        have endpoints, there needs to be some backend hooked up to it.
        If there is no backend, the Service will never have endpoints,
        so this will never resolve to True.

        Returns:
            True if in the ready state; False otherwise.
        """
        await self.refresh()

        # check the status. if there is no status, the service is
        # definitely not ready.
        if self.obj.status is None:
            return False

        endpoints = await self.get_endpoints()

        # if the Service has no endpoints, its not ready.
        if len(endpoints) == 0:
            return False

        # get the service endpoints and check that they are all ready.
        for endpoint in endpoints:
            # if we have an endpoint, but there are no subsets, we
            # consider the endpoint to be not ready.
            if endpoint.subsets is None:
                return False

            for subset in endpoint.subsets:
                # if the endpoint has no addresses setup yet, its not ready
                if subset.addresses is None or len(subset.addresses) == 0:
                    return False

                # if there are still addresses that are not ready, the
                # service is not ready
                not_ready = subset.not_ready_addresses
                if not_ready is not None and len(not_ready) > 0:
                    return False

        # if we got here, then all endpoints are ready, so the service
        # must also be ready
        return True

    @property
    def status(self) -> kubernetes_asyncio.client.V1ServiceStatus:
        return self.obj.status

    async def get_status(self) -> kubernetes_asyncio.client.V1ServiceStatus:
        """Get the status of the Service.

        Returns:
            The status of the Service.
        """
        self.logger.info(f'checking status of service "{self.name}"')
        # first, refresh the service state to ensure the latest status
        await self.refresh()

        # return the status from the service
        return self.obj.status

    @property
    def ports(self) -> List[kubernetes_asyncio.client.V1ServicePort]:
        """Return the list of ports exposed by the service."""
        return self.obj.spec.ports

    def find_port(self, selector: Union[str, int]) -> Optional[kubernetes_asyncio.client.V1ServicePort]:
        for port in self.ports:
            if isinstance(selector, str):
                if port.name == selector:
                    return port
            elif isinstance(selector, int):
                if port.port == selector:
                    return port
            else:
                raise TypeError(f"Unknown port selector type '{selector.__class__.__name__}': {selector}")

        return None

    async def get_endpoints(self) -> List[kubernetes_asyncio.client.V1Endpoints]:
        """Get the endpoints for the Service.

        This can be useful for checking internal IP addresses used
        in containers, e.g. for container auto-discovery.

        Returns:
            A list of endpoints associated with the Service.
        """
        self.logger.info(f'getting endpoints for service "{self.name}"')
        async with self.api_client() as api_client:
            endpoints = await api_client.list_namespaced_endpoints(
                namespace=self.namespace,
            )

        svc_endpoints = []
        for endpoint in endpoints.items:
            # filter to include only the endpoints with the same
            # name as the service.
            if endpoint.metadata.name == self.name:
                svc_endpoints.append(endpoint)

        self.logger.debug(f'endpoints: {svc_endpoints}')
        return svc_endpoints

    async def _proxy_http_request(self, method, path, **kwargs) -> tuple:
        """Template request to proxy of a Service.

        Args:
            method: The http request method e.g. 'GET', 'POST' etc.
            path: The URI path for the request.
            kwargs: Keyword arguments for the proxy_http_get function.

        Returns:
            The response data
        """
        path_params = {
            "name": f'{self.name}:{self.obj.spec.ports[0].port}',
            "namespace": self.namespace,
            "path": path
        }
        return await kubernetes_asyncio.client.CoreV1Api().api_client.call_api(
            '/api/v1/namespaces/{namespace}/services/{name}/proxy/{path}',
            method,
            path_params=path_params,
            **kwargs
        )

    async def proxy_http_get(self, path: str, **kwargs) -> tuple:
        """Issue a GET request to proxy of a Service.

        Args:
            path: The URI path for the request.
            kwargs: Keyword arguments for the proxy_http_get function.

        Returns:
            The response data
        """
        return await self._proxy_http_request('GET', path, **kwargs)

    async def proxy_http_post(self, path: str, **kwargs) -> tuple:
        """Issue a POST request to proxy of a Service.

        Args:
            path: The URI path for the request.
            kwargs: Keyword arguments for the proxy_http_post function.

        Returns:
            The response data
        """
        return await self._proxy_http_request('POST', path, **kwargs)

    @property
    def selector(self) -> Dict[str, str]:
        return self.obj.spec.selector

    async def get_pods(self) -> List[Pod]:
        """Get the pods that the Service is routing traffic to.

        Returns:
            A list of pods that the service is routing traffic to.
        """
        self.logger.debug(f'getting pods for service "{self.name}"')

        async with Pod.preferred_client() as api_client:
            self.obj.spec.selector.match_labels
            pod_list:kubernetes_asyncio.client.V1PodList = await api_client.list_namespaced_pod(
                namespace=self.namespace, label_selector=selector_string(self.selector)
            )

        pods = [Pod(p) for p in pod_list.items]
        return pods


class WatchTimeoutError(Exception):
    """The kubernetes watch timeout has elapsed. The api client raises no error
    on timeout expiration so this should be raised in fall-through logic.
    """

class Deployment(KubernetesModel):
    """Kubetest wrapper around a Kubernetes `Deployment`_ API Object.

    The actual ``kubernetes.client.V1Deployment`` instance that this
    wraps can be accessed via the ``obj`` instance member.

    This wrapper provides some convenient functionality around the
    API Object and provides some state management for the `Deployment`_.

    .. _Deployment:
        https://kubernetes.io/docs/reference/generated/kubernetes-api/v1.18/#deployment-v1-apps
    """

    obj:kubernetes_asyncio.client.V1Deployment
    api_clients: ClassVar[Dict[str, Type]] = {
        "preferred":kubernetes_asyncio.client.AppsV1Api,
        "apps/v1":kubernetes_asyncio.client.AppsV1Api,
        "apps/v1beta1":kubernetes_asyncio.client.AppsV1beta1Api,
        "apps/v1beta2":kubernetes_asyncio.client.AppsV1beta2Api,
    }

    async def create(self, namespace: str = None) -> None:
        """Create the Deployment under the given namespace.

        Args:
            namespace: The namespace to create the Deployment under.
                If the Deployment was loaded via the kubetest client, the
                namespace will already be set, so it is not needed here.
                Otherwise, the namespace will need to be provided.
        """
        if namespace is None:
            namespace = self.namespace

        self.logger.info(
            f'creating deployment "{self.name}" in namespace "{self.namespace}"'
        )

        async with self.api_client() as api_client:
            self.obj = await api_client.create_namespaced_deployment(
                namespace=namespace,
                body=self.obj,
            )

    @classmethod
    async def read(cls, name: str, namespace: str) -> "Deployment":
        """Read a Deployment by name under the given namespace.

        Args:
            name: The name of the Deployment to read.
            namespace: The namespace to read the Deployment from.
        """

        async with cls.preferred_client() as api_client:
            obj = await api_client.read_namespaced_deployment(name, namespace)
            return Deployment(obj)

    async def patch(self) -> None:
        """Update the changed attributes of the Deployment."""
        async with self.api_client() as api_client:
            api_client.api_client.set_default_header('content-type', 'application/strategic-merge-patch+json')
            self.obj = await api_client.patch_namespaced_deployment(
                name=self.name,
                namespace=self.namespace,
                body=self.obj
            )

    async def replace(self) -> None:
        """Update the changed attributes of the Deployment."""
        async with self.api_client() as api_client:
            self.obj = await api_client.replace_namespaced_deployment(
                name=self.name,
                namespace=self.namespace,
                body=self.obj
            )

    async def delete(self, options:kubernetes_asyncio.client.V1DeleteOptions = None) ->kubernetes_asyncio.client.V1Status:
        """Delete the Deployment.

        This method expects the Deployment to have been loaded or otherwise
        assigned a namespace already. If it has not, the namespace will need
        to be set manually.

        Args:
            options: Options for Deployment deletion.

        Returns:
            The status of the delete operation.
        """
        if options is None:
            options =kubernetes_asyncio.client.V1DeleteOptions()

        self.logger.info(f'deleting deployment "{self.name}"')
        self.logger.debug(f"delete options: {options}")

        async with self.api_client() as api_client:
            return await api_client.delete_namespaced_deployment(
                name=self.name,
                namespace=self.namespace,
                body=options,
            )

    async def refresh(self) -> None:
        """Refresh the underlying Kubernetes Deployment resource."""
        async with self.api_client() as api_client:
            self.obj = await api_client.read_namespaced_deployment_status(
                name=self.name,
                namespace=self.namespace,
            )

    async def rollback(self) -> None:
        """Roll back an unstable Deployment revision to a previous version."""
        async with kubernetes_asyncio.client.api_client.ApiClient() as api:
            api_client =kubernetes_asyncio.client.ExtensionsV1beta1Api(api)
            self.obj = await api_client.create_namespaced_deployment_rollback(
                name=self.name,
                namespace=self.namespace,
                body=self.obj,
            )

    async def get_status(self) ->kubernetes_asyncio.client.V1DeploymentStatus:
        """Get the status of the Deployment.

        Returns:
            The status of the Deployment.
        """
        self.logger.info(f'checking status of deployment "{self.name}"')
        # first, refresh the deployment state to ensure the latest status
        await self.refresh()

        # return the status from the deployment
        return cast(kubernetes_asyncio.client.V1DeploymentStatus, self.obj.status)

    async def get_pods(self) -> List[Pod]:
        """Get the pods for the Deployment.

        Returns:
            A list of pods that belong to the deployment.
        """
        self.logger.debug(f'getting pods for deployment "{self.name}"')

        async with Pod.preferred_client() as api_client:
            label_selector = self.obj.spec.selector.match_labels
            pod_list:kubernetes_asyncio.client.V1PodList = await api_client.list_namespaced_pod(
                namespace=self.namespace, label_selector=selector_string(label_selector)
            )

        pods = [Pod(p) for p in pod_list.items]
        return pods

    async def get_latest_pods(self) -> List[Pod]:
        """Get only the Deployment pods that belong to the latest ResourceVersion.

        Returns:
            A list of pods that belong to the latest deployment replicaset.
        """
        self.logger.trace(f'getting replicaset for deployment "{self.name}"')
        async with self.api_client() as api_client:
            label_selector = self.obj.spec.selector.match_labels
            rs_list:kubernetes_asyncio.client.V1ReplicasetList = await api_client.list_namespaced_replica_set(
                namespace=self.namespace, label_selector=selector_string(label_selector)
            )

        # Verify all returned RS have this deployment as an owner
        rs_list = [
            rs for rs in rs_list.items if rs.metadata.owner_references and any(
                ownRef.kind == "Deployment" and ownRef.uid == self.obj.metadata.uid
                for ownRef in rs.metadata.owner_references
            )
        ]
        if not rs_list:
            raise servo.ConnectorError(f'Unable to locate replicaset(s) for deployment "{self.name}"')
        if missing_revision_rsets := list(filter(lambda rs: 'deployment.kubernetes.io/revision' not in rs.metadata.annotations, rs_list)):
            raise servo.ConnectorError(
                f'Unable to determine latest replicaset for deployment "{self.name}" due to missing revision annotation in replicaset(s)'
                f' "{", ".join(list(map(lambda rs: rs.metadata.name, missing_revision_rsets)))}"'
            )
        latest_rs = sorted(rs_list, key= lambda rs: int(rs.metadata.annotations['deployment.kubernetes.io/revision']), reverse=True)[0]

        return [
            pod for pod in await self.get_pods()
            if any(
                ownRef.kind == "ReplicaSet" and ownRef.uid == latest_rs.metadata.uid
                for ownRef in pod.obj.metadata.owner_references
            )]



    @property
    def status(self) ->kubernetes_asyncio.client.V1DeploymentStatus:
        """Return the status of the Deployment.

        Returns:
            The status of the Deployment.
        """
        return cast(kubernetes_asyncio.client.V1DeploymentStatus, self.obj.status)

    @property
    def resource_version(self) -> str:
        """
        Returns the resource version of the Deployment.
        """
        return self.obj.metadata.resource_version

    @property
    def observed_generation(self) -> str:
        """
        Returns the observed generation of the Deployment status.

        The generation is observed by the deployment controller.
        """
        return self.obj.status.observed_generation

    async def is_ready(self) -> bool:
        """Check if the Deployment is in the ready state.

        Returns:
            True if in the ready state; False otherwise.
        """
        await self.refresh()

        # if there is no status, the deployment is definitely not ready
        status = self.obj.status
        if status is None:
            return False

        # check the status for the number of total replicas and compare
        # it to the number of ready replicas. if the numbers are
        # equal, the deployment is ready; otherwise it is not ready.
        total = status.replicas
        ready = status.ready_replicas

        if total is None:
            return False

        return total == ready

    @property
    def containers(self) -> List[Container]:
        """
        Return a list of Container objects from the underlying pod template spec.
        """
        return list(
            map(lambda c: Container(c, None), self.obj.spec.template.spec.containers)
        )

    def find_container(self, name: str) -> Optional[Container]:
        """
        Return the container with the given name.
        """
        return next(filter(lambda c: c.name == name, self.containers), None)

    async def get_target_container(self, config: ContainerConfiguration) -> Optional[Container]:
        """Return the container targeted by the supplied configuration"""
        return self.find_container(config.name)

    def set_container(self, name: str, container: Container) -> None:
        """Set the container with the given name to a new value."""
        index = next(filter(lambda i: self.containers[i].name == name, range(len(self.containers))))
        self.containers[index] = container
        self.obj.spec.template.spec.containers[index] = container.obj

    def remove_container(self, name: str) -> Optional[Container]:
        """Set the container with the given name to a new value."""
        index = next(filter(lambda i: self.containers[i].name == name, range(len(self.containers))), None)
        if index is not None:
            return Container(
                self.obj.spec.template.spec.containers.pop(index),
                None
            )

        return None

    @property
    def replicas(self) -> int:
        """
        Return the number of desired pods.
        """
        return self.obj.spec.replicas

    @replicas.setter
    def replicas(self, replicas: int) -> None:
        """
        Set the number of desired pods.
        """
        self.obj.spec.replicas = replicas

    @property
    def label_selector(self) -> str:
        """
        Return a string for matching the Deployment in Kubernetes API calls.
        """
        return selector_string(self.obj.metadata.labels)

    # TODO: I need to model these two and add label/annotation helpers
    @property
    def pod_template_spec(self) -> kubernetes_asyncio.client.models.V1PodTemplateSpec:
        """Return the pod template spec for instances of the Deployment."""
        return self.obj.spec.template

    async def get_pod_template_spec_copy(self) -> kubernetes_asyncio.client.models.V1PodTemplateSpec:
        """Return a deep copy of the pod template spec. Eg. for creation of a tuning pod"""
        return copy.deepcopy(self.pod_template_spec)

    def update_pod(self, pod: kubernetes_asyncio.client.models.V1Pod) -> kubernetes_asyncio.client.models.V1Pod:
        """Update the pod with the latest state of the controller if needed"""
        # NOTE: Deployment currently needs no updating
        return pod

    @property
    def pod_spec(self) -> kubernetes_asyncio.client.models.V1PodSpec:
        """Return the pod spec for instances of the Deployment."""
        return self.pod_template_spec.spec

    @backoff.on_exception(backoff.expo, kubernetes_asyncio.client.exceptions.ApiException, max_tries=3)
    async def inject_sidecar(
        self,
        name: str,
        image: str,
        *,
        service: Optional[str] = None,
        port: Optional[int] = None,
        index: Optional[int] = None,
        service_port: int = 9980
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

        await self.refresh()

        if not (service or port):
            raise ValueError(f"a service or port must be given")

        if isinstance(port, str) and port.isdigit():
            port = int(port)

        # check for a port conflict
        container_ports = list(itertools.chain(*map(operator.attrgetter("ports"), self.containers)))
        if service_port in list(map(operator.attrgetter("container_port"), container_ports)):
            raise ValueError(f"Port conflict: Deployment '{self.name}' already exposes port {service_port} through an existing container")

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
                    raise ValueError(f"Target Service '{service}' does not expose any ports")
                elif port_count > 1:
                    raise ValueError(f"Target Service '{service}' exposes multiple ports -- target port must be specified")
                port_obj = service_obj.obj.spec.ports[0]
            else:
                if isinstance(port, int):
                    port_obj = next(filter(lambda p: p.port == port, service_obj.obj.spec.ports), None)
                elif isinstance(port, str):
                    port_obj = next(filter(lambda p: p.name == port, service_obj.obj.spec.ports), None)
                else:
                    raise TypeError(f"Unable to resolve port value of type {port.__class__} (port={port})")

                if not port_obj:
                    raise ValueError(f"Port '{port}' does not exist in the Service '{service}'")

            # resolve symbolic name in the service target port to a concrete container port
            if isinstance(port_obj.target_port, str):
                container_port_obj = next(filter(lambda p: p.name == port_obj.target_port, container_ports), None)
                if not container_port_obj:
                    raise ValueError(f"Port '{port_obj.target_port}' could not be resolved to a destination container port")

                container_port = container_port_obj.container_port
            else:
                container_port = port_obj.target_port

        else:
            # find the container port
            container_port_obj = next(filter(lambda p: p.container_port == port, container_ports), None)
            if not container_port_obj:
                raise ValueError(f"Port '{port}' could not be resolved to a destination container port")

            container_port = container_port_obj.container_port

        # build the sidecar container
        container = kubernetes_asyncio.client.V1Container(
            name=name,
            image=image,
            image_pull_policy="IfNotPresent",
            resources=kubernetes_asyncio.client.V1ResourceRequirements(
                requests={
                    "cpu": "125m",
                    "memory": "128Mi"
                },
                limits={
                    "cpu": "250m",
                    "memory": "256Mi"
                }
            ),
            env=[
                kubernetes_asyncio.client.V1EnvVar(name="OPSANI_ENVOY_PROXY_SERVICE_PORT", value=str(service_port)),
                kubernetes_asyncio.client.V1EnvVar(name="OPSANI_ENVOY_PROXIED_CONTAINER_PORT", value=str(container_port)),
                kubernetes_asyncio.client.V1EnvVar(name="OPSANI_ENVOY_PROXY_METRICS_PORT", value="9901")
            ],
            ports=[
                kubernetes_asyncio.client.V1ContainerPort(name="opsani-proxy", container_port=service_port),
                kubernetes_asyncio.client.V1ContainerPort(name="opsani-metrics", container_port=9901),
            ]
        )

        # add the sidecar to the Deployment
        if index is None:
            self.obj.spec.template.spec.containers.append(container)
        else:
            self.obj.spec.template.spec.containers.insert(index, container)

        # patch the deployment
        await self.patch()

    async def eject_sidecar(self, name: str) -> bool:
        """Eject an Envoy sidecar from the Deployment.

        Returns True if the sidecar was ejected.
        """
        await self.refresh()
        container = self.remove_container(name)
        if container:
            await self.replace()
            return True

        return False

    @contextlib.asynccontextmanager
    async def rollout(self, *, timeout: Optional[servo.DurationDescriptor] = None) -> None:
        """Asynchronously wait for changes to a deployment to roll out to the cluster."""
        # NOTE: The timeout_seconds argument must be an int or the request will fail
        timeout_seconds = int(servo.Duration(timeout).total_seconds()) if timeout else None

        # Resource version lets us track any change. Observed generation only increments
        # when the deployment controller sees a significant change that requires rollout
        resource_version = self.resource_version
        observed_generation = self.status.observed_generation
        desired_replicas = self.replicas

        self.logger.info(f"applying adjustments to Deployment '{self.name}' and rolling out to cluster")

        # Yield to let the changes be made
        yield self

        # Return fast if nothing was changed
        if self.resource_version == resource_version:
            self.logger.info(
                f"adjustments applied to Deployment '{self.name}' made no changes, continuing"
            )
            return

        # Create a Kubernetes watch against the deployment under optimization to track changes
        self.logger.debug(
            f"watching deployment Using label_selector={self.label_selector}, resource_version={resource_version}"
        )

        async with kubernetes_asyncio.client.api_client.ApiClient() as api:
            v1 = kubernetes_asyncio.client.AppsV1Api(api)
            async with kubernetes_asyncio.watch.Watch().stream(
                v1.list_namespaced_deployment,
                self.namespace,
                label_selector=self.label_selector,
                timeout_seconds=timeout_seconds,
            ) as stream:
                async for event in stream:
                    # NOTE: Event types are ADDED, DELETED, MODIFIED, ERROR
                    # TODO: Create an enum...
                    event_type, deployment = event["type"], event["object"]
                    status:kubernetes_asyncio.client.V1DeploymentStatus = deployment.status

                    self.logger.debug(
                        f"deployment watch yielded event: {event_type} {deployment.kind} {deployment.metadata.name} in {deployment.metadata.namespace}: {status}"
                    )

                    if event_type == "ERROR":
                        stream.stop()
                        # FIXME: Not sure what types we expect here
                        raise servo.AdjustmentRejectedError(str(deployment), reason="start-failed")

                    # Check that the conditions aren't reporting a failure
                    if status.conditions:
                        self._check_conditions(status.conditions)

                    # Early events in the watch may be against previous generation
                    if status.observed_generation == observed_generation:
                        self.logger.debug(
                            "observed generation has not changed, continuing watch"
                        )
                        continue

                    # Check the replica counts. Once available, updated, and ready match
                    # our expected count and the unavailable count is zero we are rolled out
                    if status.unavailable_replicas:
                        self.logger.debug(
                            "found unavailable replicas, continuing watch",
                            status.unavailable_replicas,
                        )
                        continue

                    replica_counts = [
                        status.replicas,
                        status.available_replicas,
                        status.ready_replicas,
                        status.updated_replicas,
                    ]
                    if replica_counts.count(desired_replicas) == len(replica_counts):
                        # We are done: all the counts match. Stop the watch and return
                        self.logger.success(f"adjustments to Deployment '{self.name}' rolled out successfully", status)
                        stream.stop()
                        return

            # watch doesn't raise a timeoutError when when elapsed, treat fall through as timeout
            raise WatchTimeoutError()



    def _check_conditions(self, conditions: List[kubernetes_asyncio.client.V1DeploymentCondition]) -> None:
        for condition in conditions:
            if condition.type == "Available":
                if condition.status == "True":
                    # If we hit on this and have not raised yet we are good to go
                    break
                elif condition.status in ("False", "Unknown"):
                    # Condition has not yet been met, log status and continue monitoring
                    self.logger.debug(
                        f"Condition({condition.type}).status == '{condition.status}' ({condition.reason}): {condition.message}"
                    )
                else:
                    raise servo.AdjustmentFailedError(
                        f"encountered unexpected Condition status '{condition.status}'"
                    )

            elif condition.type == "ReplicaFailure":
                # TODO: Check what this error looks like
                raise servo.AdjustmentRejectedError(
                    f"ReplicaFailure: message='{condition.status.message}', reason='{condition.status.reason}'",
                    reason="start-failed"
                )

            elif condition.type == "Progressing":
                if condition.status in ("True", "Unknown"):
                    # Still working
                    self.logger.debug("Deployment update is progressing", condition)
                    break
                elif condition.status == "False":
                    raise servo.AdjustmentRejectedError(
                        f"ProgressionFailure: message='{condition.status.message}', reason='{condition.status.reason}'",
                        reason="start-failed"
                    )
                else:
                    raise servo.AdjustmentFailedError(
                        f"unknown deployment status condition: {condition.status}"
                    )

    async def raise_for_status(self, adjustments: List[servo.Adjustment]) -> None:
        # NOTE: operate off of current state, assuming you have checked is_ready()
        status = self.obj.status
        self.logger.trace(f"current deployment status is {status}")
        if status is None:
            raise RuntimeError(f'No such deployment: {self.name}')

        if not status.conditions:
            raise RuntimeError(f'Deployment is not running: {self.name}')

        # Check for failure conditions
        self._check_conditions(status.conditions)
        await self.raise_for_failed_pod_adjustments(adjustments=adjustments)

        # Catchall
        self.logger.trace(f"unable to map deployment status to exception. Deployment: {self.obj}")
        raise RuntimeError(f"Unknown Deployment status for '{self.name}': {status}")

    async def raise_for_failed_pod_adjustments(self, adjustments: List[servo.Adjustment]):
        pods = await self.get_latest_pods()
        self.logger.trace(f"latest pod(s) status {list(map(lambda p: p.obj.status, pods))}")
        unschedulable_pods = [
            pod for pod in pods
            if pod.obj.status.conditions and any(
                cond.reason == "Unschedulable" for cond in pod.obj.status.conditions
            )
        ]
        if unschedulable_pods:
            pod_messages = []
            for pod in unschedulable_pods:
                cond_msgs = []
                for unschedulable_condition in filter(lambda cond: cond.reason == "Unschedulable", pod.obj.status.conditions):
                    unschedulable_adjustments = list(filter(lambda a: a.setting_name in unschedulable_condition.message, adjustments))
                    cond_msgs.append(
                        f"Requested adjustment(s) ({', '.join(map(str, unschedulable_adjustments))}) cannot be scheduled due to \"{unschedulable_condition.message}\""
                    )
                pod_messages.append(f"{pod.obj.metadata.name} - {'; '.join(cond_msgs)}")

            raise servo.AdjustmentRejectedError(
                f"{len(unschedulable_pods)} pod(s) could not be scheduled for deployment {self.name}: {', '.join(pod_messages)}",
                reason="unschedulable"
            )

        image_pull_failed_pods = [
            pod for pod in pods
            if pod.obj.status.container_statuses and any(
                cont_stat.state and cont_stat.state.waiting and cont_stat.state.waiting.reason in ["ImagePullBackOff", "ErrImagePull"]
                for cont_stat in pod.obj.status.container_statuses
            )
        ]
        if image_pull_failed_pods:
            raise servo.AdjustmentFailedError(
                f"Container image pull failure detected on {len(image_pull_failed_pods)} pods: {', '.join(map(lambda pod: pod.obj.metadata.name, pods))}",
                reason="image-pull-failed"
            )

        restarted_pods_container_statuses = [
            (pod, cont_stat) for pod in pods for cont_stat in (pod.obj.status.container_statuses or [])
            if cont_stat.restart_count > 0
        ]
        if restarted_pods_container_statuses:
            pod_to_counts = collections.defaultdict(list)
            for pod_cont_stat in restarted_pods_container_statuses:
                pod_to_counts[pod_cont_stat[0].obj.metadata.name].append(f"{pod_cont_stat[1].name} x{pod_cont_stat[1].restart_count}")

            pod_message = ", ".join(map(
                lambda kv_tup: f"{kv_tup[0]} - {'; '.join(kv_tup[1])}",
                list(pod_to_counts.items())
            ))
            raise servo.AdjustmentRejectedError(
                f"Deployment {self.name} pod(s) crash restart detected: {pod_message}",
                reason="unstable"
            )

        # Unready pod catchall
        unready_pod_conds = [
            (pod, cond) for pod in pods for cond in (pod.obj.status.conditions or [])
            if cond.type == "Ready" and cond.status == "False"
        ]
        if unready_pod_conds:
            pod_message = ", ".join(map(
                lambda pod_cond: f"{pod_cond[0].obj.metadata.name} - (reason {pod_cond[1].reason}) {pod_cond[1].message}",
                unready_pod_conds
            ))
            raise servo.AdjustmentRejectedError(
                f"Found {len(unready_pod_conds)} unready pod(s) for deployment {self.name}: {pod_message}",
                reason="start-failed"
            )

    async def get_restart_count(self) -> int:
        count = 0
        for pod in await self.get_latest_pods():
            try:
                count += await pod.get_restart_count()
            except kubernetes_asyncio.client.exceptions.ApiException as error:
                if error.status == 404:
                    # Pod no longer exists, move on
                    pass
                else:
                    raise error

        return count

# Workarounds to allow use of api_client.deserialize() public method instead of private api_client._ApiClient__deserialize
# TODO: is this workaround worth it just to avoid using the private method?
# fix for https://github.com/kubernetes-client/python/issues/977#issuecomment-594045477
def default_kubernetes_json_serializer(o: Any) -> Any:
    if isinstance(o, (datetime.datetime, datetime.date)):
        return o.isoformat()
    raise TypeError(f'Object of type {o.__class__.__name__} '
                        f'is not JSON serializable')

# https://github.com/kubernetes-client/python/issues/977#issuecomment-592030030
class FakeKubeResponse:
    """Mocks the RESTResponse object as a workaround for kubernetes python api_client deserialization"""
    def __init__(self, obj):
        self.data = json.dumps(obj, default=default_kubernetes_json_serializer)

# Use alias generator so that dromedary case can be parsed to snake case properties to match k8s python client behaviour
def to_dromedary_case(string: str) -> str:
    split = string.split('_')
    return split[0] + ''.join(word.capitalize() for word in split[1:])

class RolloutBaseModel(pydantic.BaseModel):
    class Config:
        # arbitrary_types_allowed = True
        alias_generator = to_dromedary_case
        allow_population_by_field_name = True

# Pydantic type models for argo rollout spec: https://argoproj.github.io/argo-rollouts/features/specification/
# https://github.com/argoproj/argo-rollouts/blob/master/manifests/crds/rollout-crd.yaml
# NOTE/TODO: fields typed with Any should maintain the same form when dumped as when they are parsed. Should the need
#   arise to interact with such fields, they will need to have an explicit type defined so the alias_generator is applied
class RolloutV1LabelSelector(RolloutBaseModel): # must type out k8s models as well to allow parse_obj to work
    match_expressions: Any
    match_labels: Optional[Dict[str, str]]

class RolloutV1ObjectMeta(RolloutBaseModel):
    annotations: Optional[Dict[str, str]]
    cluster_name: Optional[str]
    creation_timestamp: Optional[datetime.datetime]
    deletion_grace_period_seconds: Optional[int]
    deletion_timestamp: Optional[datetime.datetime]
    finalizers: Optional[List[str]]
    generate_name: Optional[str]
    generation: Optional[int]
    labels: Optional[Dict[str, str]]
    managed_fields: Any
    name: Optional[str]
    namespace: Optional[str]
    owner_references: Any
    resource_version: Optional[str]
    self_link: Optional[str]
    uid: Optional[str]

class RolloutV1EnvVar(RolloutBaseModel):
    name: str
    value: Optional[str]
    value_from: Any

class RolloutV1ContainerPort(RolloutBaseModel):
    container_port: int
    host_ip: Optional[str]
    host_port: Optional[int]
    name: Optional[str]
    protocol: Optional[str]

class RolloutV1ResourceRequirements(RolloutBaseModel):
    limits: Optional[Dict[str, str]]
    requests: Optional[Dict[str, str]]

class RolloutV1Container(RolloutBaseModel):
    args: Optional[List[str]]
    command: Optional[List[str]]
    env: Optional[List[RolloutV1EnvVar]]
    env_from: Any
    image: str
    image_pull_policy: Optional[str]
    lifecycle: Any
    liveness_probe: Any
    name: str
    ports: Optional[List[RolloutV1ContainerPort]]
    readiness_probe: Any
    resources: Optional[RolloutV1ResourceRequirements]
    security_context: Any
    startup_probe: Any
    stdin: Optional[bool]
    stdin_once: Optional[bool]
    termination_message_path: Optional[str]
    termination_message_policy: Optional[str]
    tty: Optional[bool]
    volume_devices: Any
    volume_mounts: Any
    working_dir: Optional[str]

class RolloutV1PodSpec(RolloutBaseModel):
    active_deadline_seconds: Optional[int]
    affinity: Any
    automount_service_account_token: Optional[bool]
    containers: List[RolloutV1Container]
    dns_config: Any
    dns_policy: Optional[str]
    enable_service_links: Optional[bool]
    ephemeral_containers: Any
    host_aliases: Any
    host_ipc: Optional[bool]
    host_network: Optional[bool]
    host_pid: Optional[bool]
    hostname: Optional[str]
    image_pull_secrets: Any
    init_containers: Optional[List[RolloutV1Container]]
    node_name: Optional[str]
    node_selector: Optional[Dict[str, str]]
    overhead: Optional[Dict[str, str]]
    preemption_policy: Optional[str]
    priority: Optional[int]
    priority_class_name: Optional[str]
    readiness_gates: Any
    restart_policy: Optional[str]
    runtime_class_name: Optional[str]
    scheduler_name: Optional[str]
    security_context: Any
    service_account: Optional[str]
    service_account_name: Optional[str]
    share_process_namespace: Optional[bool]
    subdomain: Optional[str]
    termination_grace_period_seconds: Optional[int]
    tolerations: Any
    topology_spread_constraints: Any
    volumes: Any

class RolloutV1PodTemplateSpec(RolloutBaseModel):
    metadata: RolloutV1ObjectMeta
    spec: RolloutV1PodSpec

class RolloutSpec(RolloutBaseModel):
    replicas: int
    selector: RolloutV1LabelSelector
    template: RolloutV1PodTemplateSpec
    min_ready_seconds: Optional[int]
    revision_history_limit: Optional[int]
    paused: Optional[bool]
    progress_deadline_seconds: Optional[int]
    restart_at: Optional[datetime.datetime]
    strategy: Any

class RolloutBlueGreenStatus(RolloutBaseModel):
    active_selector: Optional[str]
    post_promotion_analysis_run: Optional[str]
    post_promotion_analysis_run_status: Any
    pre_promotion_analysis_run: Optional[str]
    pre_promotion_analysis_run_status: Any
    preview_selector: Optional[str]
    previous_active_selector: Optional[str]
    scale_down_delay_start_time: Optional[datetime.datetime]
    scale_up_preview_check_point: Optional[bool]

class RolloutStatusCondition(RolloutBaseModel):
    last_transition_time: datetime.datetime
    last_update_time: datetime.datetime
    message: str
    reason: str
    status: str
    type: str

class RolloutStatus(RolloutBaseModel):
    hpa_replicas: Optional[int] = pydantic.Field(..., alias="HPAReplicas")
    abort: Optional[bool]
    aborted_at: Optional[datetime.datetime]
    available_replicas: Optional[int]
    blue_green: RolloutBlueGreenStatus
    canary: Any #  TODO type this out if connector needs to interact with it
    collision_count: Optional[int]
    conditions: List[RolloutStatusCondition]
    controller_pause: Optional[bool]
    current_pod_hash: str
    current_step_hash: Optional[str]
    current_step_index: Optional[int]
    observed_generation: str
    pause_conditions: Any
    ready_replicas: Optional[int]
    replicas: Optional[int]
    restarted_at: Optional[datetime.datetime]
    selector: str
    stable_RS: Optional[str]
    updated_replicas: Optional[int]

class RolloutObj(RolloutBaseModel): # TODO is this the right base to inherit from?
    api_version: str
    kind: str
    metadata: RolloutV1ObjectMeta
    spec: RolloutSpec
    status: Optional[RolloutStatus]

# TODO expose to config if needed
ROLLOUT_GROUP = "argoproj.io"
ROLLOUT_VERSION = "v1alpha1"
ROLLOUT_PURAL = "rollouts"

class Rollout(KubernetesModel):
    """Wrapper around an ArgoCD Kubernetes `Rollout` Object.
    The actual instance that this
    wraps can be accessed via the ``obj`` instance member.
    This wrapper provides some convenient functionality around the
    API Object and provides some state management for the `Rollout`.
    .. Rollout:
        https://argoproj.github.io/argo-rollouts/features/specification/
    """

    obj: RolloutObj

    _rollout_const_args: Dict[str, str] = dict(
        group=ROLLOUT_GROUP,
        version=ROLLOUT_VERSION,
        plural=ROLLOUT_PURAL,
    )

    api_clients: ClassVar[Dict[str, Type]] = {
        "preferred":kubernetes_asyncio.client.CustomObjectsApi,
        f"{ROLLOUT_GROUP}/{ROLLOUT_VERSION}":kubernetes_asyncio.client.CustomObjectsApi,
    }

    async def create(self, namespace: str = None) -> None:
        """Create the Rollout under the given namespace.
        Args:
            namespace: The namespace to create the Rollout under.
        """
        if namespace is None:
            namespace = self.namespace

        self.logger.info(
            f'creating rollout "{self.name}" in namespace "{namespace}"'
        )
        self.logger.debug(f"rollout: {self.obj}")

        async with self.api_client() as api_client:
            self.obj = RolloutObj.parse_obj(await api_client.create_namespaced_custom_object(
                namespace=namespace,
                body=self.obj.dict(by_alias=True, exclude_none=True),
                **self._rollout_const_args,
            ))

    @classmethod
    async def read(cls, name: str, namespace: str) -> "Rollout":
        """Read a Rollout by name under the given namespace.
        Args:
            name: The name of the Rollout to read.
            namespace: The namespace to read the Rollout from.
        """

        async with cls.preferred_client() as api_client:
            obj = await api_client.get_namespaced_custom_object(
                namespace=namespace,
                name=name,
                **cls._rollout_const_args,
            )
            return Rollout(RolloutObj.parse_obj(obj))

    async def patch(self) -> None:
        """Update the changed attributes of the Rollout."""
        async with self.api_client() as api_client:
            self.obj = RolloutObj.parse_obj(await api_client.patch_namespaced_custom_object(
                namespace=self.namespace,
                name=self.name,
                body=self.obj.dict(by_alias=True, exclude_none=True),
                **self._rollout_const_args,
            ))

    async def delete(self, options:kubernetes_asyncio.client.V1DeleteOptions = None) ->kubernetes_asyncio.client.V1Status:
        """Delete the Rollout.
        This method expects the Rollout to have been loaded or otherwise
        assigned a namespace already. If it has not, the namespace will need
        to be set manually.
        Args:
            options: Unsupported, options for Rollout deletion.
        Returns:
            The status of the delete operation.
        """
        if options is not None:
            raise RuntimeError("Rollout deletion does not support V1DeleteOptions")

        self.logger.info(f'deleting rollout "{self.name}"')
        self.logger.trace(f"rollout: {self.obj}")

        async with self.api_client() as api_client:
            return await api_client.delete_namespaced_custom_object(
                namespace=self.namespace,
                name=self.name,
                **self._rollout_const_args,
            )

    async def refresh(self) -> None:
        """Refresh the underlying Kubernetes Rollout resource."""
        async with self.api_client() as api_client:
            self.obj = RolloutObj.parse_obj(await api_client.get_namespaced_custom_object_status(
                namespace=self.namespace,
                name=self.name,
                **self._rollout_const_args
            ))

    async def rollback(self) -> None:
        # TODO rollbacks are automated in Argo Rollouts, not sure if making this No Op will cause issues
        #   but I was unable to locate a means of triggering a rollout rollback manually
        raise TypeError(
            (
                "rollback is not supported under the optimization of rollouts because rollbacks are applied to "
                "Kubernetes Deployment objects whereas this is automated by argocd"
            )
        )

    async def get_status(self) -> RolloutStatus:
        """Get the status of the Rollout.
        Returns:
            The status of the Rollout.
        """
        self.logger.info(f'checking status of rollout "{self.name}"')
        # first, refresh the rollout state to ensure the latest status
        await self.refresh()

        # return the status from the rollout
        return self.obj.status

    async def get_pods(self) -> List[Pod]:
        """Get the pods for the Rollout.

        Returns:
            A list of pods that belong to the rollout.
        """
        self.logger.debug(f'getting pods for rollout "{self.name}"')

        async with Pod.preferred_client() as api_client:
            label_selector = self.obj.spec.selector.match_labels
            pod_list:kubernetes_asyncio.client.V1PodList = await api_client.list_namespaced_pod(
                namespace=self.namespace, label_selector=selector_string(label_selector)
            )

        pods = [Pod(p) for p in pod_list.items]
        return pods

    @property
    def status(self) -> RolloutStatus:
        """Return the status of the Rollout.
        Returns:
            The status of the Rollout.
        """
        return self.obj.status

    async def is_ready(self) -> bool:
        """Check if the Rollout is in the ready state.

        Returns:
            True if in the ready state; False otherwise.
        """
        await self.refresh()

        # if there is no status, the deployment is definitely not ready
        status = self.obj.status
        if status is None:
            return False

        # check the status for the number of total replicas and compare
        # it to the number of ready replicas. if the numbers are
        # equal, the deployment is ready; otherwise it is not ready.
        total = status.replicas
        ready = status.ready_replicas

        if total is None:
            return False

        return total == ready

    @property
    def containers(self) -> List[Container]:
        """
        Return a list of Container objects from the underlying pod template spec.
        """
        return list(
            map(lambda c: Container(c, None), self.obj.spec.template.spec.containers)
        )

    def find_container(self, name: str) -> Optional[Container]:
        """
        Return the container with the given name.
        """
        return next(filter(lambda c: c.name == name, self.containers), None)

    async def get_target_container(self, config: ContainerConfiguration) -> Optional[Container]:
        """Return the container targeted by the supplied configuration"""
        target_container = self.find_container(config.name)
        if target_container is not None:
            async with kubernetes_asyncio.client.ApiClient() as api_client:
                target_container.obj = api_client.deserialize(
                        response=FakeKubeResponse(target_container.obj.dict(by_alias=True, exclude_none=True)),
                        response_type=kubernetes_asyncio.client.models.V1Container
                    )
        return target_container

    @property
    def replicas(self) -> int:
        """
        Return the number of desired pods.
        """
        return self.obj.spec.replicas

    @replicas.setter
    def replicas(self, replicas: int) -> None:
        """
        Set the number of desired pods.
        """
        self.obj.spec.replicas = replicas

    @property
    def pod_template_spec(self) -> RolloutV1PodTemplateSpec:
        """Return the pod template spec for instances of the Rollout."""
        return self.obj.spec.template

    async def get_pod_template_spec_copy(self) -> kubernetes_asyncio.client.models.V1PodTemplateSpec:
        """Return a deep copy of the pod template spec. Eg. for creation of a tuning pod"""
        async with kubernetes_asyncio.client.ApiClient() as api_client:
            return api_client.deserialize(
                response=FakeKubeResponse(self.pod_template_spec.dict(by_alias=True, exclude_none=True)),
                response_type=kubernetes_asyncio.client.models.V1PodTemplateSpec
            )

    def update_pod(self, pod: kubernetes_asyncio.client.models.V1Pod) -> kubernetes_asyncio.client.models.V1Pod:
        """Update the pod with the latest state of the controller if needed. In the case of argo rollouts, the
        pod labels are updated with the latest template hash so that it will be routed to by the appropriate service"""
        # Apply the latest template hash so the active service register the tuning pod as an endpoint
        pod.metadata.labels["rollouts-pod-template-hash"] = self.obj.status.current_pod_hash
        return pod

    @backoff.on_exception(backoff.expo, kubernetes_asyncio.client.exceptions.ApiException, max_tries=3)
    async def inject_sidecar(
        self,
        name: str,
        image: str,
        *,
        service: Optional[str] = None,
        port: Optional[int] = None,
        index: Optional[int] = None,
        service_port: int = 9980
        ) -> None:
        """
        Injects an Envoy sidecar into a target Deployment that proxies a service
        or literal TCP port, generating scrapeable metrics usable for optimization.

        The service or port argument must be provided to define how traffic is proxied
        between the Envoy sidecar and the container responsible for fulfilling the request.

        Args:
            name: The name of the sidecar to inject.
            image: The container image for the sidecar container.
            service: Name of the service to proxy. Envoy will accept ingress traffic
                on the service port and reverse proxy requests back to the original
                target container.
            port: The name or number of a port within the Deployment to wrap the proxy around.
            index: The index at which to insert the sidecar container. When `None`, the sidecar is appended.
            service_port: The port to receive ingress traffic from an upstream service.
        """

        await self.refresh()

        if not (service or port):
            raise ValueError(f"a service or port must be given")

        if isinstance(port, str) and port.isdigit():
            port = int(port)

        # check for a port conflict
        container_ports = list(itertools.chain(*map(operator.attrgetter("ports"), self.containers)))
        if service_port in list(map(operator.attrgetter("container_port"), container_ports)):
            raise ValueError(f"Port conflict: Rollout '{self.name}' already exposes port {service_port} through an existing container")

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
                    raise ValueError(f"Target Service '{service}' does not expose any ports")
                elif port_count > 1:
                    raise ValueError(f"Target Service '{service}' exposes multiple ports -- target port must be specified")
                port_obj = service_obj.obj.spec.ports[0]
            else:
                if isinstance(port, int):
                    port_obj = next(filter(lambda p: p.port == port, service_obj.obj.spec.ports), None)
                elif isinstance(port, str):
                    port_obj = next(filter(lambda p: p.name == port, service_obj.obj.spec.ports), None)
                else:
                    raise TypeError(f"Unable to resolve port value of type {port.__class__} (port={port})")

                if not port_obj:
                    raise ValueError(f"Port '{port}' does not exist in the Service '{service}'")

            # resolve symbolic name in the service target port to a concrete container port
            if isinstance(port_obj.target_port, str):
                container_port_obj = next(filter(lambda p: p.name == port_obj.target_port, container_ports), None)
                if not container_port_obj:
                    raise ValueError(f"Port '{port_obj.target_port}' could not be resolved to a destination container port")

                container_port = container_port_obj.container_port
            else:
                container_port = port_obj.target_port

        else:
            # find the container port
            container_port_obj = next(filter(lambda p: p.container_port == port, container_ports), None)
            if not container_port_obj:
                raise ValueError(f"Port '{port}' could not be resolved to a destination container port")

            container_port = container_port_obj.container_port

        # build the sidecar container
        container = RolloutV1Container(
            name=name,
            image=image,
            image_pull_policy="IfNotPresent",
            resources=RolloutV1ResourceRequirements(
                requests={
                    "cpu": "125m",
                    "memory": "128Mi"
                },
                limits={
                    "cpu": "250m",
                    "memory": "256Mi"
                }
            ),
            env=[
                RolloutV1EnvVar(name="OPSANI_ENVOY_PROXY_SERVICE_PORT", value=str(service_port)),
                RolloutV1EnvVar(name="OPSANI_ENVOY_PROXIED_CONTAINER_PORT", value=str(container_port)),
                RolloutV1EnvVar(name="OPSANI_ENVOY_PROXY_METRICS_PORT", value="9901")
            ],
            ports=[
                RolloutV1ContainerPort(name="opsani-proxy", container_port=service_port, protocol="TCP"),
                RolloutV1ContainerPort(name="opsani-metrics", container_port=9901, protocol="TCP"),
            ]
        )

        # add the sidecar to the Deployment
        if index is None:
            self.obj.spec.template.spec.containers.append(container)
        else:
            self.obj.spec.template.spec.containers.insert(index, container)

        # patch the deployment
        await self.patch()

    # TODO: convert to rollout logic
    async def eject_sidecar(self, name: str) -> bool:
        """Eject an Envoy sidecar from the Deployment.

        Returns True if the sidecar was ejected.
        """
        await self.refresh()
        container = self.remove_container(name)
        if container:
            await self.replace()
            return True

        return False

    # TODO: rebase this and _check_conditions for saturation mode
    @contextlib.asynccontextmanager
    async def rollout(self, *, timeout: Optional[servo.Duration] = None) -> None:
        raise NotImplementedError('To be implemented in future update')

class Millicore(int):
    """
    The Millicore class represents one one-thousandth of a vCPU or hyperthread in Kubernetes.
    """

    @classmethod
    def __get_validators__(cls) -> pydantic.CallableGenerator:
        yield cls.parse

    @classmethod
    def parse(cls, v: pydantic.StrIntFloat) -> "Millicore":
        """
        Parses a string, integer, or float input value into Millicore units.

        Returns:
            The input value in Millicore units.

        Raises:
            ValueError: Raised if the input cannot be parsed.
        """
        if isinstance(v, str):
            if v[-1] == "m":
                return cls(int(v[:-1]))
            else:
                return cls(int(float(v) * 1000))
        elif isinstance(v, (int, float, decimal.Decimal)):
            return cls(int(float(v) * 1000))
        else:
            raise ValueError("could not parse millicore value")

    def __str__(self) -> str:
        if self % 1000 == 0:
            return str(int(self) // 1000)
        else:
            return f"{int(self)}m"

    def __float__(self) -> float:
        return self / 1000.0

    def __eq__(self, other) -> bool:
        if isinstance(other, str):
            return str(self) == other
        elif isinstance(other, float):
            return float(self) == other
        return super().__eq__(other)

    def human_readable(self) -> str:
        return str(self)


class CPU(servo.CPU):
    """
    The CPU class models a Kubernetes CPU resource in Millicore units.
    """

    min: Millicore
    max: Millicore
    step: Millicore
    value: Optional[Millicore]

    # Kubernetes resource requirements
    request: Optional[Millicore]
    limit: Optional[Millicore]
    get: pydantic.conlist(ResourceRequirement, min_items=1) = [ResourceRequirement.request, ResourceRequirement.limit]
    set: pydantic.conlist(ResourceRequirement, min_items=1) = [ResourceRequirement.request, ResourceRequirement.limit]

    def __opsani_repr__(self) -> dict:
        o_dict = super().__opsani_repr__()

        # normalize values into floats (see Millicore __float__)
        for field in ("min", "max", "step", "value"):
            value = getattr(self, field)
            o_dict["cpu"][field] = float(value) if value is not None else None
        return o_dict


# Gibibyte is the base unit of Kubernetes memory
MiB = 2 ** 20
GiB = 2 ** 30


class ShortByteSize(pydantic.ByteSize):
    """Kubernetes omits the 'B' suffix for some reason"""

    @classmethod
    def validate(cls, v: pydantic.StrIntFloat) -> "ShortByteSize":
        if isinstance(v, str):
            try:
                return super().validate(v)
            except:
                # Append the byte suffix and retry parsing
                return super().validate(v + "b")
        elif isinstance(v, float):
            v = v * GiB
        return super().validate(v)

    def human_readable(self) -> str:
        sup = super().human_readable()
        # Remove the 'B' suffix to align with Kubernetes units (`GiB` -> `Gi`)
        if sup[-1] == 'B' and sup[-2].isalpha():
            sup = sup[0:-1]
        return sup


class Memory(servo.Memory):
    """
    The Memory class models a Kubernetes Memory resource.
    """

    min: ShortByteSize
    max: ShortByteSize
    step: ShortByteSize
    value: Optional[ShortByteSize]

    # Kubernetes resource requirements
    request: Optional[ShortByteSize]
    limit: Optional[ShortByteSize]
    get: pydantic.conlist(ResourceRequirement, min_items=1) = [ResourceRequirement.request, ResourceRequirement.limit]
    set: pydantic.conlist(ResourceRequirement, min_items=1) = [ResourceRequirement.request, ResourceRequirement.limit]

    def __opsani_repr__(self) -> dict:
        o_dict = super().__opsani_repr__()

        # normalize values into floating point Gibibyte units
        for field in ("min", "max", "step", "value"):
            value = getattr(self, field)
            o_dict["mem"][field] = float(value) / GiB if value is not None else None
        return o_dict


def _normalize_adjustment(adjustment: servo.Adjustment) -> Tuple[str, Union[str, servo.Numeric]]:
    """Normalize an adjustment object into a Kubernetes native setting key/value pair."""
    setting = "memory" if adjustment.setting_name == "mem" else adjustment.setting_name
    value = adjustment.value

    if setting == "memory":
        # Add GiB suffix to Numerics and Numeric strings
        if (isinstance(value, (int, float)) or
            (isinstance(value, str) and value.replace('.', '', 1).isdigit())):
            value = f"{value}Gi"
    elif setting == "cpu":
        value = str(Millicore.parse(value))
    elif setting == "replicas":
        value = int(float(value))

    return setting, value

class BaseOptimization(abc.ABC, pydantic.BaseModel, servo.logging.Mixin):
    """
    BaseOptimization is the base class for concrete implementations of optimization strategies.

    Attributes:
        name (str): The name of the Optimization. Used to set the name for the corresponding component.
        timeout (Duration): Time interval to wait before considering Kubernetes operations to have failed.
        adjustments (List[Adjustment]): List of adjustments applied to this optimization (NOTE optimizations are re-created for each
            event dispatched to the connector. Thus, this value will only be populated during adjust event handling with only the adjustments
            pertaining to that adjust event dispatch)
    """

    name: str
    timeout: servo.Duration
    adjustments: List[servo.Adjustment] = []

    @abc.abstractclassmethod
    async def create(
        cls, config: "BaseKubernetesConfiguration", *args, **kwargs
    ) -> "BaseOptimization":
        """"""
        ...

    @abc.abstractmethod
    def adjust(
        self, adjustment: servo.Adjustment, control: servo.Control = servo.Control()
    ) -> servo.Description:
        """
        Adjust a setting on the underlying Deployment/Pod or Container.
        """
        ...

    @abc.abstractmethod
    async def apply(self) -> None:
        """
        Apply the adjusted settings to the Kubernetes cluster.
        """
        ...

    @abc.abstractmethod
    async def raise_for_status(self) -> None:
        """Raise an exception if in an unhealthy state."""
        ...

    @property
    @abc.abstractmethod
    def on_failure(self) -> FailureMode:
        """
        Return the configured failure behavior.
        """
        ...

    async def handle_error(self, error: Exception) -> bool:
        """
        Handle an operational failure in accordance with the failure mode configured by the operator.

        Well executed error handling requires context and strategic thinking. The servo base library
        provides a rich set of primitives and patterns for approaching error handling but ultimately
        the experience is reliant on the connector developer who has knowledge of the essential context
        and understands the user needs and expectations.

        The error handling implementation provided in this method handles the general cases out of the
        box and relies on abstract methods (see below) to implement more advanced behaviors such as
        rollback and tear-down.

        Returns:
            A boolean value that indicates if the error was handled.

        Raises:
            NotImplementedError: Raised if there is no handler for a given failure mode. Subclasses
                must filter failure modes before calling the superclass implementation.
        """
        # Ensure that we chain any underlying exceptions that may occur
        try:
            self.logger.error(f"handling error with with failure mode {self.on_failure}: {error.__class__.__name__} - {str(error)}")
            self.logger.opt(exception=error).debug(f"kubernetes error details")

            if self.on_failure == FailureMode.exception:
                raise error

            elif self.on_failure == FailureMode.ignore:
                self.logger.opt(exception=error).warning(f"ignoring exception")
                return True

            elif self.on_failure == FailureMode.rollback:
                await self.rollback(error)

            elif self.on_failure == FailureMode.destroy:
                await self.destroy(error)

            else:
                # Trap any new modes that need to be handled
                raise NotImplementedError(
                    f"missing error handler for failure mode '{self.on_failure}'"
                )

            raise error # Always communicate errors to backend unless ignored

        except Exception as handler_error:
            raise handler_error from error  # reraising an error from itself is safe


    @abc.abstractmethod
    async def rollback(self, error: Optional[Exception] = None) -> None:
        """
        Asynchronously roll back the Optimization to a previous known
        good state.

        Args:
            error: An optional exception that contextualizes the cause of the rollback.
        """
        ...

    @abc.abstractmethod
    async def destroy(self, error: Optional[Exception] = None) -> None:
        """
        Asynchronously destroy the Optimization.

        Args:
            error: An optional exception that contextualizes the cause of the destruction.
        """
        ...

    @abc.abstractmethod
    def to_components(self) -> List[servo.Component]:
        """
        Return a list of Component representations of the Optimization.

        Components are the canonical representation of optimizations in the Opsani API.
        """
        ...

    @abc.abstractmethod
    async def is_ready(self) -> bool:
        """
        Verify Optimization target Resource/Controller is ready.
        """
        ...

    def __hash__(self):
        return hash(
            (
                self.name,
                id(self),
            )
        )

    class Config:
        arbitrary_types_allowed = True


class DeploymentOptimization(BaseOptimization):
    """
    The DeploymentOptimization class implements an optimization strategy based on directly reconfiguring a Kubernetes
    Deployment and its associated containers.
    """

    deployment_config: "DeploymentConfiguration"
    deployment: Deployment
    container_config: "ContainerConfiguration"
    container: Container

    @classmethod
    async def create(
        cls, config: "DeploymentConfiguration", **kwargs
    ) -> "DeploymentOptimization":
        deployment = await Deployment.read(config.name, config.namespace)

        replicas = config.replicas.copy()
        replicas.value = deployment.replicas

        # FIXME: Currently only supporting one container
        for container_config in config.containers:
            container = deployment.find_container(container_config.name)
            if not container:
                names = servo.utilities.strings.join_to_series(
                    list(map(lambda c: c.name, deployment.containers))
                )
                raise ValueError(
                    f'no container named "{container_config.name}" exists in the Pod (found {names})'
                )

            name = container_config.alias or (
                f"{deployment.name}/{container.name}" if container else deployment.name
            )
            return cls(
                name=name,
                deployment_config=config,
                deployment=deployment,
                container_config=container_config,
                container=container,
                **kwargs,
            )

    @property
    def cpu(self) -> CPU:
        """
        Return the current CPU setting for the optimization.
        """
        cpu = self.container_config.cpu.copy()

        # Determine the value in priority order from the config
        resource_requirements = self.container.get_resource_requirements('cpu')
        value = resource_requirements.get(
            next(filter(lambda r: resource_requirements[r] is not None, self.container_config.cpu.get), None)
        )
        cpu.value = value
        cpu.request = resource_requirements.get(ResourceRequirement.request)
        cpu.limit = resource_requirements.get(ResourceRequirement.limit)
        return cpu

    @property
    def memory(self) -> Memory:
        """
        Return the current Memory setting for the optimization.
        """
        memory = self.container_config.memory.copy()

        # Determine the value in priority order from the config
        resource_requirements = self.container.get_resource_requirements('memory')
        value = resource_requirements.get(
            next(filter(lambda r: resource_requirements[r] is not None, self.container_config.memory.get), None)
        )
        memory.value = value
        memory.request = resource_requirements.get(ResourceRequirement.request)
        memory.limit = resource_requirements.get(ResourceRequirement.limit)
        return memory

    @property
    def replicas(self) -> servo.Replicas:
        """
        Return the current Replicas setting for the optimization.
        """
        replicas = self.deployment_config.replicas.copy()
        replicas.value = self.deployment.replicas
        return replicas

    @property
    def on_failure(self) -> FailureMode:
        """
        Return the configured failure behavior. If not set explicitly, this will be cascaded
        from the base kubernetes configuration (or its default)
        """
        return self.deployment_config.on_failure

    async def rollback(self, error: Optional[Exception] = None) -> None:
        """
        Initiates an asynchronous rollback to a previous version of the Deployment.

        Args:
            error: An optional error that triggered the rollback.
        """
        self.logger.info(f"adjustment failed: rolling back deployment... ({error})")
        await asyncio.wait_for(
            self.deployment.rollback(),
            timeout=self.timeout.total_seconds(),
        )

    async def destroy(self, error: Optional[Exception] = None) -> None:
        """
        Initiates the asynchronous deletion of the Deployment under optimization.

        Args:
            error: An optional error that triggered the destruction.
        """
        self.logger.info(f"adjustment failed: destroying deployment...")
        await asyncio.wait_for(
            self.deployment.delete(),
            timeout=self.timeout.total_seconds(),
        )

    def to_components(self) -> List[servo.Component]:
        return [
            servo.Component(name=self.name, settings=[self.cpu, self.memory, self.replicas])
        ]

    def adjust(self, adjustment: servo.Adjustment, control: servo.Control = servo.Control()) -> None:
        """
        Adjust the settings on the Deployment or a component Container.

        Adjustments do not take effect on the cluster until the `apply` method is invoked
        to enable aggregation of related adjustments and asynchronous application.
        """
        self.adjustments.append(adjustment)
        setting_name, value = _normalize_adjustment(adjustment)
        self.logger.info(f"adjusting {setting_name} to {value}")

        if setting_name in ("cpu", "memory"):
            # NOTE: Assign to the config to trigger validations
            setting = getattr(self.container_config, setting_name)
            setting.value = value

            # Set only the requirements defined in the config
            requirements: Dict[ResourceRequirement, Optional[str]] = {}
            for requirement in setting.set:
                requirements[requirement] = value

            self.container.set_resource_requirements(setting_name, requirements)

        elif setting_name == "replicas":
            # NOTE: Assign to the config to trigger validations
            self.deployment_config.replicas.value = value
            self.deployment.replicas = value

        else:
            raise RuntimeError(
                f"failed adjustment of unsupported Kubernetes setting '{adjustment.setting_name}'"
            )

    async def apply(self) -> None:
        """
        Apply changes asynchronously and wait for them to roll out to the cluster.

        Kubernetes deployments orchestrate a number of underlying resources. Awaiting the
        outcome of a deployment change requires observation of the `resource_version` which
        indicates if a given patch actually changed the resource, the `observed_generation`
        which is a value managed by the deployments controller and indicates the effective
        version of the deployment exclusive of insignificant changes that do not affect runtime
        (such as label updates), and the `conditions` of the deployment status which reflect
        state at a particular point in time. How these elements change during a rollout is
        dependent on the deployment strategy in effect and its requirements (max unavailable,
        surge, etc).

        The logic implemented by this method is as follows:
            - Capture the `resource_version` and `observed_generation`.
            - Patch the underlying Deployment object via the Kubernetes API.
            - Check that `resource_version` has been incremented or return early if nothing has changed.
            - Create a Kubernetes Watch on the Deployment targeted by label selector and resource version.
            - Observe events streamed via the watch.
            - Look for the Deployment to report a Status Condition of `"Progressing"`.
            - Wait for the `observed_generation` to increment indicating that the Deployment is applying our changes.
            - Track the value of the `available_replicas`, `ready_replicas`, `unavailable_replicas`,
                and `updated_replicas` attributes of the Deployment Status until `available_replicas`,
                `ready_replicas`, and `updated_replicas` are all equal to the value of the `replicas` attribute of
                the Deployment and `unavailable_replicas` is `None`. Return success.
            - Raise an error upon expiration of an adjustment timeout or encountering a Deployment Status Condition
                where `type=Progressing` and `status=False`.

        This method abstracts the details of adjusting a Deployment and returns once the desired
        changes have been fully rolled out to the cluster or an error has been encountered.

        See https://kubernetes.io/docs/concepts/workloads/controllers/deployment/

        # The resource_version attribute lets us efficiently watch for changes
        # reference: https://kubernetes.io/docs/reference/using-api/api-concepts/#efficient-detection-of-changes
        """
        try:
            async with self.deployment.rollout(timeout=self.timeout) as deployment:
                # Patch the Deployment via the Kubernetes API
                await deployment.patch()
        except WatchTimeoutError:
            servo.logger.error(f"Timed out waiting for Deployment to become ready...")
            await self.raise_for_status()

    async def is_ready(self) -> bool:
        is_ready, restart_count = await asyncio.gather(
            self.deployment.is_ready(),
            self.deployment.get_restart_count()
        )
        return is_ready and restart_count == 0

    async def raise_for_status(self) -> None:
        """Raise an exception if in an unhealthy state."""
        await self.deployment.raise_for_status(adjustments=self.adjustments)


# TODO: Break down into CanaryDeploymentOptimization and CanaryContainerOptimization
class CanaryOptimization(BaseOptimization):
    """CanaryOptimization objects manage the optimization of Containers within a Deployment using
    a tuning Pod that is adjusted independently and compared against the performance and cost profile
    of its siblings.
    """

    # The deployment and container stanzas from the configuration
    deployment_config: Optional["DeploymentConfiguration"]
    rollout_config: Optional["RolloutConfiguration"]
    container_config: "ContainerConfiguration"

    # State for mainline resources. Read from the cluster
    deployment: Optional[Deployment]
    rollout: Optional[Rollout]
    main_container: Container

    # State for tuning resources
    tuning_pod: Optional[Pod]
    tuning_container: Optional[Container]

    _tuning_pod_template_spec: Optional[kubernetes_asyncio.client.models.V1PodTemplateSpec] = pydantic.PrivateAttr()


    @pydantic.root_validator
    def check_deployment_and_rollout(cls, values):
        if values.get('deployment_config') is not None and values.get('rollout_config') is not None:
            raise ValueError("Cannot create a CanaryOptimization with both rollout and deployment configurations")
        if values.get('deployment') is not None and values.get('rollout') is not None:
            raise ValueError("Cannot create a CanaryOptimization with both rollout and deployment")

        if values.get('deployment_config') is None and values.get('rollout_config') is None:
            raise ValueError("CanaryOptimization must be initialized with either a rollout or deployment configuration")
        if values.get('deployment') is None and values.get('rollout') is None:
            raise ValueError("CanaryOptimization must be initialized with either a rollout or deployment")

        return values

    @property
    def target_controller_config(self) -> Union["DeploymentConfiguration", "RolloutConfiguration"]:
        return self.deployment_config or self.rollout_config

    @property
    def target_controller(self) -> Union[Deployment, Rollout]:
        return self.deployment or self.rollout

    @property
    def target_controller_type(self) -> str:
        return type(self.target_controller).__name__

    @classmethod
    async def create(
        cls, deployment_or_rollout_config: Union["DeploymentConfiguration", "RolloutConfiguration"], **kwargs
    ) -> "CanaryOptimization":
        read_args = (deployment_or_rollout_config.name, cast(str, deployment_or_rollout_config.namespace))
        if isinstance(deployment_or_rollout_config, DeploymentConfiguration):
            controller_type = "Deployment"
            deployment_or_rollout = await Deployment.read(*read_args)
            init_args = dict(deployment_config = deployment_or_rollout_config, deployment = deployment_or_rollout)
        elif isinstance(deployment_or_rollout_config, RolloutConfiguration):
            controller_type = "Rollout"
            deployment_or_rollout = await Rollout.read(*read_args)
            init_args = dict(rollout_config = deployment_or_rollout_config, rollout = deployment_or_rollout)
        else:
            raise NotImplementedError(f"Unknown configuration type '{type(deployment_or_rollout_config).__name__}'")
        if not deployment_or_rollout:
            raise ValueError(
                f'cannot create CanaryOptimization: target {controller_type} "{deployment_or_rollout_config.name}"'
                f' does not exist in Namespace "{deployment_or_rollout_config.namespace}"'
            )

        # NOTE: Currently only supporting one container
        assert len(deployment_or_rollout_config.containers) == 1, "CanaryOptimization currently only supports a single container"
        container_config = deployment_or_rollout_config.containers[0]
        main_container = await deployment_or_rollout.get_target_container(container_config)
        name = (
            deployment_or_rollout_config.strategy.alias
            if isinstance(deployment_or_rollout_config.strategy, CanaryOptimizationStrategyConfiguration)
            and deployment_or_rollout_config.strategy.alias
            else f"{deployment_or_rollout.name}/{main_container.name}-tuning"
        )

        optimization = cls(
            name=name,
            **init_args,
            container_config=container_config,
            main_container=main_container,
            **kwargs,
        )
        await optimization._load_tuning_state()

        return optimization

    async def _load_tuning_state(self) -> None:
        # Find an existing tuning Pod/Container if available
        try:
            tuning_pod = await Pod.read(self.tuning_pod_name, cast(str, self.namespace))
            tuning_container = tuning_pod.get_container(self.container_config.name)

        except kubernetes_asyncio.client.exceptions.ApiException as e:
            if e.status != 404 or e.reason != "Not Found":
                servo.logger.trace(f"Failed reading tuning pod: {e}")
                raise
            else:
                tuning_pod = None
                tuning_container = None

        # TODO: Factor into a new class?
        self.tuning_pod = tuning_pod
        self.tuning_container = tuning_container
        await self._configure_tuning_pod_template_spec()

    @property
    def pod_template_spec_container(self) -> Container:
        container_obj = next(filter(lambda c: c.name == self.container_config.name, self._tuning_pod_template_spec.spec.containers))
        return Container(container_obj, None)

    def adjust(self, adjustment: servo.Adjustment, control: servo.Control = servo.Control()) -> None:
        assert self.tuning_pod, "Tuning Pod not loaded"
        assert self.tuning_container, "Tuning Container not loaded"

        self.adjustments.append(adjustment)
        setting_name, value = _normalize_adjustment(adjustment)
        self.logger.info(f"adjusting {setting_name} to {value}")

        if setting_name in ("cpu", "memory"):
            # NOTE: Assign to the config model to trigger validations
            setting = getattr(self.container_config, setting_name).copy()
            servo.logger.debug(f"Adjusting {setting_name}={value}")
            setting.value = value

            # Set only the requirements defined in the config
            requirements: Dict[ResourceRequirement, Optional[str]] = {}
            for requirement in setting.set:
                requirements[requirement] = value
                servo.logger.debug(f"Assigning {setting_name}.{requirement}={value}")

            servo.logger.debug(f"Setting resource requirements for {setting_name} to {requirements} on PodTemplateSpec")
            self.pod_template_spec_container.set_resource_requirements(setting_name, requirements)

        elif setting_name == "replicas":
            if value != 1:
                servo.logger.warning(
                    f'ignored attempt to set replicas to "{value}"'
                )

        else:
            raise servo.AdjustmentFailedError(
                f"failed adjustment of unsupported Kubernetes setting '{setting_name}'"
            )

    async def apply(self) -> None:
        """Apply the adjustments to the target."""
        assert self.tuning_pod, "Tuning Pod not loaded"
        assert self.tuning_container, "Tuning Container not loaded"

        servo.logger.info("Applying adjustments to Tuning Pod")
        task = asyncio.create_task(self.create_or_recreate_tuning_pod())
        try:
            await task
        except asyncio.CancelledError:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

            raise

        # TODO: logging the wrong values -- should be coming from the podtemplatespec?
        servo.logger.success(f"Built new tuning pod with container resources: {self.tuning_container.resources}")

    @property
    def namespace(self) -> str:
        return self.target_controller_config.namespace

    @property
    def tuning_pod_name(self) -> str:
        """
        Return the name of tuning Pod for this optimization.
        """
        return f"{self.target_controller_config.name}-tuning"

    async def delete_tuning_pod(self, *, raise_if_not_found: bool = True) -> Optional[Pod]:
        """
        Delete the tuning Pod.
        """
        try:
            # TODO: Provide context manager or standard read option that handle not found? Lots of duplication on not found/conflict handling...
            tuning_pod = await Pod.read(self.tuning_pod_name, self.namespace)
            self.logger.info(
                f"Deleting tuning Pod '{tuning_pod.name}' from namespace '{tuning_pod.namespace}'..."
            )
            await tuning_pod.delete()
            await tuning_pod.wait_until_deleted()
            self.logger.info(
                f"Deleted tuning Pod '{tuning_pod.name}' from namespace '{tuning_pod.namespace}'."
            )

            self.tuning_pod = None
            self.tuning_container = None
            return tuning_pod

        except kubernetes_asyncio.client.exceptions.ApiException as e:
            if e.status != 404 or e.reason != "Not Found" and raise_if_not_found:
                raise

            self.tuning_pod = None
            self.tuning_container = None

        return None

    @property
    def target_controller_name(self) -> str:
        return self.target_controller_config.name

    @property
    def container_name(self) -> str:
        return self.container_config.name

    # TODO: Factor into another class?
    async def _configure_tuning_pod_template_spec(self) -> None:
        # Configure a PodSpecTemplate for the tuning Pod state
        pod_template_spec: kubernetes_asyncio.client.models.V1PodTemplateSpec = await self.target_controller.get_pod_template_spec_copy()
        pod_template_spec.metadata.name = self.tuning_pod_name

        if pod_template_spec.metadata.annotations is None:
            pod_template_spec.metadata.annotations = {}
        pod_template_spec.metadata.annotations["opsani.com/opsani_tuning_for"] = self.name
        if pod_template_spec.metadata.labels is None:
            pod_template_spec.metadata.labels = {}
        pod_template_spec.metadata.labels["opsani_role"] = "tuning"

        # Build a container from the raw podspec
        container_obj = next(filter(lambda c: c.name == self.container_config.name, pod_template_spec.spec.containers))
        container = Container(container_obj, None)
        servo.logger.debug(f"Initialized new tuning container from Pod spec template: {container.name}")

        if self.tuning_container:
            servo.logger.debug(f"Copying resource requirements from existing tuning pod container '{self.tuning_pod.name}/{self.tuning_container.name}'")
            resource_requirements = self.tuning_container.resources
            container.resources = resource_requirements
        else:
            servo.logger.debug(f"No existing tuning pod container found, initializing resource requirement defaults")
            set_container_resource_defaults_from_config(container, self.container_config)

        # If the servo is running inside Kubernetes, register self as the controller for the Pod and ReplicaSet
        servo_pod_name = os.environ.get("POD_NAME")
        servo_pod_namespace = os.environ.get("POD_NAMESPACE")
        if servo_pod_name is not None and servo_pod_namespace is not None:
            self.logger.debug(
                f"running within Kubernetes, registering as Pod controller... (pod={servo_pod_name}, namespace={servo_pod_namespace})"
            )
            servo_pod = await Pod.read(servo_pod_name, servo_pod_namespace)
            pod_controller = next(
                iter(
                    ow
                    for ow in servo_pod.obj.metadata.owner_references
                    if ow.controller
                )
            )

            # TODO: Create a ReplicaSet class...
            async with kubernetes_asyncio.client.api_client.ApiClient() as api:
                api_client = kubernetes_asyncio.client.AppsV1Api(api)

                servo_rs: kubernetes_asyncio.client.V1ReplicaSet = (
                    await api_client.read_namespaced_replica_set(
                        name=pod_controller.name, namespace=servo_pod_namespace
                    )
                )  # still ephemeral
                rs_controller = next(
                    iter(
                        ow for ow in servo_rs.metadata.owner_references if ow.controller
                    )
                )
                servo_dep: kubernetes_asyncio.client.V1Deployment = (
                    await api_client.read_namespaced_deployment(
                        name=rs_controller.name, namespace=servo_pod_namespace
                    )
                )

            pod_template_spec.metadata.owner_references = [
               kubernetes_asyncio.client.V1OwnerReference(
                    api_version=servo_dep.api_version,
                    block_owner_deletion=True,
                    controller=True,  # Ensures the pod will not be adopted by another controller
                    kind="Deployment",
                    name=servo_dep.metadata.name,
                    uid=servo_dep.metadata.uid,
                )
            ]

        self._tuning_pod_template_spec = pod_template_spec

    async def create_or_recreate_tuning_pod(self) -> Pod:
        """
        Creates a new Tuning Pod or deletes and recreates one from the current optimization state.
        """
        servo.logger.info("Deleting existing tuning pod (if any)")
        await self.delete_tuning_pod(raise_if_not_found=False)
        return await self.create_tuning_pod()

    async def create_tuning_pod(self) -> Pod:
        """
        Creates a new Tuning Pod from the current optimization state.
        """
        assert self._tuning_pod_template_spec, "Must have tuning pod template spec"
        assert self.tuning_pod is None, "Tuning Pod already exists"
        assert self.tuning_container is None, "Tuning Pod Container already exists"
        self.logger.debug(
            f"creating tuning pod '{self.tuning_pod_name}' based on {self.target_controller_type} '{self.target_controller_name}' in namespace '{self.namespace}'"
        )

        # Setup the tuning Pod -- our settings are updated on the underlying PodSpec template
        self.logger.trace(f"building new tuning pod")
        pod_obj = kubernetes_asyncio.client.V1Pod(
            metadata=self._tuning_pod_template_spec.metadata, spec=self._tuning_pod_template_spec.spec
        )

        # Update pod with latest controller state
        pod_obj = self.target_controller.update_pod(pod_obj)

        tuning_pod = Pod(obj=pod_obj)

        # Create the Pod and wait for it to get ready
        self.logger.info(
            f"Creating tuning Pod '{self.tuning_pod_name}' in namespace '{self.namespace}'"
        )
        await tuning_pod.create(self.namespace)
        servo.logger.success(f"Created Tuning Pod '{self.tuning_pod_name}' in namespace '{self.namespace}'")

        servo.logger.info(f"waiting up to {self.timeout} for Tuning Pod to become ready...")
        progress = servo.EventProgress(self.timeout)
        progress_logger = lambda p: self.logger.info(
            p.annotate(f"waiting for '{self.tuning_pod_name}' to become ready...", prefix=False)
        )
        progress.start()

        task = asyncio.create_task(tuning_pod.wait_until_ready())
        task.add_done_callback(lambda _: progress.complete())
        gather_task = asyncio.gather(
            task,
            progress.watch(progress_logger),
        )

        try:
            await asyncio.wait_for(
                gather_task,
                timeout=self.timeout.total_seconds()
            )

        except asyncio.TimeoutError:
            servo.logger.error(f"Timed out waiting for Tuning Pod to become ready...")
            servo.logger.debug(f"Cancelling Task: {task}, progress: {progress}")
            for t in {task, gather_task}:
                t.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await t
                    servo.logger.debug(f"Cancelled Task: {t}, progress: {progress}")

            await tuning_pod.raise_for_status(adjustments=self.adjustments)

        # Load the in memory model for various convenience accessors
        await tuning_pod.refresh()
        await tuning_pod.get_containers()

        # Hydrate local state
        self.tuning_pod = tuning_pod
        self.tuning_container = tuning_pod.get_container(self.container_config.name)

        servo.logger.info(f"Tuning Pod successfully created")
        return tuning_pod

    @property
    def tuning_cpu(self) -> Optional[CPU]:
        """
        Return the current CPU setting for the target container of the tuning Pod (if any).
        """
        if not self.tuning_pod:
            return None

        cpu = self.container_config.cpu.copy()

        # Determine the value in priority order from the config
        resource_requirements = self.tuning_container.get_resource_requirements('cpu')
        value = resource_requirements.get(
            next(filter(lambda r: resource_requirements[r] is not None, self.container_config.cpu.get), None)
        )

        cpu.value = value
        cpu.request = resource_requirements.get(ResourceRequirement.request)
        cpu.limit = resource_requirements.get(ResourceRequirement.limit)
        return cpu

    @property
    def tuning_memory(self) -> Optional[Memory]:
        """
        Return the current Memory setting for the target container of the tuning Pod (if any).
        """
        if not self.tuning_pod:
            return None

        memory = self.container_config.memory.copy()

        # Determine the value in priority order from the config
        resource_requirements = self.tuning_container.get_resource_requirements('memory')
        value = resource_requirements.get(
            next(filter(lambda r: resource_requirements[r] is not None, self.container_config.memory.get), None)
        )
        memory.value = value
        memory.request = resource_requirements.get(ResourceRequirement.request)
        memory.limit = resource_requirements.get(ResourceRequirement.limit)
        return memory

    @property
    def tuning_replicas(self) -> servo.Replicas:
        """
        Return the current Replicas setting for the optimization.
        """
        value = 1 if self.tuning_pod else 0
        return servo.Replicas(
            min=0,
            max=1,
            value=value,
            pinned=True,
        )

    @property
    def on_failure(self) -> FailureMode:
        """
        Return the configured failure behavior. If not set explicitly, this will be cascaded
        from the base kubernetes configuration (or its default)
        """
        return self.target_controller_config.on_failure

    @property
    def main_cpu(self) -> CPU:
        """
        Return the current CPU setting for the main containers.
        """
        # Determine the value in priority order from the config
        resource_requirements = self.main_container.get_resource_requirements('cpu')
        value = resource_requirements.get(
            next(filter(lambda r: resource_requirements[r] is not None, self.container_config.cpu.get), None)
        )
        millicores = Millicore.parse(value)

        # NOTE: use copy + update to accept values from mainline outside of our range
        cpu = self.container_config.cpu.copy(update={"pinned": True, "value": millicores})
        cpu.request = resource_requirements.get(ResourceRequirement.request)
        cpu.limit = resource_requirements.get(ResourceRequirement.limit)
        return cpu

    @property
    def main_memory(self) -> Memory:
        """
        Return the current Memory setting for the main containers.
        """
        # Determine the value in priority order from the config
        resource_requirements = self.main_container.get_resource_requirements('memory')
        value = resource_requirements.get(
            next(filter(lambda r: resource_requirements[r] is not None, self.container_config.memory.get), None)
        )
        short_byte_size = ShortByteSize.validate(value)

        # NOTE: use copy + update to accept values from mainline outside of our range
        memory = self.container_config.memory.copy(update={"pinned": True, "value": short_byte_size})
        memory.request = resource_requirements.get(ResourceRequirement.request)
        memory.limit = resource_requirements.get(ResourceRequirement.limit)
        return memory

    @property
    def main_replicas(self) -> servo.Replicas:
        """
        Return the current Replicas setting for the main Pods Deployment.

        NOTE: This is a synthetic setting because the replica count of the main Deployment is not
        under out control. The min, max, and value are aligned on each synthetic read.
        """
        return servo.Replicas(
            min=0,
            max=99999,
            value=self.target_controller.replicas,
            pinned=True,
        )

    @property
    def main_name(self) -> str:
        """Return the name for identifying the main instance settings & metrics.

        The name respects the alias defined in the config or else synthesizes a name from the Deployment
        and Container names.
        """
        return (
            self.container_config.alias
            or f"{self.target_controller_config.name}/{self.container_config.name}"
        )

    def to_components(self) -> List[servo.Component]:
        """
        Return a Component representation of the canary and its reference target.

        Note that all settings on the target are implicitly pinned because only the canary
        is to be modified during optimization.
        """
        return [
            servo.Component(
                name=self.main_name,
                settings=[
                    self.main_cpu,
                    self.main_memory,
                    self.main_replicas,
                ],
            ),
            servo.Component(
                name=self.name,
                settings=[
                    self.tuning_cpu,
                    self.tuning_memory,
                    self.tuning_replicas,
                ],
            ),
        ]

    async def rollback(self, error: Optional[Exception] = None) -> None:
        """
        Not supported. Raises a TypeError when called.

        Rollbacks are not supported by the canary optimization strategy
        because they are dependent on Kubernetes Deployments.
        """
        raise TypeError(
            (
                "rollback is not supported under the canary optimization strategy because rollbacks are applied to "
                "Kubernetes Deployment objects and canary optimization is performed against a standalone Pod."
            )
        )

    async def destroy(self, error: Optional[Exception] = None) -> None:
        if await self.delete_tuning_pod(raise_if_not_found=False) is None:
            self.logger.debug(f'no tuning pod exists, ignoring destroy')
            return

        self.logger.success(f'destroyed tuning Pod "{self.tuning_pod_name}"')

    async def handle_error(self, error: Exception) -> bool:
        if self.on_failure == FailureMode.rollback or self.on_failure == FailureMode.destroy:
            # Ensure that we chain any underlying exceptions that may occur
            try:
                if self.on_failure == FailureMode.rollback:
                    self.logger.warning(
                        f"cannot rollback a tuning Pod: falling back to destroy: {error}"
                    )

                await asyncio.wait_for(self.destroy(), timeout=self.timeout.total_seconds())

                # create a new canary against baseline
                self.logger.info(
                    "creating new tuning pod against baseline following failed adjust"
                )
                await self._configure_tuning_pod_template_spec()  # reset to baseline from the target controller
                self.tuning_pod = await self.create_or_recreate_tuning_pod()

                raise error # Always communicate errors to backend unless ignored

            except Exception as handler_error:
                raise handler_error from error

        else:
            return await super().handle_error(error)


    async def is_ready(self) -> bool:
        is_ready, restart_count = await asyncio.gather(
            self.tuning_pod.is_ready(),
            self.tuning_pod.get_restart_count()
        )
        return is_ready and restart_count == 0

    async def raise_for_status(self) -> None:
        """Raise an exception if in an unhealthy state."""
        await self.tuning_pod.raise_for_status(adjustments=self.adjustments)


    class Config:
        arbitrary_types_allowed = True
        extra = pydantic.Extra.forbid


class KubernetesOptimizations(pydantic.BaseModel, servo.logging.Mixin):
    """
    Models the state of resources under optimization in a Kubernetes cluster.
    """

    config: "KubernetesConfiguration"
    namespace: Namespace
    optimizations: List[BaseOptimization]
    runtime_id: str
    spec_id: str
    version_id: str

    @classmethod
    async def create(
        cls, config: "KubernetesConfiguration"
    ) -> "KubernetesOptimizations":
        """
        Read the state of all components under optimization from the cluster and return an object representation.
        """
        namespace = await Namespace.read(config.namespace)
        optimizations: List[BaseOptimization] = []
        images = {}
        runtime_ids = {}
        pod_tmpl_specs = {}

        for deployment_or_rollout_config in (config.deployments or []) + (config.rollouts or []):
            if deployment_or_rollout_config.strategy == OptimizationStrategy.default:
                if isinstance(deployment_or_rollout_config, RolloutConfiguration):
                    raise NotImplementedError("Saturation mode not currently supported on Argo Rollouts")
                optimization = await DeploymentOptimization.create(
                    deployment_or_rollout_config, timeout=deployment_or_rollout_config.timeout
                )
                deployment_or_rollout = optimization.deployment
                container = optimization.container
            elif deployment_or_rollout_config.strategy == OptimizationStrategy.canary:
                optimization = await CanaryOptimization.create(
                    deployment_or_rollout_config, timeout=deployment_or_rollout_config.timeout
                )
                deployment_or_rollout = optimization.target_controller
                container = optimization.main_container

                # Ensure the canary is available
                # TODO: We don't want to do this implicitly but this is a first step
                if not optimization.tuning_pod:
                    servo.logger.info("Creating new tuning pod...")
                    await optimization.create_tuning_pod()
            else:
                raise ValueError(
                    f"unknown optimization strategy: {deployment_or_rollout_config.strategy}"
                )

            optimizations.append(optimization)

            # compile artifacts for checksum calculation
            pods = await deployment_or_rollout.get_pods()
            runtime_ids[optimization.name] = [pod.uid for pod in pods]
            pod_tmpl_specs[deployment_or_rollout.name] = deployment_or_rollout.obj.spec.template.spec
            images[container.name] = container.image

        # Compute checksums for change detection
        spec_id = servo.utilities.hashing.get_hash([pod_tmpl_specs[k] for k in sorted(pod_tmpl_specs.keys())])
        runtime_id = servo.utilities.hashing.get_hash(runtime_ids)
        version_id = servo.utilities.hashing.get_hash([images[k] for k in sorted(images.keys())])

        return KubernetesOptimizations(
            config=config,
            namespace=namespace,
            optimizations=optimizations,
            spec_id=spec_id,
            runtime_id=runtime_id,
            version_id=version_id,
        )

    def to_components(self) -> List[servo.Component]:
        """
        Return a list of Component objects modeling the state of local optimization activities.

        Components are the canonical representation of systems under optimization. They
        are used for data exchange with the Opsani API
        """
        components = list(map(lambda opt: opt.to_components(), self.optimizations))
        return list(itertools.chain(*components))

    def to_description(self) -> servo.Description:
        """
        Return a representation of the current state as a Description object.

        Description objects are used to report state to the Opsani API in order
        to synchronize with the Optimizer service.

        Returns:
            A Description of the current state.
        """
        return servo.Description(components=self.to_components())

    def find_optimization(self, name: str) -> Optional[BaseOptimization]:
        """
        Find and return an optimization by name.
        """
        return next(filter(lambda a: a.name == name, self.optimizations), None)

    async def apply(self, adjustments: List[servo.Adjustment]) -> None:
        """
        Apply a sequence of adjustments and wait for them to take effect on the cluster.
        """
        # Exit early if there is nothing to do
        if not adjustments:
            self.logger.debug("early exiting from adjust: no adjustments")
            return

        summary = f"[{', '.join(list(map(str, adjustments)))}]"
        self.logger.info(
            f"Applying {len(adjustments)} Kubernetes adjustments: {summary}"
        )

        # Adjust settings on the local data model
        for adjustment in adjustments:
            if adjustable := self.find_optimization(adjustment.component_name):
                self.logger.info(f"adjusting {adjustment.component_name}: {adjustment}")
                adjustable.adjust(adjustment)

            else:
                self.logger.debug(f'ignoring unrecognized adjustment "{adjustment}"')

        # Apply the changes to Kubernetes and wait for the results
        timeout = self.config.timeout
        if self.optimizations:
            self.logger.debug(
                f"waiting for adjustments to take effect on {len(self.optimizations)} optimizations"
            )
            try:
                gather_apply = asyncio.gather(
                    *list(map(lambda a: a.apply(), self.optimizations)),
                    return_exceptions=True,
                )
                results = await asyncio.wait_for(gather_apply, timeout=timeout.total_seconds() + 60) # allow sub-optimization timeouts to expire first

            except asyncio.exceptions.TimeoutError as error:
                self.logger.error(
                    f"timed out after {timeout} + 60s waiting for adjustments to apply"
                )
                # Prevent "_GatheringFuture exception was never retrieved" warning if the above wait_for raises a timeout error
                # https://bugs.python.org/issue29432
                try:
                    await gather_apply
                except asyncio.CancelledError:
                    pass
                for optimization in self.optimizations:
                    if await optimization.handle_error(error):
                        # Stop error propagation once it has been handled
                        break
                raise  # No results to process in this case, reraise timeout if handlers didn't

            for result in results:
                if isinstance(result, Exception):
                    for optimization in self.optimizations:
                        if await optimization.handle_error(result):
                            # Stop error propagation once it has been handled
                            break
        else:
            self.logger.warning(f"failed to apply adjustments: no adjustables")

        # TODO: Run sanity checks to look for out of band changes

    async def raise_for_status(self) -> None:
        handle_error_tasks = []
        def _raise_for_task(task: asyncio.Task, optimization: BaseOptimization) -> None:
            if task.done() and not task.cancelled():
                if exception := task.exception():
                    handle_error_tasks.append(asyncio.create_task(optimization.handle_error(exception)))

        tasks = []
        for optimization in self.optimizations:
            task = asyncio.create_task(optimization.raise_for_status())
            task.add_done_callback(functools.partial(_raise_for_task, optimization=optimization))
            tasks.append(task)

        for future in asyncio.as_completed(tasks, timeout=self.config.timeout.total_seconds()):
            try:
                await future
            except Exception as error:
                servo.logger.exception(f"Optimization failed with error: {error}")

        # TODO: first handler to raise will likely interrupt other tasks.
        #   Gather with return_exceptions=True and aggregate resulting exceptions before raising
        await asyncio.gather(*handle_error_tasks)

    async def is_ready(self):
        if self.optimizations:
            self.logger.debug(
                f"Checking for readiness of {len(self.optimizations)} optimizations"
            )
            try:
                results = await asyncio.wait_for(
                    asyncio.gather(
                        *list(map(lambda a: a.is_ready(), self.optimizations)),
                    ),
                    timeout=self.config.timeout.total_seconds()
                )

                return all(results)

            except asyncio.TimeoutError:
                return False

        else:
            return True


    class Config:
        arbitrary_types_allowed = True


DNSSubdomainName = pydantic.constr(
    strip_whitespace=True,
    min_length=1,
    max_length=253,
    regex="^[0-9a-zA-Z]([0-9a-zA-Z\\.-])*[0-9A-Za-z]$",
)
DNSSubdomainName.__doc__ = (
    """DNSSubdomainName models a Kubernetes DNS Subdomain Name used as the name for most resource types.

    Valid DNS Subdomain Names conform to [RFC 1123](https://tools.ietf.org/html/rfc1123) and must:
        * contain no more than 253 characters
        * contain only lowercase alphanumeric characters, '-' or '.'
        * start with an alphanumeric character
        * end with an alphanumeric character

    See https://kubernetes.io/docs/concepts/overview/working-with-objects/names/#dns-subdomain-names
    """
)



DNSLabelName = pydantic.constr(
    strip_whitespace=True,
    min_length=1,
    max_length=63,
    regex="^[0-9a-zA-Z]([0-9a-zA-Z-])*[0-9A-Za-z]$",
)
DNSLabelName.__doc__ = (
    """DNSLabelName models a Kubernetes DNS Label Name identified used to name some resource types.

    Valid DNS Label Names conform to [RFC 1123](https://tools.ietf.org/html/rfc1123) and must:
        * contain at most 63 characters
        * contain only lowercase alphanumeric characters or '-'
        * start with an alphanumeric character
        * end with an alphanumeric character

    See https://kubernetes.io/docs/concepts/overview/working-with-objects/names/#dns-label-names
    """
)


ContainerTagName = pydantic.constr(
    strip_whitespace=True,
    min_length=1,
    max_length=128,
    regex="^[0-9a-zA-Z]([0-9a-zA-Z_\\.\\-/:@])*$",
)  # NOTE: This regex is not a full validation
ContainerTagName.__doc__ = (
    """ContainerTagName models the name of a container referenced in a Kubernetes manifest.

    Valid container tags must:
        * be valid ASCII and may contain lowercase and uppercase letters, digits, underscores, periods and dashes.
        * not start with a period or a dash
        * may contain a maximum of 128 characters
    """
)


class EnvironmentConfiguration(servo.BaseConfiguration):
    ...


class CommandConfiguration(servo.BaseConfiguration):
    ...


class ContainerConfiguration(servo.BaseConfiguration):
    """
    The ContainerConfiguration class models the configuration of an optimizeable container within a Kubernetes Deployment.
    """

    name: ContainerTagName
    alias: Optional[ContainerTagName]
    command: Optional[str]  # TODO: create model...
    cpu: CPU
    memory: Memory
    env: Optional[List[str]]  # TODO: create model...



class OptimizationStrategy(str, enum.Enum):
    """
    OptimizationStrategy is an enumeration of the possible ways to perform optimization on a Kubernetes Deployment.
    """

    default = "default"
    """The default strategy directly applies adjustments to the target Deployment and its containers.
    """

    canary = "canary"
    """The canary strategy creates a servo managed standalone tuning Pod based on the target Deployment and makes
    adjustments to it instead of the Deployment itself.
    """


class BaseOptimizationStrategyConfiguration(pydantic.BaseModel):
    type: OptimizationStrategy = pydantic.Field(..., const=True)

    def __eq__(self, other) -> bool:
        if isinstance(other, OptimizationStrategy):
            return self.type == other
        return super().__eq__(other)

    class Config:
        extra = pydantic.Extra.forbid


class DefaultOptimizationStrategyConfiguration(BaseOptimizationStrategyConfiguration):
    type = pydantic.Field(OptimizationStrategy.default, const=True)


class CanaryOptimizationStrategyConfiguration(BaseOptimizationStrategyConfiguration):
    type = pydantic.Field(OptimizationStrategy.canary, const=True)
    alias: Optional[ContainerTagName]


class FailureMode(str, enum.Enum):
    """
    The FailureMode enumeration defines how to handle a failed adjustment of a Kubernetes resource.
    """

    rollback = "rollback"
    destroy = "destroy"
    ignore = "ignore"
    exception = "exception"

    @classmethod
    def options(cls) -> List[str]:
        """
        Return a list of strings that identifies all failure mode configuration options.
        """
        return list(map(lambda mode: mode.value, cls.__members__.values()))


class PermissionSet(pydantic.BaseModel):
    """Permissions objects model Kubernetes permissions granted through RBAC."""

    group: str
    resources: List[str]
    verbs: List[str]


STANDARD_PERMISSIONS = [
    PermissionSet(
        group="apps",
        resources=["deployments", "replicasets"],
        verbs=["get", "list", "watch", "update", "patch"],
    ),
    PermissionSet(
        group="",
        resources=["namespaces"],
        verbs=["get"],
    ),
    PermissionSet(
        group="",
        resources=["pods", "pods/logs", "pods/status"],
        verbs=["create", "delete", "get", "list", "watch"],
    ),
]

ROLLOUT_PERMISSIONS = [
    PermissionSet(
        group="argoproj.io",
        resources=["rollouts", "rollouts/status"],
        verbs=["get", "list", "watch", "update", "patch"],
    ),
]


class BaseKubernetesConfiguration(servo.BaseConfiguration):
    """
    BaseKubernetesConfiguration provides a set of configuration primitives for optimizable Kubernetes resources.

    Child classes of `BaseKubernetesConfiguration` such as the `DeploymentConfiguration` can benefit from
    the cascading configuration behavior implemented on the `KubernetesConfiguration` class.

    Common settings will be cascaded from the containing class for attributes if they have not been explicitly set
    and are equal to the default value. Settings that are mandatory in the superclass (such as timeout and namespace)
    but are available for override should be declared as optional on `BaseKubernetesConfiguration` and overridden and
    declared as mandatory in `BaseKubernetesConfiguration`'.
    """

    kubeconfig: Optional[pydantic.FilePath] = pydantic.Field(
        description="Path to the kubeconfig file. If `None`, use the default from the environment.",
    )
    context: Optional[str] = pydantic.Field(description="Name of the kubeconfig context to use.")
    namespace: Optional[DNSSubdomainName] = pydantic.Field(
        description="Kubernetes namespace where the target deployments are running.",
    )
    settlement: Optional[servo.Duration] = pydantic.Field(
        description="Duration to observe the application after an adjust to ensure the deployment is stable. May be overridden by optimizer supplied `control.adjust.settlement` value."
    )
    on_failure: FailureMode = pydantic.Field(
        FailureMode.exception,
        description=f"How to handle a failed adjustment. Options are: {servo.utilities.strings.join_to_series(list(FailureMode.__members__.values()))}",
    )
    timeout: Optional[servo.Duration] = pydantic.Field(
        description="Time interval to wait before considering Kubernetes operations to have failed."
    )


StrategyTypes = Union[
    OptimizationStrategy,
    DefaultOptimizationStrategyConfiguration,
    CanaryOptimizationStrategyConfiguration,
]


class DeploymentConfiguration(BaseKubernetesConfiguration):
    """
    The DeploymentConfiguration class models the configuration of an optimizable Kubernetes Deployment.
    """

    name: DNSSubdomainName
    containers: List[ContainerConfiguration]
    strategy: StrategyTypes = OptimizationStrategy.default
    replicas: servo.Replicas

class RolloutConfiguration(BaseKubernetesConfiguration):
    """
    The RolloutConfiguration class models the configuration of an optimizable Argo Rollout.
    """

    name: DNSSubdomainName
    containers: List[ContainerConfiguration]
    strategy: StrategyTypes = OptimizationStrategy.canary
    replicas: servo.Replicas


class KubernetesConfiguration(BaseKubernetesConfiguration):
    namespace: DNSSubdomainName = DNSSubdomainName("default")
    timeout: servo.Duration = "5m"
    permissions: List[PermissionSet] = pydantic.Field(
        STANDARD_PERMISSIONS,
        description="Permissions required by the connector to operate in Kubernetes.",
    )

    deployments: Optional[List[DeploymentConfiguration]] = pydantic.Field(
        description="Deployments to be optimized.",
    )

    rollouts: Optional[List[RolloutConfiguration]] = pydantic.Field(
        description="Argo rollouts to be optimized.",
    )

    @pydantic.root_validator
    def check_deployment_and_rollout(cls, values):
        if (not values.get('deployments')) and (not values.get('rollouts')):
            raise ValueError("No optimization target(s) were specified")

        return values

    @classmethod
    def generate(cls, **kwargs) -> "KubernetesConfiguration":
        return cls(
            namespace="default",
            description="Update the namespace, deployment, etc. to match your Kubernetes cluster",
            deployments=[
                DeploymentConfiguration(
                    name="app",
                    replicas=servo.Replicas(
                        min=1,
                        max=2,
                    ),
                    containers=[
                        ContainerConfiguration(
                            name="opsani/fiber-http:latest",
                            cpu=CPU(min="250m", max=4, step="125m"),
                            memory=Memory(min="256MiB", max="4GiB", step="128MiB"),
                        )
                    ],
                )
            ],
            **kwargs,
        )

    def __init__(self, *args, **kwargs) -> None: # noqa: D107
        super().__init__(*args, **kwargs)
        self.cascade_common_settings()

    def cascade_common_settings(self, *, overwrite: bool = False) -> None:
        """
        Apply common settings to child models that inherit from BaseKubernetesConfiguration.

        This method provides enables hierarchical overrides of common configuration values
        based on shared inheritance. Each attribute is introspected and if it inherits from
        `BaseKubernetesConfiguration`, any common attribute values are copied onto the child
        model, cascading them downward. Only attributes whose value is equal to the default
        and have not been explicitly set are updated.

        # FIXME: Cascaded settings should only be optional if they can be optional at the top level. Right now we are implying that namespace can be None as well.
        """
        for name, field in self.__fields__.items():
            if issubclass(field.type_, BaseKubernetesConfiguration):
                attribute = getattr(self, name)
                for obj in (
                    attribute if isinstance(attribute, Collection) else [attribute]
                ):
                    # don't cascade if optional and not set
                    if obj is None:
                        continue
                    for (
                        field_name,
                        field,
                    ) in BaseKubernetesConfiguration.__fields__.items():
                        if field_name in servo.BaseConfiguration.__fields__:
                            # don't cascade from the base class
                            continue

                        if field_name in obj.__fields_set__ and not overwrite:
                            self.logger.trace(
                                f"skipping config cascade for field '{field_name}' set with value '{getattr(obj, field_name)}'"
                            )
                            continue

                        current_value = getattr(obj, field_name)
                        if overwrite or current_value == field.default:
                            parent_value = getattr(self, field_name)
                            setattr(obj, field_name, parent_value)
                            self.logger.trace(
                                f"cascaded setting '{field_name}' from KubernetesConfiguration to child '{attribute}': value={parent_value}"
                            )

                        else:
                            self.logger.trace(
                                f"declining to cascade value to field '{field_name}': the default value is set and overwrite is false"
                            )

    async def load_kubeconfig(self) -> None:
        """
        Asynchronously load the Kubernetes configuration
        """
        config_file = pathlib.Path(self.kubeconfig or kubernetes_asyncio.config.kube_config.KUBE_CONFIG_DEFAULT_LOCATION).expanduser()
        if config_file.exists():
            await kubernetes_asyncio.config.load_kube_config(
                config_file=str(config_file),
                context=self.context,
            )
        elif os.getenv("KUBERNETES_SERVICE_HOST"):
            kubernetes_asyncio.config.load_incluster_config()
        else:
            raise RuntimeError(
                f"unable to configure Kubernetes client: no kubeconfig file nor in-cluser environment variables found"
            )


KubernetesOptimizations.update_forward_refs()
DeploymentOptimization.update_forward_refs()
CanaryOptimization.update_forward_refs()


class KubernetesChecks(servo.BaseChecks):
    """Checks for ensuring that the Kubernetes connector is ready to run."""

    config: KubernetesConfiguration

    @servo.require("Connectivity to Kubernetes")
    async def check_connectivity(self) -> None:
        async with kubernetes_asyncio.client.api_client.ApiClient() as api:
            v1 =kubernetes_asyncio.client.VersionApi(api)
            await v1.get_code()

    @servo.warn("Kubernetes version")
    async def check_version(self) -> None:
        async with kubernetes_asyncio.client.api_client.ApiClient() as api:
            v1 =kubernetes_asyncio.client.VersionApi(api)
            version = await v1.get_code()
            assert int(version.major) >= 1
            # EKS sets minor to "17+"
            assert int(int("".join(c for c in version.minor if c.isdigit()))) >= 16

    @servo.require("Required permissions")
    async def check_permissions(self) -> None:
        async with kubernetes_asyncio.client.api_client.ApiClient() as api:
            v1 = kubernetes_asyncio.client.AuthorizationV1Api(api)
            required_permissions = self.config.permissions
            if self.config.rollouts:
                required_permissions.append(ROLLOUT_PERMISSIONS)
            for permission in required_permissions:
                for resource in permission.resources:
                    for verb in permission.verbs:
                        attributes = kubernetes_asyncio.client.models.V1ResourceAttributes(
                            namespace=self.config.namespace,
                            group=permission.group,
                            resource=resource,
                            verb=verb,
                        )

                        spec =kubernetes_asyncio.client.models.V1SelfSubjectAccessReviewSpec(
                            resource_attributes=attributes
                        )
                        review =kubernetes_asyncio.client.models.V1SelfSubjectAccessReview(spec=spec)
                        access_review = await v1.create_self_subject_access_review(
                            body=review
                        )
                        assert (
                            access_review.status.allowed
                        ), f'Not allowed to "{verb}" resource "{resource}"'

    @servo.require('Namespace "{self.config.namespace}" is readable')
    async def check_namespace(self) -> None:
        await Namespace.read(self.config.namespace)

    @servo.multicheck('Deployment "{item.name}" is readable')
    async def check_deployments(self) -> Tuple[Iterable, servo.CheckHandler]:
        async def check_dep(dep_config: DeploymentConfiguration) -> None:
            await Deployment.read(dep_config.name, dep_config.namespace)

        return (self.config.deployments or []), check_dep

    @servo.multicheck('Rollout "{item.name}" is readable')
    async def check_rollouts(self) -> Tuple[Iterable, servo.CheckHandler]:
        async def check_rol(rol_config: RolloutConfiguration) -> None:
            await Rollout.read(rol_config.name, rol_config.namespace)

        return (self.config.rollouts or []), check_rol

    async def _check_container_resource_requirements(
        self,
        target_controller: Union[Deployment, Rollout],
        target_config: Union[DeploymentConfiguration, RolloutConfiguration]
    ) -> None:
        for cont_config in target_config.containers:
            container = target_controller.find_container(cont_config.name)
            assert container, f"{type(target_controller).__name__} {target_config.name} has no container {cont_config.name}"

            for resource in Resource.values():
                current_state = None
                container_requirements = container.get_resource_requirements(resource)
                get_requirements = getattr(cont_config, resource).get
                for requirement in get_requirements:
                    current_state = container_requirements.get(requirement)
                    if current_state:
                        break

                assert current_state, (
                    f"{type(target_controller).__name__} {target_config.name} target container {cont_config.name} spec does not define the resource {resource}. "
                    f"At least one of the following must be specified: {', '.join(map(lambda req: req.resources_key, get_requirements))}"
                )

    @servo.multicheck('Containers in the "{item.name}" Deployment have resource requirements')
    async def check_resource_requirements(self) -> Tuple[Iterable, servo.CheckHandler]:
        async def check_dep_resource_requirements(
            dep_config: DeploymentConfiguration,
        ) -> None:
            deployment = await Deployment.read(dep_config.name, dep_config.namespace)
            await self._check_container_resource_requirements(deployment, dep_config)

        return (self.config.deployments or []), check_dep_resource_requirements


    @servo.multicheck('Containers in the "{item.name}" Rollout have resource requirements')
    async def check_rollout_resource_requirements(self) -> Tuple[Iterable, servo.CheckHandler]:
        async def check_rol_resource_requirements(
            rol_config: RolloutConfiguration,
        ) -> None:
            rollout = await Rollout.read(rol_config.name, rol_config.namespace)
            await self._check_container_resource_requirements(rollout, rol_config)

        return (self.config.rollouts or []), check_rol_resource_requirements


    @servo.multicheck('Deployment "{item.name}" is ready')
    async def check_deployments_are_ready(self) -> Tuple[Iterable, servo.CheckHandler]:
        async def check_deployment(dep_config: DeploymentConfiguration) -> None:
            deployment = await Deployment.read(dep_config.name, dep_config.namespace)
            if not await deployment.is_ready():
                raise RuntimeError(f'Deployment "{deployment.name}" is not ready')

        return (self.config.deployments or []), check_deployment

    @servo.multicheck('Rollout "{item.name}" is ready')
    async def check_rollouts_are_ready(self) -> Tuple[Iterable, servo.CheckHandler]:
        async def check_rollout(rol_config: RolloutConfiguration) -> None:
            rollout = await Rollout.read(rol_config.name, rol_config.namespace)
            if not await rollout.is_ready():
                raise RuntimeError(f'Rollout "{rollout.name}" is not ready')

        return (self.config.rollouts or []), check_rollout


@servo.metadata(
    description="Kubernetes adjust connector",
    version="1.5.0",
    homepage="https://github.com/opsani/kubernetes-connector",
    license=servo.License.apache2,
    maturity=servo.Maturity.stable,
)
class KubernetesConnector(servo.BaseConnector):
    config: KubernetesConfiguration

    @servo.on_event()
    async def attach(self, servo_: servo.Servo) -> None:
        # Ensure we are ready to talk to Kubernetes API
        await self.config.load_kubeconfig()

        self.telemetry[f"{self.name}.namespace"] = self.config.namespace

        with self.logger.catch(level="DEBUG", message=f"Unable to set version telemetry for connector {self.name}"):
            async with kubernetes_asyncio.client.api_client.ApiClient() as api:
                v1 =kubernetes_asyncio.client.VersionApi(api)
                version_obj = await v1.get_code()
                self.telemetry[f"{self.name}.version"] = f"{version_obj.major}.{version_obj.minor}"
                self.telemetry[f"{self.name}.platform"] = version_obj.platform

    @servo.on_event()
    async def detach(self, servo_: servo.Servo) -> None:
        self.telemetry.remove(f"{self.name}.namespace")
        self.telemetry.remove(f"{self.name}.version")
        self.telemetry.remove(f"{self.name}.platform")

    @servo.on_event()
    async def describe(self, control: servo.Control = servo.Control()) -> servo.Description:
        state = await self._create_optimizations()
        return state.to_description()

    @servo.on_event()
    async def components(self) -> List[servo.Component]:
        state = await self._create_optimizations()
        return state.to_components()

    @servo.before_event(servo.Events.measure)
    async def before_measure(self, *, metrics: List[str] = None, control: servo.Control = servo.Control()) -> None:
        # Build state before a measurement to ensure all necessary setup is done
        # (e.g., Tuning Pod is up and running)
        await self._create_optimizations()

    @servo.on_event()
    async def adjust(
        self, adjustments: List[servo.Adjustment], control: servo.Control = servo.Control()
    ) -> servo.Description:
        state = await self._create_optimizations()

        # Apply the adjustments and emit progress status
        progress_logger = lambda p: self.logger.info(
            p.annotate(f"waiting up to {p.timeout} for adjustments to be applied...", prefix=False),
            progress=p.progress,
        )
        progress = servo.EventProgress(timeout=self.config.timeout)
        future = asyncio.create_task(state.apply(adjustments))
        future.add_done_callback(lambda _: progress.trigger())

        await asyncio.gather(
            future,
            progress.watch(progress_logger),
        )

        # Handle settlement
        settlement = control.settlement or self.config.settlement
        if settlement:
            self.logger.info(
                f"Settlement duration of {settlement} requested, waiting for pods to settle..."
            )
            progress = servo.DurationProgress(settlement)
            progress_logger = lambda p: self.logger.info(
                p.annotate(f"waiting {settlement} for pods to settle...", False),
                progress=p.progress,
            )
            async def readiness_monitor() -> None:
                while not progress.finished:
                    if not await state.is_ready():
                        # Raise a specific exception if the optimization defines one
                        try:
                            await state.raise_for_status()
                        except servo.AdjustmentRejectedError as e:
                            # Update rejections with start-failed to indicate the initial rollout was successful
                            if e.reason == "start-failed":
                                e.reason = "unstable"
                            raise

                    await asyncio.sleep(servo.Duration('50ms').total_seconds())

            await asyncio.gather(
                progress.watch(progress_logger),
                readiness_monitor()
            )
            if not await state.is_ready():
                self.logger.warning("Rejection triggered without running error handler")
                raise servo.AdjustmentRejectedError(
                    "Optimization target became unready after adjustment settlement period (WARNING: error handler was not run)",
                    reason="unstable"
                )
            self.logger.info(
                f"Settlement duration of {settlement} has elapsed, resuming optimization."
            )

        description = state.to_description()
        return description

    @servo.on_event()
    async def check(
        self,
        matching: Optional[servo.CheckFilter],
        halt_on: Optional[servo.ErrorSeverity] = servo.ErrorSeverity.critical,
    ) -> List[servo.Check]:
        return await KubernetesChecks.run(
            self.config, matching=matching, halt_on=halt_on
        )

    async def _create_optimizations(self) -> KubernetesOptimizations:
        # Build a KubernetesOptimizations object with progress reporting
        # This ensures that the Servo isn't reported as offline
        progress_logger = lambda p: self.logger.info(
            p.annotate(f"waiting up to {p.timeout} for Kubernetes optimization setup to complete", prefix=False),
            progress=p.progress,
        )
        progress = servo.EventProgress(timeout=self.config.timeout)
        future = asyncio.create_task(KubernetesOptimizations.create(self.config))
        future.add_done_callback(lambda _: progress.trigger())

        await asyncio.gather(
            future,
            progress.watch(progress_logger),
        )

        return future.result()


def selector_string(selectors: Mapping[str, str]) -> str:
    """Create a selector string from the given dictionary of selectors.

    Args:
        selectors: The selectors to stringify.

    Returns:
        The selector string for the given dictionary.
    """
    return ",".join([f"{k}={v}" for k, v in selectors.items()])


def selector_kwargs(
    fields: Mapping[str, str] = None,
    labels: Mapping[str, str] = None,
) -> Dict[str, str]:
    """Create a dictionary of kwargs for Kubernetes object selectors.

    Args:
        fields: A mapping of fields used to restrict the returned collection of
            Objects to only those which match these field selectors. By default,
            no restricting is done.
        labels: A mapping of labels used to restrict the returned collection of
            Objects to only those which match these label selectors. By default,
            no restricting is done.

    Returns:
        A dictionary that can be used as kwargs for many Kubernetes API calls for
        label and field selectors.
    """
    kwargs = {}
    if fields is not None:
        kwargs["field_selector"] = selector_string(fields)
    if labels is not None:
        kwargs["label_selector"] = selector_string(labels)

    return kwargs

class ConfigMap(KubernetesModel):
    """Kubetest wrapper around a Kubernetes `ConfigMap`_ API Object.

    The actual ``kubernetes.client.V1ConfigMap`` instance that this
    wraps can be accessed via the ``obj`` instance member.

    This wrapper provides some convenient functionality around the
    API Object and provides some state management for the `ConfigMap`_.

    .. _ConfigMap:
        https://kubernetes.io/docs/reference/generated/kubernetes-api/v1.18/#configmap-v1-core
    """

    obj_type =kubernetes_asyncio.client.V1ConfigMap

    api_clients = {
        "preferred":kubernetes_asyncio.client.CoreV1Api,
        "v1":kubernetes_asyncio.client.CoreV1Api,
    }

    @classmethod
    async def read(cls, name: str, namespace: str) -> "ConfigMap":
        """Read a ConfigMap by name under the given namespace.

        Args:
            name: The name of the Deployment to read.
            namespace: The namespace to read the Deployment from.
        """

        async with cls.preferred_client() as api_client:
            obj = await api_client.read_namespaced_config_map(name, namespace)
            return ConfigMap(obj)

    async def create(self, namespace: str = None) -> None:
        """Create the ConfigMap under the given namespace.

        Args:
            namespace: The namespace to create the ConfigMap under.
                If the ConfigMap was loaded via the kubetest client, the
                namespace will already be set, so it is not needed here.
                Otherwise, the namespace will need to be provided.
        """
        if namespace is None:
            namespace = self.namespace

        servo.logger.info(f'creating configmap "{self.name}" in namespace "{self.namespace}"')
        servo.logger.debug(f"configmap: {self.obj}")

        self.obj = await self.api_client.create_namespaced_config_map(
            namespace=namespace,
            body=self.obj,
        )

    async def patch(self) -> None:
        """
        Patches a ConfigMap.
        """
        self.logger.info(f'patching ConfigMap "{self.name}"')
        self.logger.trace(f"ConfigMap: {self.obj}")
        async with self.api_client() as api_client:
            await api_client.patch_namespaced_config_map(
                name=self.name,
                namespace=self.namespace,
                body=self.obj,
            )

    async def delete(self, options:kubernetes_asyncio.client.V1DeleteOptions = None) ->kubernetes_asyncio.client.V1Status:
        """Delete the ConfigMap.

        This method expects the ConfigMap to have been loaded or otherwise
        assigned a namespace already. If it has not, the namespace will need
        to be set manually.

        Args:
             options: Options for ConfigMap deletion.

        Returns:
            The status of the delete operation.
        """
        if options is None:
            options = kubernetes_asyncio.client.V1DeleteOptions()

        servo.logger.info(f'deleting configmap "{self.name}"')
        servo.logger.debug(f"delete options: {options}")
        servo.logger.debug(f"configmap: {self.obj}")

        return await self.api_client.delete_namespaced_config_map(
            name=self.name,
            namespace=self.namespace,
            body=options,
        )

    async def refresh(self) -> None:
        """Refresh the underlying Kubernetes ConfigMap resource."""
        self.obj = await self.api_client.read_namespaced_config_map(
            name=self.name,
            namespace=self.namespace,
        )

    async def is_ready(self) -> bool:
        """Check if the ConfigMap is in the ready state.

        ConfigMaps do not have a "status" field to check, so we will
        measure their readiness status by whether or not they exist
        on the cluster.

        Returns:
            True if in the ready state; False otherwise.
        """
        try:
            await self.refresh()
        except:  # noqa
            return False

        return True

def dns_subdomainify(name: str) -> str:
    """
    Valid DNS Subdomain Names conform to [RFC 1123](https://tools.ietf.org/html/rfc1123) and must:
        * contain no more than 253 characters
        * contain only lowercase alphanumeric characters, '-' or '.'
        * start with an alphanumeric character
        * end with an alphanumeric character

    See https://kubernetes.io/docs/concepts/overview/working-with-objects/names/#dns-subdomain-names
    """

    # lowercase alphanumerics
    name = name.lower()

    # replace slashes with dots
    name = re.sub(r'\/', '.', name)

    # replace whitespace with hyphens
    name = re.sub(r'\s', '-', name)

    # strip any remaining disallowed characters
    name = re.sub(r'/[^a-z0-9\.\-]+/g', '', name)

    # truncate to our maximum length
    name = name[:253]

    # ensure starts with an alphanumeric by prefixing with `0-`
    boundaryRegex = re.compile('^[a-z0-9]')
    if not boundaryRegex.match(name):
        name = ('0-' + name)[:253]

    # ensure ends with an alphanumeric by suffixing with `-1`
    if not boundaryRegex.match(name[-1]):
        name = name[:251] + '-1'

    return name

def dns_labelize(name: str) -> str:
    """
    Transform a string into a valid Kubernetes label value.

    Valid Kubernetes label values:
        * must be 63 characters or less (cannot be empty)
        * must begin and end with an alphanumeric character ([a-z0-9A-Z])
        * may contain dashes (-), underscores (_), dots (.), and alphanumerics between
    """

    # replace slashes with underscores
    name = re.sub(r'\/', '_', name)

    # replace whitespace with hyphens
    name = re.sub(r'\s', '-', name)

    # strip any remaining disallowed characters
    name = re.sub(r'[^a-z0-9A-Z\.\-_]+', '', name)

    # truncate to our maximum length
    name = name[:63]

    # ensure starts with an alphanumeric by prefixing with `0-`
    boundaryRegex = re.compile('[a-z0-9A-Z]')
    if not boundaryRegex.match(name[0]):
        name = ('0-' + name)[:63]

    # ensure ends with an alphanumeric by suffixing with `-1`
    if not boundaryRegex.match(name[-1]):
        name = name[:61] + '-1'

    return name


def set_container_resource_defaults_from_config(container: Container, config: ContainerConfiguration) -> None:
    for resource in Resource.values():
        # NOTE: cpu/memory stanza in container config
        resource_config = getattr(config, resource)
        requirements = container.get_resource_requirements(resource)
        servo.logger.debug(f"Loaded resource requirements for '{resource}': {requirements}")
        for requirement in ResourceRequirement:
            # Use the request/limit from the container.[cpu|memory].[request|limit] as default/override
            if resource_value := getattr(resource_config, requirement.name):
                if (existing_resource_value := requirements.get(requirement)) is None:
                    servo.logger.debug(f"Setting default value for {resource}.{requirement} to: {resource_value}")
                else:
                    servo.logger.debug(f"Overriding existing value for {resource}.{requirement} ({existing_resource_value}) to: {resource_value}")

                requirements[requirement] = resource_value

        servo.logger.debug(f"Setting resource requirements for '{resource}' to: {requirements}")
        container.set_resource_requirements(resource, requirements)
