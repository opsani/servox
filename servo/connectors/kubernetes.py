from __future__ import annotations, print_function

import abc
import asyncio
import enum
import os
from servo.types import Numeric
import time
from pathlib import Path
from typing import Callable, List, Optional, Dict, Any, Tuple
from kubetest.objects import namespace

from pydantic import BaseModel, ByteSize, Field, FilePath

from servo import (
    Adjustment,
    BaseChecks,
    BaseConfiguration,
    BaseConnector,
    Check,
    Component,
    Control,
    Description,
    Duration,
    DurationProgress,
    License,
    Maturity,
    Setting,
    SettingType,
    connector,
    on_event,
    get_hash
)
from kubernetes_asyncio import client, config as kubernetes_asyncio_config, watch
from kubernetes_asyncio.config.kube_config import KUBE_CONFIG_DEFAULT_LOCATION
from kubernetes_asyncio.client.api_client import ApiClient
import loguru
from loguru import logger as default_logger
from typing import ClassVar, Generator, Mapping, Protocol, Type, Union, cast, get_type_hints, runtime_checkable
from contextlib import asynccontextmanager


# TODO: This has behavior that removes request/limit if it exists
def set_rsrc(cp, sn, sv, sel="both"):
    rn = RESOURCE_MAP[sn]
    if sn == "mem":
        sv = str(round(sv, 3)) + "GiB"  # internal memory representation is in GiB
    else:
        sv = str(round(sv, 3))

    if sel == "request":
        cp.setdefault("resources", {}).setdefault("requests", {})[rn] = sv
        cp["resources"].setdefault("limits", {})[
            rn
        ] = None  # Remove corresponding limit if exists
    elif sel == "limit":
        cp.setdefault("resources", {}).setdefault("limits", {})[rn] = sv
        cp["resources"].setdefault("requests", {})[
            rn
        ] = None  # Remove corresponding request if exists
    else:  # both
        cp.setdefault("resources", {}).setdefault("requests", {})[rn] = sv
        cp.setdefault("resources", {}).setdefault("limits", {})[rn] = sv


class Condition:
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

    def __init__(self, name: str, fn: Callable, *args, **kwargs) -> None:
        if not callable(fn):
            raise ValueError('The Condition function must be callable')

        self.name = name
        self.fn = fn
        self.args = args
        self.kwargs = kwargs

        # last check holds the state of the last check.
        self.last_check = False

    def __str__(self) -> str:
        return f'<Condition (name: {self.name}, met: {self.last_check})>'

    def __repr__(self) -> str:
        return self.__str__()

    async def check(self) -> bool:
        """Check that the condition was met.

        Returns:
            True if the condition was met; False otherwise.
        """
        self.last_check = bool(await self.fn(*self.args, **self.kwargs))
        return self.last_check


async def wait_for_condition(
        condition: Condition,
        timeout: int = None,
        interval: Union[int, float] = 1,
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
    default_logger.info(f'waiting for condition: {condition}')

    # define the maximum time to wait. once this is met, we should
    # stop waiting.
    max_time = None
    if timeout is not None:
        max_time = time.time() + timeout

    # start the wait block
    start = time.time()
    while True:
        if max_time and time.time() >= max_time:
            raise TimeoutError(
                f'timed out ({timeout}s) while waiting for condition {condition}'
            )

        # check if the condition is met and break out if it is
        try:
            if await condition.check():
                break
        except client.exceptions.ApiException as e:
            default_logger.warning(f'got api exception while waiting: {e}')
            if fail_on_api_error:
                raise

        # if the condition is not met, sleep for the interval
        # to re-check later
        await asyncio.sleep(interval)

    end = time.time()
    default_logger.info(f'wait completed (total={end-start:.2f}s) {condition}')


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
    def metadata(self) -> client.V1ObjectMeta:
        ...


class KubernetesModel(abc.ABC):
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
    '''The underlying Kubernetes API object. Subclasses must update
    the type hint to reflect the type that they are wrapping.
    '''

    api_clients: ClassVar[Dict[str, Type]]
    '''A mapping of all the supported api clients for the API
    object type. Various resources can have multiple versions,
    e.g. "apps/v1", "apps/v1beta1", etc. The preferred version
    for each resource type should be defined under the "preferred"
    key. The preferred API client will be used when the apiVersion
    is not specified for the resource.
    '''

    def __init__(self, obj, logger: loguru.Logger = default_logger, **kwargs) -> None:
        self.obj = obj
        self._logger = logger
    
    def __str__(self) -> str:
        return str(self.obj)

    def __repr__(self) -> str:
        return self.__str__()

    @classmethod
    def obj_type(cls) -> Type:
        """The type of the underlying Kubernetes API object."""
        return get_type_hints(cls)["obj"]

    @property
    def logger(self) -> loguru.Logger:
        """A logger instance for outputting operational messages."""
        return self._logger

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

    @asynccontextmanager
    async def api_client(self) -> Generator[Any]:
        """The API client for the Kubernetes object. This is determined
        by the ``apiVersion`` of the object configuration.

        Raises:
            ValueError: The API version is not supported.
        """
        c = self.api_clients.get(self.api_version)
        # If we didn't find the client in the api_clients dict, use the
        # preferred version.
        if c is None:
            self.logger.warning(
                f'unknown version ({self.api_version}), falling back to preferred version'
            )
            c = self.api_clients.get('preferred')
            if c is None:
                raise ValueError(
                    'unknown version specified and no preferred version '
                    f'defined for resource ({self.api_version})'
                )
        # If we did find it, initialize that client version.
        async with ApiClient() as api:
            yield c(api)
    
    @classmethod
    @asynccontextmanager
    async def preferred_client(cls) -> Generator[Any]:
        """The preferred API client type for the Kubernetes object. This is defined in the
        ``api_clients`` class member dict for each object.

        Raises:
             ValueError: No preferred client is defined for the object.
        """
        c = cls.api_clients.get('preferred')
        if c is None:
            raise ValueError(
                f'no preferred api client defined for object {cls.__name__}',
            )
        async with ApiClient() as api:
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
                via the kubetest client.
        """
    
    @abc.abstractmethod
    async def patch(self) -> None:
        """Partially update the underlying Kubernetes resource in the cluster.
        """

    @abc.abstractmethod
    async def delete(self, options: client.V1DeleteOptions) -> client.V1Status:
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
    def is_ready(self) -> bool:
        """Check if the resource is in the ready state.

        It is up to the wrapper subclass to define what "ready" means for
        that particular resource.

        Returns:
            True if in the ready state; False otherwise.
        """
    
    # TODO: Add Duration support
    async def wait_until_ready(
            self,
            timeout: int = None,
            interval: Union[int, float] = 1,
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
            'api object ready',
            self.is_ready,
        )

        await wait_for_condition(
            condition=ready_condition,
            timeout=timeout,
            interval=interval,
            fail_on_api_error=fail_on_api_error,
        )
    
    # TODO: Add Duration support
    async def wait_until_deleted(self, timeout: int = None, interval: Union[int, float] = 1) -> None:
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
            except client.exceptions.ApiException as e:
                # If we can no longer find the deployment, it is deleted.
                # If we get any other exception, raise it.
                if e.status == 404 and e.reason == 'Not Found':
                    return True
                else:
                    self.logger.error('error refreshing object state')
                    raise e
            else:
                # The object was still found, so it has not been deleted
                return False

        delete_condition = Condition(
            'api object deleted',
            deleted_fn
        )

        await wait_for_condition(
            condition=delete_condition,
            timeout=timeout,
            interval=interval,
        )


class Namespace(KubernetesModel):
    """Kubetest wrapper around a Kubernetes `Namespace`_ API Object.

    The actual ``kubernetes.client.V1Namespace`` instance that this
    wraps can be accessed via the ``obj`` instance member.

    This wrapper provides some convenient functionality around the
    API Object and provides some state management for the `Namespace`_.

    .. _Namespace:
        https://kubernetes.io/docs/reference/generated/kubernetes-api/v1.18/#namespace-v1-core
    """

    obj: client.V1Namespace
    api_clients: ClassVar[Dict[str, Type]] = {
        'preferred': client.CoreV1Api,
        'v1': client.CoreV1Api,
    }

    @classmethod
    def new(cls, name: str) -> "Namespace":
        """Create a new Namespace with object backing.

        Args:
            name: The name of the new Namespace.

        Returns:
            A new Namespace instance.
        """
        return cls(obj=client.V1Namespace(
            api_version='v1',
            metadata=client.V1ObjectMeta(
                name=name
            )
        ))
    
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
        self.logger.debug(f'namespace: {self.obj}')

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

    async def delete(self, options: client.V1DeleteOptions = None) -> client.V1Status:
        """Delete the Namespace.

        Args:
             options: Options for Namespace deletion.

        Returns:
            The status of the delete operation.
        """
        if options is None:
            options = client.V1DeleteOptions()

        self.logger.info(f'deleting namespace "{self.name}"')
        self.logger.debug(f'delete options: {options}')
        self.logger.debug(f'namespace: {self.obj}')

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

        return status.phase.lower() == 'active'

class Container:
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

    def __init__(self, api_object, pod) -> None:
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

    def get_restart_count(self) -> int:
        """Get the number of times the Container has been restarted.

        Returns:
            The number of times the Container has been restarted.
        """
        container_name = self.obj.name
        pod_status = self.pod.status()

        # If there are no container status, the container hasn't started
        # yet, so there cannot be any restarts.
        if pod_status.container_statuses is None:
            return 0

        for status in pod_status.container_statuses:
            if status.name == container_name:
                return status.restart_count

        raise RuntimeError(
            f'Unable to determine container status for {container_name}'
        )
    
    @property
    def resources(self) -> client.V1ResourceRequirements:
        """
        Return the resource requirements for the Container.

        Returns:
            The Container resource requirements.
        """
        return self.obj.resources
        
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

    obj: client.V1Pod

    api_clients: ClassVar[Dict[str, Type]] = {
        'preferred': client.CoreV1Api,
        'v1': client.CoreV1Api,
    }

    @classmethod
    async def read(cls, name: str, namespace: str) -> "Pod":
        """Read the Pod from the cluster under the given namespace.

        Args:
            name: The name of the Pod to read.
            namespace: The namespace to read the POd from.
        """
        async with cls.preferred_client() as api_client:
            obj = await api_client.read_namespaced_pod(name, namespace)
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

        self.logger.info(f'creating pod "{self.name}" in namespace "{self.namespace}"')
        self.logger.trace(f'pod: {self.obj}')

        async with self.preferred_client() as api_client:
            self.obj = await api_client.create_namespaced_pod(
            namespace=namespace,
            body=self.obj,
        )
    
    async def patch(self) -> None:
        """
        TODO: Add docs....
        """
        self.logger.info(f'patching pod "{self.name}"')
        self.logger.trace(f'pod: {self.obj}')
        async with self.api_client() as api_client:
            await api_client.patch_namespaced_pod(
                name=self.name,
                namespace=self.namespace,
                body=self.obj,
            )

    async def delete(self, options: client.V1DeleteOptions = None) -> client.V1Status:
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
            options = client.V1DeleteOptions()

        self.logger.info(f'deleting pod "{self.name}"')
        self.logger.debug(f'delete options: {options}')
        self.logger.trace(f'pod: {self.obj}')

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
        await self.refresh()

        # if there is no status, the pod is definitely not ready
        status = self.obj.status
        if status is None:
            return False

        # check the pod phase to make sure it is running. a pod in
        # the 'failed' or 'success' state will no longer be running,
        # so we only care if the pod is in the 'running' state.
        phase = status.phase
        if phase.lower() != 'running':
            return False

        # TODO: Check for Ready and ContainersReady (Check if below logic matches)
        # 'Returns bool indicating pod readiness'
        # cont_stats = pod.status.container_statuses
        # conts_ready = cont_stats and len(cont_stats) >= len(pod.spec.containers) and all([cs.ready for cs in pod.status.container_statuses])
        # rdy_conditions = [] if not pod.status.conditions else [con for con in pod.status.conditions if con.type in ['Ready', 'ContainersReady']]
        # pod_ready = len(rdy_conditions) > 1 and all([con.status == 'True' for con in rdy_conditions])
        # return conts_ready and pod_ready
        for cond in status.conditions:
            # we only care about the condition type 'ready'
            if cond.type.lower() != 'ready':
                continue

            # check that the readiness condition is True
            return cond.status.lower() == 'true'

        # Catchall
        return False

    async def get_status(self) -> client.V1PodStatus:
        """Get the status of the Pod.

        Returns:
            The status of the Pod.
        """
        # first, refresh the pod state to ensure latest status
        await self.refresh()

        # return the status of the pod
        return cast(client.V1PodStatus, self.obj.status)
    

    @property
    def containers(self) -> List[Container]:
        """
        Return a list of Container objects from the underlying pod template spec.
        """
        return list(map(lambda c: Container(c, None), self.obj.spec.containers))

    def get_container(self, name: str) -> Container:
        """
        Return the container with the given name.
        """
        return next(filter(lambda c: c.name == name, self.containers))


    async def get_containers(self) -> List[Container]:
        """Get the Pod's containers.

        Returns:
            A list of containers that belong to the Pod.
        """
        self.logger.info(f'getting containers for pod "{self.name}"')
        await self.refresh()

        return [Container(c, self) for c in self.obj.spec.containers]

    # TODO: Rename `find_container` ??
    def get_container(self, name: str) -> Union[Container, None]:
        """Get a container in the Pod by name.

        Args:
            name (str): The name of the Container.

        Returns:
            Container: The Pod's Container with the matching name. If
            no container with the given name is found, ``None`` is returned.
        """
        return next(filter(lambda c: c.name == name, self.containers))

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


class Deployment(KubernetesModel):
    """Kubetest wrapper around a Kubernetes `Deployment`_ API Object.

    The actual ``kubernetes.client.V1Deployment`` instance that this
    wraps can be accessed via the ``obj`` instance member.

    This wrapper provides some convenient functionality around the
    API Object and provides some state management for the `Deployment`_.

    .. _Deployment:
        https://kubernetes.io/docs/reference/generated/kubernetes-api/v1.18/#deployment-v1-apps
    """

    obj: client.V1Deployment
    api_clients: ClassVar[Dict[str, Type]] = {
        'preferred': client.AppsV1Api,
        'apps/v1': client.AppsV1Api,
        'apps/v1beta1': client.AppsV1beta1Api,
        'apps/v1beta2': client.AppsV1beta2Api,
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

        self.logger.info(f'creating deployment "{self.name}" in namespace "{self.namespace}"')
        self.logger.debug(f'deployment: {self.obj}')

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
        """Update the changed attributes of the Deployment.
        """
        async with self.api_client() as api_client:
            self.obj = await api_client.patch_namespaced_deployment(
                name=self.name,
                namespace=self.namespace,
                body=self.obj,
            )

    async def delete(self, options: client.V1DeleteOptions = None) -> client.V1Status:
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
            options = client.V1DeleteOptions()

        self.logger.info(f'deleting deployment "{self.name}"')
        self.logger.debug(f'delete options: {options}')
        self.logger.trace(f'deployment: {self.obj}')

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
    
    async def get_status(self) -> client.V1DeploymentStatus:
        """Get the status of the Deployment.

        Returns:
            The status of the Deployment.
        """
        self.logger.info(f'checking status of deployment "{self.name}"')
        # first, refresh the deployment state to ensure the latest status
        await self.refresh()

        # return the status from the deployment
        return cast(client.V1DeploymentStatus, self.obj.status)

    async def get_pods(self) -> List[Pod]:
        """Get the pods for the Deployment.

        Returns:
            A list of pods that belong to the deployment.
        """
        self.logger.info(f'getting pods for deployment "{self.name}"')
        
        async with Pod.preferred_client() as api_client:
            label_selector = self.obj.spec.selector.match_labels
            pod_list: client.V1PodList = await api_client.list_namespaced_pod(
                namespace=self.namespace,
                label_selector=selector_string(label_selector)
            )

        pods = [Pod(p) for p in pod_list.items]
        return pods
    
    @property
    def status(self) -> client.V1DeploymentStatus:
        """Return the status of the Deployment.

        Returns:
            The status of the Deployment.
        """
        return cast(client.V1DeploymentStatus, self.obj.status)

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
    
    # TODO: Determine if we want this...
    def is_complete(self, target_generation: int) -> bool:
        # Kubernetes marks a Deployment as complete when it has the following characteristics:

        # All of the replicas associated with the Deployment have been updated to the latest version you've specified, meaning any updates you've requested have been completed.
        # All of the replicas associated with the Deployment are available.
        # No old replicas for the Deployment are running.
        ...

    @property
    def containers(self) -> List[Container]:
        """
        Return a list of Container objects from the underlying pod template spec.
        """
        return list(map(lambda c: Container(c, None), self.obj.spec.template.spec.containers))

    def get_container(self, name: str) -> Container:
        """
        Return the container with the given name.
        """
        return next(filter(lambda c: c.name == name, self.containers))

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
        return selector_string(self.obj.spec.selector.match_labels)
    
    ##
    # Canary support

    @property
    def canary_pod_name(self) -> str:
        """
        Return the name of canary Pod for this Deployment.
        """
        return f"{self.name}-canary"

    async def get_canary_pod(self) -> Pod:
        """
        Retrieve the canary Pod for this Deployment (if any).

        Will raise a Kubernetes API exception if not found.
        """
        return await Pod.read(self.canary_pod_name, self.namespace)
    

    async def delete_canary_pod(self, *, raise_if_not_found: bool = True, timeout: Numeric = 20) -> Optional[Pod]:
        """
        Delete the canary Pod.
        """
        try:
            canary = await self.get_canary_pod()
            self.logger.warning(f"Deleting canary Pod '{canary.name}' in namespace '{canary.namespace}'...")
            await canary.delete()
            await canary.wait_until_deleted(timeout=timeout)
            self.logger.info(f"Deleted canary Pod '{canary.name}' in namespace '{canary.namespace}'.")
            return canary
        except client.exceptions.ApiException as e:
            self.logger.debug(f"failed loading canary pod: {e}")
            if e.status != 404 or e.reason != 'Not Found' and raise_if_not_found:
                raise
        
        return None


    async def ensure_canary_pod(self, *, timeout: Numeric = 20) -> Pod:
        """
        Ensures that a canary Pod exists by deleting and recreating an existing Pod or creating one from scratch.

        TODO: docs...
        """        
        canary_pod_name = self.canary_pod_name
        namespace = self.namespace
        self.logger.debug(f"ensuring existence of canary pod '{canary_pod_name}' in namespace '{namespace}'")
        
        # Delete any pre-existing canary debris
        await self.delete_canary_pod(raise_if_not_found=False, timeout=timeout)
        
        # Setup the canary Pod -- our settings are updated on the underlying PodSpec template
        self.logger.trace(f"building new canary")
        pod_obj = client.V1Pod(metadata=self.obj.spec.template.metadata, spec=self.obj.spec.template.spec)
        pod_obj.metadata.name = canary_pod_name
        pod_obj.metadata.annotations['opsani.com/opsani_tuning_for'] = self.name
        pod_obj.metadata.labels['opsani_role'] = 'tuning'
        canary_pod = Pod(obj=pod_obj)
        canary_pod.namespace = namespace
        self.logger.trace(f"initialized new canary: {canary_pod}")
        
        # If the servo is running inside Kubernetes, register self as the controller for the Pod and ReplicaSet
        SERVO_POD_NAME = os.environ.get('POD_NAME')
        SERVO_POD_NAMESPACE = os.environ.get('POD_NAMESPACE')
        if SERVO_POD_NAME is not None and SERVO_POD_NAMESPACE is not None:
            self.logger.debug(f"running within Kubernetes, registering as Pod controller... (pod={SERVO_POD_NAME}, namespace={SERVO_POD_NAMESPACE})")
            servo_pod = await Pod.read(SERVO_POD_NAME, SERVO_POD_NAMESPACE)
            pod_controller = next(iter(ow for ow in servo_pod.obj.metadata.owner_references if ow.controller))

            # # TODO: Create a ReplicaSet class...
            # async with ApiClient() as api:
            #     api_client = client.AppsV1Api(api)

            #     servo_rs = await api_client.read_namespaced_replica_set(name=pod_controller.name, namespace=SERVO_POD_NAMESPACE) # still ephemeral
            #     rs_controller = next(iter(ow for ow in servo_rs.metadata.owner_references if ow.controller))
            #     # deployment info persists thru updates. only remove servo pod if deployment is deleted
            #     servo_dep: client.V1Deployment = await api_client.read_namespaced_deployment(name=rs_controller.name, namespace=SERVO_POD_NAMESPACE)

            canary_pod.obj.metadata.owner_references = [ client.V1OwnerReference(
                api_version=self.api_version,
                block_owner_deletion=False, # TODO will setting this to true cause issues or assist in waiting for cleanup?
                controller=True, # Ensures the pod will not be adopted by another controller
                kind='Deployment',
                name=self.obj.metadata.name,
                uid=self.obj.metadata.uid
            ) ]

        # Create the Pod and wait for it to get ready
        self.logger.info(f"Creating canary Pod '{canary_pod_name}' in namespace '{namespace}'")
        # self.logger.debug(canary_pod)
        await canary_pod.create()

        self.logger.info(f"Created canary Pod '{canary_pod_name}' in namespace '{namespace}', waiting for it to become ready...")
        await canary_pod.wait_until_ready(timeout=timeout)

        # TODO: Add settlement time. Check for unexpected changes to version, etc.    

        return canary_pod


class ResourceConstraint(enum.Enum):
    """
    The ResourceConstraint enumeration determines how optimization values are applied
    to the cpu and memory resource requests & limits of a container.
    """
    request = "request"
    limit = "limit"
    both = "both"


class Resource(Setting):
    """
    Resource is a class that models Kubernetes specific Setting objects that are subject
    to request and limit configuration.
    """
    constraint: ResourceConstraint = ResourceConstraint.both


class Millicore(int):
    """
    The Millicore class represents one one-hundreth of a vCPU or hyperthread in Kubernetes.
    """
    @classmethod
    def __get_validators__(cls) -> 'CallableGenerator':
        yield cls.parse
    
    @classmethod
    def parse(cls, v: StrIntFloat) -> 'Millicore':
        """
        Parse an input value into Millicore units.

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
        elif isinstance(v, (int, float)):
            return cls(int(v * 1000))
        else:
            raise ValueError("could not parse millicore value")
            
    def __str__(self) -> str:
        return f'{int(self)}m'
    
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


class CPU(Resource):
    """
    The CPU class models a Kubernetes CPU resource in Millicore units.
    """
    value: Millicore
    # min: Millicore
    # max: Millicore
    # step: Millicore
    # name = "cpu"
    # type = SettingType.RANGE

    def opsani_dict(self) -> dict:
        o_dict = super().opsani_dict()
        o_dict["cpu"]["value"] = float(self.value)
        return o_dict


# Gibibyte is the base unit of Kubernetes memory
GiB = 1024 * 1024 * 1024


class ShortByteSize(ByteSize):
    """Kubernetes omits the 'B' suffix for some reason"""
    @classmethod
    def validate(cls, v: StrIntFloat) -> 'ShortByteSize':
        if isinstance(v, str):            
            try:
                return super().validate(v)
            except:
                # Append the byte suffix and retry parsing
                return super().validate(v + "b")
        return super().validate(v)


class Memory(Resource):
    """
    The Memory class models a Kubernetes Memory resource.
    """
    value: ShortByteSize
    # min: ShortByteSize
    # max: ShortByteSize
    # step: ShortByteSize
    # name = "memory"
    # type = SettingType.RANGE

    @property
    def gibibytes(self) -> float:
        return float(self.value) / GiB

    def opsani_dict(self) -> dict:
        o_dict = super().opsani_dict()
        o_dict["memory"]["value"] = self.gibibytes
        return o_dict


class Replicas(Setting):
    """
    The Replicas class models a Kubernetes setting that specifies the number of
    desired Pods running in a Deployment.
    """
    value: int
    # name = "replicas"
    # type = SettingType.RANGE


# TODO: The Adjustment needs to marshal value appropriately on ingress
def _qualify(value, unit):
    if unit == "memory":
        return f"{value}Gi"
    elif unit == "cpu":
        return str(Millicore.parse(value))
    elif unit == "replicas":
        return int(float(value))
    return value


class Adjustable(BaseModel):
    name: str
    deployment: Deployment
    container: Container
    canary: bool = False

    # Resources
    # TODO: This crap moves back to the DeploymentComponent
    cpu: CPU
    memory: Memory
    replicas: Replicas
    env: Dict[str, str] = {}
    
    def adjust(self, adjustment: Adjustment, control: Control = Control()) -> None:
        """
        Adjust a setting on the underlying Deployment/Pod or Container.
        """

        name = adjustment.setting_name
        value = _qualify(adjustment.value, name)
        if name in ("cpu", "memory"):
            resource = getattr(self, name)            
            if resource.constraint in (ResourceConstraint.request, ResourceConstraint.both):
                self.container.resources.requests[name] = value
            if resource.constraint in (ResourceConstraint.limit, ResourceConstraint.both):
                self.container.resources.limits[name] = value

        elif adjustment.setting_name == "replicas":
            if self.canary and self.deployment.replicas != value:
                self.logger.warning(f"rejected attempt to adjust replicas in canary mode: pin or remove replicas setting to avoid this warning")
            else:
                self.deployment.replicas = value
            
        else:
            raise RuntimeError(f"failed adjustment of unsupported Kubernetes setting '{adjustment.setting_name}'")
    
    async def apply(self) -> None:
        """
        TODO: add docs...
        """
        if self.canary:            
            await self.deployment.ensure_canary_pod()
        else:
            await self.apply_to_deployment()

    async def apply_to_deployment(self) -> None:
        """
        Apply changes asynchronously and wait for them to roll out to the cluster.

        Kubernetes deployments orchestrate a number of underlying resources. Awaiting the
        outcome of a deployment change requires observation of the `resource_version` which
        indicates if a given patch actually changed the resource, the `observed_generation`
        which is a value managed by the deployments controller and indicates the effective 
        version of the deployment exclusive of insignificant changes that do not affect runtime
        (such as label updates), and the `conditions` of the deployment status which reflect
        state at a particular point in time. How these elements change during a rollout is 
        dependent on the deployment strategy in effect and its constraints (max unavailable, 
        surge, etc).

        The logic implemented by this method is as follows:
            - Capture the `resource_version` and `observed_generation`.
            - Patch the underlying Deployment object via the Kubernetes API.
            - Check that `resource_version` has been incremented or return because nothing has changed.
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
        
        # Resource version lets us track any change. Observed generation only increments
        # when the deployment controller sees a significant change that requires rollout
        resource_version = self.deployment.resource_version
        observed_generation = self.deployment.status.observed_generation
        desired_replicas = self.deployment.replicas

        # Patch the Deployment via the Kubernetes API
        await self.deployment.patch()

        # Return fast if nothing was changed
        if self.deployment.resource_version == resource_version:
            self.logger.info(f"adjustments applied to Deployment '{self.deployment.name}' made no changes, continuing")
            return
                
        # Create a Kubernetes watch against the deployment under optimization to track changes
        self.logger.info(f"Using label_selector={self.deployment.label_selector}, resource_version={resource_version}")
        async with client.ApiClient() as api:
            v1 = client.AppsV1Api(api)
            async with watch.Watch().stream(
                v1.list_namespaced_deployment,
                self.deployment.namespace,
                label_selector=self.deployment.label_selector,
                # resource_version=resource_version, # FIXME: The resource version might be expired and fail the watch. Decide if we care
            ) as stream:
                async for event in stream:
                    # NOTE: Event types are ADDED, DELETED, MODIFIED, ERROR
                    event_type, deployment = event['type'], event['object']
                    status: client.V1DeploymentStatus = deployment.status
                    
                    self.logger.debug(f"deployment watch yielded event: {event_type} {deployment.kind} {deployment.metadata.name} in {deployment.metadata.namespace}: {status}")

                    if event_type == 'ERROR':
                        stream.stop()
                        raise RuntimeError(str(deployment))                    

                    # Check that the conditions aren't reporting a failure
                    self._check_conditions(status.conditions)

                    # Early events in the watch may be against previous generation
                    if status.observed_generation == observed_generation:
                        self.logger.debug("observed generation has not changed, continuing watch")
                        continue
                    
                    # Check the replica counts. Once available, updated, and ready match
                    # our expected count and the unavailable count is zero we are rolled out
                    if status.unavailable_replicas:
                        self.logger.debug("found unavailable replicas, continuing watch", status.unavailable_replicas)
                        continue
                    
                    replica_counts = [status.replicas, status.available_replicas, status.ready_replicas, status.updated_replicas]
                    if replica_counts.count(desired_replicas) == len(replica_counts):
                        # We are done: all the counts match. Stop the watch and return
                        self.logger.info("adjustment applied successfully", status)
                        stream.stop()
                        return
    
    def _check_conditions(self, conditions: List[client.V1DeploymentCondition]):
        for condition in conditions:
            if condition.type == "Available":                            
                if condition.status == "True":
                    # If we hit on this and have not raised yet we are good to go
                    break                        
                elif condition.status in ("False", "Unknown"):
                    # Condition has not yet been met, log status and continue monitoring
                    self.logger.debug(f"Condition({condition.type}).status == '{condition.status}' ({condition.reason}): {condition.message}")
                else:
                    raise RuntimeError(f"encountered unexpected Condition status '{condition.status}'")

            elif condition.type == "ReplicaFailure":
                # TODO: Create a specific error type
                raise RuntimeError("ReplicaFailure: message='{condition.status.message}', reason='{condition.status.reason}'")

            elif condition.type == "Progressing":
                if condition.status in ("True", "Unknown"):
                    # Still working
                    self.logger.debug("Deployment update is progressing", condition)
                    break
                if condition.status == "False":
                    # TODO: Create specific error type
                    raise RuntimeError("ProgressionFailure: message='{condition.status.message}', reason='{condition.status.reason}'")
                else:
                    raise AssertionError(f"unknown deployment status condition: {condition.status}")
    
    def to_component(self) -> Component:
        return Component(
            name=self.name,
            settings=[
                self.cpu,
                self.memory,
                self.replicas
            ]
        )
    
    @property
    def logger(self) -> loguru.Logger:
        return default_logger

    def __hash__(self):
        return hash((self.name, id(self),))    

    class Config:
        arbitrary_types_allowed = True


class KubernetesState(BaseModel):
    """
    Models the state of resources under optimization in a Kubernetes cluster.
    """
    namespace: Namespace
    adjustables: List[Adjustable]
    runtime_id: str
    spec_id: str
    version_id: str

    @classmethod
    async def read(cls, config: KubernetesConfiguration) -> 'KubernetesState':
        """
        Read the state of all components under optimization from the cluster and return an object representation.
        """
        await config.load_kubeconfig()

        namespace = await Namespace.read(config.namespace)
        adjustables = []
        images = {}
        runtime_ids = {}
        pod_tmpl_specs = {}

        for component in config.deployments:
            deployment_name = component.name
            container_name = None
            if "/" in deployment_name:
                deployment_name, container_name = component.name.split("/")

            deployment = await Deployment.read(deployment_name, namespace.name)
            if container_name:
                container = deployment.get_container(container_name)
            else:
                container = deployment.containers[0]
            
            pod_tmpl_specs[deployment_name] = deployment.obj.spec.template.spec
            images[deployment_name] = container.image

            # TODO: Needs to respect the constraint (limit vs. request)... (maybe... container.get_resource("cpu", constraint), set_resource("cpu", value, constraint))
            # TODO: These just become direct property assignments once DeploymentComponent is live (!!! maybe not -- need to support pods)
            cpu_setting = next(filter(lambda c: c.name == "cpu", component.settings))
            cpu_setting.value = container.resources.limits["cpu"]
            mem_setting = next(filter(lambda c: c.name == "memory", component.settings))
            mem_setting.value = container.resources.limits["memory"]
            replicas_setting = next(filter(lambda c: c.name == "replicas", component.settings))
            replicas_setting.value = deployment.replicas

            adjustables.append(
                Adjustable(
                    name=component.name,
                    canary=component.canary,
                    deployment=deployment,
                    container=container,
                    cpu=CPU.parse_obj(cpu_setting.dict()),
                    memory=Memory.parse_obj(mem_setting.dict()),
                    replicas=Replicas.parse_obj(replicas_setting.dict()),
                )
            )

            pods = await deployment.get_pods()
            runtime_ids[deployment_name] = [pod.uid for pod in pods]
        
        # Compute checksums for change detection
        spec_id = get_hash([pod_tmpl_specs[k] for k in sorted(pod_tmpl_specs.keys())])
        runtime_id = get_hash(runtime_ids)
        version_id = get_hash([images[k] for k in sorted(images.keys())])

        return KubernetesState(
            namespace=namespace,
            adjustables=adjustables,
            spec_id=spec_id,
            runtime_id=runtime_id,
            version_id=version_id,
        )
    
    def to_description(self) -> Description:
        """
        Return a representation of the current state as a Description object.

        Description objects are used to report state to the Opsani API in order
        to synchronize with the Optimizer service.

        Returns:
            A Description of the current state.
        """
        return Description(
            components=list(map(lambda a: a.to_component(), self.adjustables))
        )
    
    def get_adjustable(self, name: str) -> Adjustable:
        """
        Find and return an adjustable by name.
        """
        return next(filter(lambda a: a.name == name, self.adjustables))

    async def apply(self, adjustments: List[Adjustment]) -> None:
        """
        ...
        """
        # Exit early if there is nothing to do
        if not adjustments:
            return
        
        summary = f"[{', '.join(list(map(str, adjustments)))}]"
        self.logger.info(f"Applying {len(adjustments)} Kubernetes adjustments: {summary}")
        
        # Adjust settings on the local data model
        for adjustment in adjustments:
            adjustable = self.get_adjustable(adjustment.component_name)
            self.logger.trace(f"adjusting {adjustment.component_name}: {adjustment}")
            adjustable.adjust(adjustment)
        
        # Apply the changes to Kubernetes and wait for the results
        if self.adjustables:
            self.logger.debug(f"apply adjustments to {len(self.adjustables)} adjustables")
            await asyncio.wait_for(
                asyncio.gather(
                    *list(map(lambda a: a.apply(), self.adjustables))
                ),
                timeout=30.0
            )
        else:
            self.logger.warning(f"failed to apply adjustments: no adjustables")

        # TODO: Run sanity checks to look for out of band changes
        # TODO: Figure out how to do progress...
        # TODO: Rollout undo support

        return
    
    @property
    def logger(self) -> loguru.Logger:
        return default_logger

        # wait for update to complete (and print progress)
            # timeout default is set to be slightly higher than the default K8s timeout (so we let k8s detect progress stall first)
        #     try:
        #         await self.wait_for_update(
        #             namespace,
        #             n,
        #             patch_r["metadata"]["generation"],
        #             patched_count,
        #             len(patchlst),
        #             cfg.get("timeout", 630),
        #         )
        #     except AdjustError as e:
        #         if e.reason != "start-failed":  # not undo-able
        #             raise
        #         onfail = cfg.get(
        #             "on_fail", "keep"
        #         )  # valid values: keep, destroy, rollback (destroy == scale-to-zero, not supported)
        #         if onfail == "rollback":
        #             try:
        #                 # TODO: This has to be ported
        #                 subprocess.call(
        #                     kubectl(appname, "rollout", "undo", DEPLOYMENT + "/" + n)
        #                 )
        #                 print("UNDONE", file=sys.stderr)
        #             except subprocess.CalledProcessError:
        #                 # progress msg with warning TODO
        #                 print("undo for {} failed: {}".format(n, e), file=sys.stderr)
        #         raise
        #     patched_count = patched_count + 1

        # # spec_id and version_id should be tested without settlement_time, too - TODO

        # # post-adjust settlement, if enabled
        # testdata0 = await self.raw_query(namespace, app_state.components)
        # settlement_time = cfg.get("settlement", 0)
        # mon0 = testdata0.monitoring

        # if "ref_version_id" in mon0 and mon0["version_id"] != mon0["ref_version_id"]:
        #     raise AdjustError(
        #         "application version does not match reference version",
        #         status="aborted",
        #         reason="version-mismatch",
        #     )
        
        # TODO: What are these status reasons??
        # # aborted status reasons that aren't supported: ref-app-inconsistent, ref-app-unavailable

        # # TODO: This response needs to be modeled
        # if not settlement_time:
        #     return {"monitoring": mon0, "status": "ok", "reason": "success"}

        # # wait and watch the app, checking for changes
        # # TODO: Port this...
        # w = Waiter(
        #     settlement_time, delay=min(settlement_time, 30)
        # )  # NOTE: delay between tests may be made longer than the delay between progress reports
        # while w.wait():
        #     testdata = raw_query(namespace, app_state.components)
        #     mon = testdata.monitoring
        #     # compare to initial mon data set
        #     if mon["runtime_id"] != mon0["runtime_id"]:  # restart detected
        #         # TODO: allow limited number of restarts? (and how to distinguish from rejected/unstable??)
        #         raise AdjustError(
        #             "component(s) restart detected",
        #             status="transient-failure",
        #             reason="app-restart",
        #         )
        #     # TODO: what to do with version change?
        #     #        if mon["version_id"] != mon0["version_id"]:
        #     #            raise AdjustError("application was modified unexpectedly during settlement", status="transient-failure", reason="app-update")
        #     if mon["spec_id"] != mon0["spec_id"]:
        #         raise AdjustError(
        #             "application configuration was modified unexpectedly during settlement",
        #             status="transient-failure",
        #             reason="app-update",
        #         )
        #     if mon["ref_spec_id"] != mon0["ref_spec_id"]:
        #         raise AdjustError(
        #             "reference application configuration was modified unexpectedly during settlement",
        #             status="transient-failure",
        #             reason="ref-app-update",
        #         )
        #     if mon["ref_runtime_count"] != mon0["ref_runtime_count"]:
        #         raise AdjustError("", status="transient-failure", reason="ref-app-scale")

    class Config:
        arbitrary_types_allowed = True


# NOTE: This class is not yet live
class DeploymentComponent(Component):
    """
    The DeploymentComponent class models an optimizable Kubernetes Deployment.
    """
    namespace: str = "default"
    container: str = "main" # TODO: Is this a Kubernetes naming or Opsani?
    canary: bool = False
    memory: Memory
    replicas: Replicas
    settings: List[Setting]



class KubernetesConfiguration(BaseConfiguration):
    kubeconfig: Optional[FilePath] = Field(
        description="Path to the kubeconfig file. If `None`, use the default from the environment.",
    )
    context: Optional[str] = Field(
        description="Name of the kubeconfig context to use."
    )
    namespace: str = Field(
        "default",
        description="Kubernetes namespace where the target deployments are running.",
    )
    deployments: List[Component] = Field(
        description="The deployments and adjustable settings to optimize.",
    )
    settlement_duration: Duration = Field(
        0,
        description="Duration to observe the application after an adjust to ensure the deployment is stable."
    )

    @classmethod
    def generate(cls, **kwargs) -> "KubernetesConfiguration":
        return cls(
            namespace="default",
            description="Update the namespace, deployment, etc. to match your Kubernetes cluster",
            deployments=[
                Component(
                    name="app",
                    settings=[
                        Setting(
                            name="cpu",
                            type="range",
                            min="0.125",
                            max="4.0",
                            step="0.125",
                        )
                    ]
                )
            ],
            **kwargs
        )
    
    # TODO: This might not be the right home for this method...
    async def load_kubeconfig(self) -> None:
        """
        Asynchronously load the Kubernetes configuration
        """
        config_file = Path(self.kubeconfig or KUBE_CONFIG_DEFAULT_LOCATION).expanduser()
        if config_file.exists():
            await kubernetes_asyncio_config.load_kube_config(
                config_file=str(config_file),
                context=self.context,
            )
        elif os.getenv('KUBERNETES_SERVICE_HOST'):
            kubernetes_asyncio_config.load_incluster_config()
        else:
            raise RuntimeError(f"unable to configure Kubernetes client: no kubeconfig file nor in-cluser environment variables found")


class KubernetesChecks(BaseChecks):
    config: KubernetesConfiguration

    async def check_connectivity(self) -> Check:
        try:
            await KubernetesState.read(self.config)
        except Exception as e:
            return Check(
                name="Connect to Kubernetes", success=False, comment=str(e)
            )

        return Check(name="Connect to Kubernetes", success=True, comment="")
    
    # TODO: Verify the connectivity & permissions
    # TODO: Check the Deployments exist
    # TODO: Check that the Deployment is available
    # TODO: What other unhealthy conditions?

    # def check_access(self) -> Check:
    #     ...
    
    # def check_deployment_exists(self) -> Check:
    #     ...
    
    # def check_deployment_is_available(self) -> Check:
    #     ...


@connector.metadata(
    description="Kubernetes adjust connector",
    version="1.5.0",
    homepage="https://github.com/opsani/kubernetes-connector",
    license=License.APACHE2,
    maturity=Maturity.EXPERIMENTAL,
)
class KubernetesConnector(BaseConnector):
    config: KubernetesConfiguration

    @on_event()
    async def startup(self) -> None:
        # Ensure we are ready to talk to Kubernetes API
        await self.config.load_kubeconfig()

    @on_event()
    async def describe(self) -> Description:        
        state = await KubernetesState.read(self.config)
        return state.to_description()

    @on_event()
    def components(self) -> List[Component]:
        return self.config.deployments

    @on_event()
    async def adjust(self, adjustments: List[Adjustment], control: Control = Control()) -> None:
        # TODO: Handle this adjust_on stuff (Do we even need this???)
        # adjust_on = desc.get("adjust_on", False)

        # if adjust_on:
        #     try:
        #         should_adjust = eval(adjust_on, {"__builtins__": None}, {"data": data})
        #     except:
        #         should_adjust = False
        #     if not should_adjust:
        #         return {"status": "ok", "reason": "Skipped due to 'adjust_on' condition"}

        state = await KubernetesState.read(self.config)
        await state.apply(adjustments)

        # TODO: Move this into event declaration??
        settlement_duration = self.config.settlement_duration
        if settlement_duration:
            self.logger.info(f"Settlement duration of {settlement_duration} requested, sleeping...")            
            progress = DurationProgress(settlement_duration)
            progress_logger = lambda p: self.logger.info("allowing application to settle", progress=p.progress)
            await progress.watch(progress_logger)
            self.logger.info(f"Settlement duration of {settlement_duration} expired, resuming.")

    @on_event()
    async def check(self) -> List[Check]:
        return await KubernetesChecks.run(self.config)


def selector_string(selectors: Mapping[str, str]) -> str:
    """Create a selector string from the given dictionary of selectors.

    Args:
        selectors: The selectors to stringify.

    Returns:
        The selector string for the given dictionary.
    """
    return ','.join([f'{k}={v}' for k, v in selectors.items()])


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
        kwargs['field_selector'] = selector_string(fields)
    if labels is not None:
        kwargs['label_selector'] = selector_string(labels)

    return kwargs


def __todo_encoders():
    # TODO: This is broken atm. Bake ENV support into core
    # TODO: This has dynamic keys to kill off.
    env = component.env
    if env:
        for en, ev in env.items():
            assert isinstance(
                ev, dict
            ), 'Setting "{}" in section "env" of a config file is not a dictionary.'
            if "encoder" in ev:
                for name, setting in describe_encoder(
                    cont_env_dict.get(en),
                    ev["encoder"],
                    exception_context="an environment variable {}" "".format(en),
                ):
                    settings[name] = setting
            if issetting(ev):
                defval = ev.pop("default", None)
                val = cont_env_dict.get(en, defval)
                val = (
                    float(val)
                    if israngesetting(ev) and isinstance(val, (int, str))
                    else val
                )
                assert val is not None, (
                    'Environment variable "{}" does not have a current value defined and '
                    "neither it has a default value specified in a config file. "
                    "Please, set current value for this variable or adjust the "
                    "configuration file to include its default value."
                    "".format(en)
                )
                val = {**ev, "value": val}
                settings[en] = val

            # TODO: Must be added to model...
            # command = comp.get("command")
            # if command:
            #     if command.get("encoder"):
            #         for name, setting in describe_encoder(
            #             cont.get("command", []),
            #             command["encoder"],
            #             exception_context="a command section",
            #         ):
            #             settings[name] = setting
            #         # Remove section "command" from final descriptor
            #     del comp["command"]


             # TODO: Port this
            # command = component.command
            # if command:
            #     if command.get("encoder"):
            #         cont_patch["command"], encoded_settings = encode_encoder(
            #             settings, command["encoder"], expected_type=list
            #         )

            #         # Prevent encoded settings from further processing
            #         for setting in encoded_settings:
            #             del settings[setting]

            # env = component.env
            # if env:
            #     for en, ev in env.items():
            #         if ev.get("encoder"):
            #             val, encoded_settings = encode_encoder(
            #                 settings, ev["encoder"], expected_type=str
            #             )
            #             patch_env = cont_patch.setdefault("env", [])
            #             patch_env.append({"name": en, "value": val})

            #             # Prevent encoded settings from further processing
            #             for setting in encoded_settings:
            #                 del settings[setting]
            #         elif issetting(ev):
            #             patch_env = cont_patch.setdefault("env", [])
            #             patch_env.append({"name": en, "value": str(settings[en]["value"])})
            #             del settings[en]