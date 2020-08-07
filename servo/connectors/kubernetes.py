from __future__ import annotations, print_function

import asyncio
from typing import List, Optional, Dict, Any

from pydantic import BaseModel, ByteSize, Extra, Field, FilePath, validator

from servo import (
    Adjustment,
    BaseConfiguration,
    BaseConnector,
    Check,
    Component,
    Description,
    License,
    Maturity,
    Setting,
    connector,
    on_event,
    get_hash
)
from kubernetes_asyncio import client, config, watch
from kubernetes_asyncio.client.api_client import ApiClient
import abc
import enum
import loguru
from loguru import logger as default_logger
from typing import ClassVar, Generator, Mapping, Protocol, Type, Union, cast, get_type_hints, runtime_checkable
from contextlib import asynccontextmanager


# Gibibyte is the base unit of Kubernetes memory
GiB = 1024 * 1024 * 1024


# TODO: This is some kind of progress watcher... WaitCondition?
def test_dep_generation(dep, g):
    """ check if the deployment status indicates it has been updated to the given generation number"""
    return dep["status"]["observedGeneration"] == g

def test_dep_progress(dep):
    """check if the deployment object 'dep' has reached final successful status
    ('dep' should be the data returned by 'kubectl get deployment' or the equivalent API call, e.g.,
    GET /apis/(....)/namespaces/:ns/deployments/my-deployment-name).
    This tests the conditions[] array and the replica counts and converts the data to a simplified status, as follows:
    - if the deployment appears to be in progress and k8s is still waiting for updates from the controlled objects (replicasets and their pods),
      return a tuple (x, ""), where x is the fraction of the updated instances (0.0 .. 1.0, excluding 1.0).
    - if the deployment has completed, return (1.0, "")
    - if the deployment has stalled or failed, return (x, "(errormsg)"), with an indication of the
      detected failure (NOTE: in k8s, the 'stall' is never final and could be unblocked by change
      of resources or other modifications of the cluster not related to the deployment in question,
      but we assume that the system is operating under stable conditions and there won't be anyone
      or anything that can unblock such a stall)
    """
    dbg_log("test_dep_progress:")
    spec_replicas = dep["spec"]["replicas"]  # this is what we expect as target
    dep_status = dep["status"]
    for co in dep_status["conditions"]:
        dbg_log(
            "... condition type {}, reason {}, status {}, message {}".format(
                co.get("type"), co.get("reason"), co.get("status"), co.get("message")
            )
        )
        if co["type"] == "Progressing":
            if co["status"] == "True" and co["reason"] == "NewReplicaSetAvailable":
                # if the replica set was updated, test the replica counts
                if (
                    dep_status.get("updatedReplicas", None) == spec_replicas
                ):  # update complete, check other counts
                    if (
                        dep_status.get("availableReplicas", None) == spec_replicas
                        and dep_status.get("readyReplicas", None) == spec_replicas
                    ):
                        return (1.0, "")  # done
            elif co["status"] == "False":  # failed
                return (
                    dep_status.get("updatedReplicas", 0) / spec_replicas,
                    co["reason"] + ", " + co.get("message", ""),
                )
            # otherwise, assume in-progress
        elif co["type"] == "ReplicaFailure":
            # note if this status is found, we report failure early here, before k8s times out
            return (
                dep_status.get("updatedReplicas", 0) / spec_replicas,
                co["reason"] + ", " + co.get("message", ""),
            )

    # no errors and not complete yet, assume in-progress
    # (NOTE if "Progressing" condition isn't found, but updated replicas is good, we will return 100% progress; in this case check that other counts are correct, as well!
    progress = dep_status.get("updatedReplicas", 0) / spec_replicas
    if progress == 1.0:
        if (
            dep_status.get("availableReplicas", None) == spec_replicas
            and dep_status.get("readyReplicas", None) == spec_replicas
        ):
            return (1.0, "")  # all good
        progress = 0.99  # available/ready counts aren't there - don't report 100%, wait loop will contiune until ready or time out
    return (progress, "")


# NOTE: update of 'observedGeneration' does not mean that the 'deployment' object is done updating; also checking readyReplicas or availableReplicas in status does not help (these numbers may be for OLD replicas, if the new replicas cannot be started at all). We check for a 'Progressing' condition with a specific 'reason' code as an indication that the deployment is fully updated.
# ? do we need to use --to-revision with the undo command?


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
    def version(self) -> str:
        """The API version of the Kubernetes object (`obj.apiVersion``)."""
        return self.obj.api_version

    @property
    def name(self) -> str:
        """The name of the Kubernetes object (``obj.metadata.name``)."""
        return cast(str, self.obj.metadata.name)

    @name.setter
    def name(self, name: str):
        """Set the name of the Kubernetes objects (``obj.metadata.name``)."""
        self.obj.metadata.name = name

    @property
    def namespace(self) -> str:
        """The namespace of the Kubernetes object (``obj.metadata.namespace``)."""
        return cast(str, self.obj.metadata.namespace)

    @asynccontextmanager
    async def api_client(self) -> Generator[Any]:
        """The API client for the Kubernetes object. This is determined
        by the ``apiVersion`` of the object configuration.

        Raises:
            ValueError: The API version is not supported.
        """
        c = self.api_clients.get(self.version)
        # If we didn't find the client in the api_clients dict, use the
        # preferred version.
        if c is None:
            self.logger.warning(
                f'unknown version ({self.version}), falling back to preferred version'
            )
            c = self.api_clients.get('preferred')
            if c is None:
                raise ValueError(
                    'unknown version specified and no preferred version '
                    f'defined for resource ({self.version})'
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
    
    # @abc.abstractmethod
    # async def patch(self) -> None:
    #     """Partially update the underlying Kubernetes resource in the cluster.
    #     """

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
        self.logger.debug(f'pod: {self.obj}')

        self.obj = await self.api_client.create_namespaced_pod(
            namespace=namespace,
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
        self.logger.debug(f'pod: {self.obj}')

        return await self.api_client.delete_namespaced_pod(
            name=self.name,
            namespace=self.namespace,
            body=options,
        )

    async def refresh(self) -> None:
        """Refresh the underlying Kubernetes Pod resource."""
        self.obj = await self.api_client.read_namespaced_pod_status(
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

    def get_containers(self) -> List[Container]:
        """Get the Pod's containers.

        Returns:
            A list of containers that belong to the Pod.
        """
        self.logger.info(f'getting containers for pod "{self.name}"')
        self.refresh()

        return [Container(c, self) for c in self.obj.spec.containers]

    def get_container(self, name: str) -> Union[Container, None]:
        """Get a container in the Pod by name.

        Args:
            name (str): The name of the Container.

        Returns:
            Container: The Pod's Container with the matching name. If
            no container with the given name is found, ``None`` is returned.
        """
        for c in self.get_containers():
            if c.obj.name == name:
                return c
        return None

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
        self.logger.debug(f'deployment: {self.obj}')

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

    async def status(self) -> client.V1DeploymentStatus:
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
        self.logger.debug(f'pods: {pods}')
        return pods
    
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
        Returns the number of desired pods.
        """
        return self.obj.spec.replicas
    
    @replicas.setter
    def replicas(self, replicas: int) -> None:
        """
        Sets the number of desired pods.
        """
        self.obj.spec.replicas = replicas


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
        yield cls.validate
    
    @classmethod
    def validate(cls, v: StrIntFloat) -> 'Millicore':
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

    def opsani_dict(self) -> dict:
        o_dict = super().opsani_dict()
        o_dict["cpu"]["value"] = float(self.value)
        return o_dict


class Memory(Resource):
    """
    The Memory class models a Kubernetes Memory resource.
    """
    value: ByteSize

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


class Adjustable(BaseModel):
    name: str
    deployment: Deployment
    container: Container

    # Resources
    cpu: CPU
    memory: Memory
    replicas: Replicas
    env: Dict[str, str] = {}

    def to_component(self) -> Component:
        return Component(
            name=self.name,
            settings=[
                self.cpu,
                self.memory,
                self.replicas
            ]
        )
    
    def apply_adjustment(self, adjustment: Adjustment) -> None:
        # TODO: The original driver code modifies the ranges if they are above or below

        name = adjustment.setting_name
        if name in ("cpu", "memory"):
            resource = getattr(self, name)
            if resource.constraint in (ResourceConstraint.request, ResourceConstraint.both):
                self.container.resources.requests[name] = adjustment.value
            if resource.constraint in (ResourceConstraint.limit, ResourceConstraint.both):
                self.container.resources.limits[name] = adjustment.value

        elif adjustment.setting_name == "replicas":
            self.deployment.replicas = int(float(adjustment.value))
            
        else:
            raise RuntimeError(f"adjustment of Kubernetes setting '{adjustment.setting_name}' is not supported")
    
    def __hash__(self):
        return hash((self.name, id(self),))

    class Config:
        arbitrary_types_allowed = True

# TODO: Fold in logger
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

            # TODO: Needs to respect the constraint... (maybe... container.get_resource("cpu", constraint), set_resource("cpu", value, constraint))
            cpu_setting = next(filter(lambda c: c.name == "cpu", component.settings))
            cpu_setting.value = container.resources.limits["cpu"]
            mem_setting = next(filter(lambda c: c.name == "memory", component.settings))
            mem_setting.value = container.resources.limits["memory"]
            replicas_setting = next(filter(lambda c: c.name == "replicas", component.settings))
            replicas_setting.value = deployment.replicas

            adjustables.append(
                Adjustable(
                    name=component.name,
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
        return Description(
            components=list(map(lambda a: a.to_component(), self.adjustables))
        )
    
    def get_adjustable(self, name: str) -> Adjustable:
        return next(filter(lambda a: a.name == name, self.adjustables))

    async def apply_adjustments(self, adjustments: List[Adjustment]) -> None:
        changed = set()
        debug("WTF: ", adjustments)

        for adjustment in adjustments:
            adjustable = self.get_adjustable(adjustment.component_name)
            adjustable.apply_adjustment(adjustment)
            changed.add(adjustable)
        
        patches = list(map(lambda a: a.deployment.patch(), changed))
        results = await asyncio.gather(*patches)
        debug("Results from patches: ", results)

        # TODO: For each one of the modified deployments, we need to watch observedGeneration to progress

        # TODO: Wire in watch/wait for the adjustments=
        async with client.ApiClient() as api:
            v1 = client.CoreV1Api(api)
            async with watch.Watch().stream(v1.list_namespaced_pod, "default") as stream:
                async for event in stream:
                    evt, obj = event['type'], event['object']
                    print("{} pod {} in NS {}".format(evt, obj.metadata.name, obj.metadata.namespace))

        # async with Pod.preferred_client() as api_client:
        # TODO: Start watching as soon as we come online.
        # TODO: Use a queue to allow 
        #     pod_list: client.V1PodList = await api_client.list_namespaced_pod(

        return
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

        # # aborted status reasons that aren't supported: ref-app-inconsistent, ref-app-unavailable

        # # TODO: This response needs to be modeled
        # if not settlement_time:
        #     return {"monitoring": mon0, "status": "ok", "reason": "success"}

        # # TODO: adjust progress accounting when there is settlement_time!=0

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

class KubernetesConfiguration(BaseConfiguration):
    kubeconfig: Optional[FilePath] = Field(
        description="Path to the kubeconfig file. If `None`, use the default from the environment.",
    )
    context: Optional[str] = Field(
        description="The name of the kubeconfig context to use."
    )
    namespace: str = Field(
        "default",
        description="The Kubernetes namespace where the target deployments are running.",
    )
    deployments: List[Component] = Field(
        description="The deployments and adjustable settings to optimize.",
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
    async def describe(self) -> Description:
        # TODO: Need to find the right home for this... (and specify the config file being loaded)
        await config.load_kube_config()
        state = await KubernetesState.read(self.config)
        return state.to_description()

    @on_event()
    def components(self) -> List[Component]:
        return self.config.deployments

    @on_event()
    async def adjust(self, adjustments: List[Adjustment]) -> dict:
        # TODO: What we will want to do is pass in updated settings + component ready to go
        # TODO: change the return value... can it just be none?
        # TODO: Handle this adjust_on stuff
        # adjust_on = desc.get("adjust_on", False)

        # if adjust_on:
        #     try:
        #         should_adjust = eval(adjust_on, {"__builtins__": None}, {"data": data})
        #     except:
        #         should_adjust = False
        #     if not should_adjust:
        #         return {"status": "ok", "reason": "Skipped due to 'adjust_on' condition"}

        await config.load_kube_config()
        state = await KubernetesState.read(self.config)
        await state.apply_adjustments(adjustments)
        # TODO: Unwind this crap: return status and reason
        return { "status": "ok" }

    @on_event()
    def check(self) -> List[Check]:
        # TODO: Verify the connectivity & permissions
        try:
            self.describe()
        except Exception as e:
            return [Check(
                name="Connect to Kubernetes", success=False, comment=str(e)
            )]

        return [Check(name="Connect to Kubernetes", success=True, comment="")]

    async def wait_for_update(
        self,
        appname, obj, patch_gen, c=0, t=1, wait_for_progress=40
    ):
        """wait for a patch to take effect. appname is the namespace, obj is the deployment name, patch_gen is the object generation immediately after the patch was applied (should be a k8s obj with "kind":"Deployment")"""
        wait_for_gen = 5  # time to wait for object update ('observedGeneration')
        # wait_for_progress = 40 # time to wait for rollout to complete

        part = 1.0 / float(t)
        m = "updating {}".format(obj)

        dbg_log("waiting for update: deployment {}, generation {}".format(obj, patch_gen))

        # NOTE: best to implement this with a 'watch', not using an API poll!

        # ?watch=1 & resourceVersion = metadata[resourceVersion], timeoutSeconds=t,
        # --raw=''
        # GET /apis/apps/v1/namespaces/{namespace}/deployments

        # w = Waiter(wait_for_gen, 2)
        # while w.wait():
        while True:
            # NOTE: no progress prints here, this wait should be short
            kubectl = Kubectl(self.config, self.logger)
            r = await kubectl.k_get(appname, DEPLOYMENT + "/" + obj)
            # ydump("tst_wait{}_output_{}.yaml".format(rc,obj),r) ; rc = rc+1

            if test_dep_generation(r, patch_gen):
                break
            
            await asyncio.sleep(2)

        # if w.expired:
        #     raise AdjustError(
        #         "update of {} failed, timed out waiting for k8s object update".format(obj),
        #         status="failed",
        #         reason="adjust-failed",
        #     )

        dbg_log("waiting for progress: deployment {}, generation {}".format(obj, patch_gen))

        p = 0.0  #

        m = "waiting for progress from k8s {}".format(obj)

        # w = Waiter(wait_for_progress, 2)
        c = float(c)
        err = "(wait skipped)"
        # while w.wait():
        while True:
            kubectl = Kubectl(self.config, self.logger)
            r = await kubectl.k_get(appname, DEPLOYMENT + "/" + obj)
            # print_progress(int((c + p) * part * 100), m)
            progress = int((c + p) * part * 100)
            self.logger.info("Awaiting adjustment to take effect...", progress=progress)
            p, err = test_dep_progress(r)
            if p == 1.0:
                return  # all done
            if err:
                break

            await asyncio.sleep(2)

        # loop ended, timed out:
        raise AdjustError(
            "update of {} failed: timed out waiting for replicas to come up, status: {}".format(
                obj, err
            ),
            status="failed",
            reason="start-failed",
        )


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
