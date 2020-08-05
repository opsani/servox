from __future__ import annotations, print_function

import asyncio
from datetime import datetime
import json
import os

from kubetest.objects import namespace
from servo import kubernetes
import sys
import time
from typing import List, Optional, Dict, Any

from pydantic import BaseModel, Extra, Field, FilePath, validator

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

json_enc = json.JSONEncoder(separators=(",", ":")).encode


# TODO: Move these into errors module...
class AdjustError(Exception):
    """base class for error exceptions defined by drivers.
    """

    def __init__(self, *args, status="failed", reason="unknown"):
        self.status = status
        self.reason = reason
        super().__init__(*args)


# === constants
EXCLUDE_LABEL = "optune.ai/exclude"
Gi = 1024 * 1024 * 1024
MEM_STEP = 128 * 1024 * 1024  # minimal useful increment in mem limit/reserve, bytes
CPU_STEP = 0.0125  # 1.25% of a core (even though 1 millicore is the highest resolution supported by k8s)
MAX_MEM = 4 * Gi  # bytes, may be overridden to higher limit
MAX_CPU = 4.0  # cores

# the k8s obj to which we make queries/updates:
DEPLOYMENT = "deployment"
# DEPLOYMENT = "deployment.v1.apps"  # new, not supported in 1.8 (it has v1beta1)
RESOURCE_MAP = {"mem": "memory", "cpu": "cpu"}

# TODO: Class Millicpu? 2000m == 2 == 2.0
def cpuunits(s):
    """convert a string for CPU resource (with optional unit suffix) into a number"""
    if s[-1] == "m":  # there are no units other than 'm' (millicpu)
        return float(s[:-1]) / 1000.0
    return float(s)


# valid mem units: E, P, T, G, M, K, Ei, Pi, Ti, Gi, Mi, Ki
# nb: 'm' suffix found after setting 0.7Gi
mumap = {
    "E": 1000 ** 6,
    "P": 1000 ** 5,
    "T": 1000 ** 4,
    "G": 1000 ** 3,
    "M": 1000 ** 2,
    "K": 1000,
    "m": 1000 ** -1,
    "Ei": 1024 ** 6,
    "Pi": 1024 ** 5,
    "Ti": 1024 ** 4,
    "Gi": 1024 ** 3,
    "Mi": 1024 ** 2,
    "Ki": 1024,
}


def memunits(s):
    """convert a string for memory resource (with optional unit suffix) into a number"""
    for u, m in mumap.items():
        if s.endswith(u):
            return float(s[: -len(u)]) * m
    return float(s)

# def islookinglikerangesetting(s):
#     return "min" in s or "max" in s or "step" in s


# def islookinglikeenumsetting(s):
#     return "values" in s


# def israngesetting(s):
#     return s.get("type") == "range" or islookinglikerangesetting(s)


# def isenumsetting(s):
#     return s.get("type") == "enum" or islookinglikeenumsetting(s)


# def issetting(s):
#     return isinstance(s, dict) and (israngesetting(s) or isenumsetting(s))
# ===

class Kubectl:
    def __init__(self, 
        config: KubernetesConfiguration,
        logger: Logger,
    ) -> None:
        self.config = config
        self.logger = logger

    def kubectl(self, namespace, *args):
        cmd_args = ["kubectl"]

        if self.config.namespace:
            cmd_args.append("--namespace=" + self.config.namespace)
        
        # append conditional args as provided by env vars
        if self.config.server:
            cmd_args.append("--server=" + self.config.server)

        if self.config.token is not None:
            cmd_args.append("--token=" + self.config.server)

        if self.config.insecure_skip_tls_verify:
            cmd_args.append("--insecure-skip-tls-verify=true")

        dbg_txt = "DEBUG: ns='{}', env='{}', r='{}', args='{}'".format(
            namespace,
            os.environ.get("OPTUNE_USE_DEFAULT_NAMESPACE", "???"),
            cmd_args,
            list(args),
        )
        self.logger.debug(dbg_txt)
        return cmd_args + list(args)

    async def k_get(self, namespace, qry):
        """run kubectl get and return parsed json output"""
        if not isinstance(qry, list):
            qry = [qry]

        proc = await asyncio.subprocess.create_subprocess_exec(
            *self.kubectl(namespace, "get", "--output=json", *qry),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout_data, stderr_data = await proc.communicate()
        await proc.wait()
        
        output = stdout_data.decode("utf-8")
        output = json.loads(output)
        return output


    async def k_patch(self, namespace, typ, obj, patchstr):
        """run kubectl patch and return parsed json output"""
        cmd = self.kubectl(namespace, "patch", "--output=json", typ, obj, "-p", patchstr)
        proc = await asyncio.subprocess.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout_data, stderr_data = await proc.communicate()
        await proc.wait()

        output = stdout_data.decode("utf-8")
        output = json.loads(output)
        return output




def dbg_log(*args):
    # TODO: Eliminate this
    from loguru import logger
    logger.debug(args)
    # if os.getenv("TDR_DEBUG_LOG"):
    #     print(*args, file=sys.stderr)


# TODO: Replace
class Waiter(object):
    """an object for use to poll and wait for a condition;
    use:
        w = Waiter(max_time, delay)
        while w.wait():
            if test_condition(): break
        if w.expired:
            raise Hell
    """

    def __init__(self, timeout, delay=1):
        self.timefn = time.time  # change that on windows to time.clock
        self.start = self.timefn()
        self.end = self.start + timeout
        self.delay = delay
        self.expired = False

    def wait(self):
        time.sleep(self.delay)  # TODO: add support for increasing delay over time
        self.expired = self.end < self.timefn()
        return not self.expired

# TODO: This is some kind of progress watcher... WaitCondition?
def test_dep_generation(dep, g):
    """ check if the deployment status indicates it has been updated to the given generation number"""
    return dep["status"]["observedGeneration"] == g

# TODO: Test cases will be: change memory, change cpu, change replica count. 

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


# FIXME: observed a patch trigger spontaneous reduction in replica count! (happened when update was attempted without replica count changes and 2nd replica was not schedulable according to k8s)
# NOTE: update of 'observedGeneration' does not mean that the 'deployment' object is done updating; also checking readyReplicas or availableReplicas in status does not help (these numbers may be for OLD replicas, if the new replicas cannot be started at all). We check for a 'Progressing' condition with a specific 'reason' code as an indication that the deployment is fully updated.
# The 'kubectl rollout status' command relies only on the deployment object - therefore info in it should be sufficient to track progress.
# ? do we need to use --to-revision with the undo command?
# FIXME: cpu request above 0.05 fails for 2 replicas on minikube. Not understood. (NOTE also that setting cpu_limit without specifying request causes request to be set to the same value, except if limit is very low - in that case, request isn't set at all)



def set_rsrc(cp, sn, sv, sel="both"):
    rn = RESOURCE_MAP[sn]
    if sn == "mem":
        sv = str(round(sv, 3)) + "Gi"  # internal memory representation is in GiB
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


def _value(x):
    if isinstance(x, dict) and "value" in x:
        return x["value"]
    return x

class AppState(BaseModel):
    components: List[Component]
    deployments: list #dict
    monitoring: Dict[str, str]

    def get_component(self, name: str) -> Optional[Component]:
        return next(filter(lambda c: c.name == name, self.components))
    
    def get_deployment(self, name: str) -> Optional[dict]:
        return next(filter(lambda d: d["metadata"]["name"] == name, self.deployments))

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

    # TODO: Eliminate in favor of kubeconfig
    server: Optional[str] = Field(
        description="",
    )
    token: Optional[str] = Field(
        description="",
    )
    insecure_skip_tls_verify: bool = Field(
        False,
        description="Disable TLS verification to connect with self-signed certificates.",
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

    # TODO: Temporary...
    class Config:
        extra = Extra.allow

from kubernetes_asyncio import client, config
from kubernetes_asyncio.client.api_client import ApiClient
import abc
import loguru
from loguru import logger as default_logger
from typing import ClassVar, Generator, Mapping, Protocol, Type, Union, cast, get_type_hints, runtime_checkable
from contextlib import asynccontextmanager

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

    class Config:
        extra = Extra.allow
        arbitrary_types_allowed = True

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

    def is_ready(self) -> bool:
        """Check if the Namespace is in the ready state.

        Returns:
            True if in the ready state; False otherwise.
        """
        self.refresh()

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

    async def get_logs(self) -> str:
        """Get all the logs for the Container.

        Returns:
            The Container logs.
        """
        async with ApiClient() as api:
            api_client = client.CoreV1Api(api)
            logs = await api_client.read_namespaced_pod_log(
                name=self.pod.name,
                namespace=self.pod.namespace,
                container=self.obj.name,
            )
            return cast(str, logs)

    def __str__(self) -> str:
        return str(self.obj)

    def __repr__(self) -> str:
        return self.__str__()


class Pod(KubernetesModel):
    """Kubetest wrapper around a Kubernetes `Pod`_ API Object.

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

    def is_ready(self) -> bool:
        """Check if the Deployment is in the ready state.

        Returns:
            True if in the ready state; False otherwise.
        """
        self.refresh()

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
        # response.spec.template.spec.containers[0].resources.limits["memory"] = "2G"
        # spec.template.containers[].resources has our limits & requests
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
        debug("READER")
        return self.obj.spec.replicas
    
    @replicas.setter
    def replicas(self, replicas: int) -> None:
        """
        Sets the number of desired pods.
        """
        debug("SETTING replicas to ", replicas)
        self.obj.spec.replicas = replicas
        debug(self.obj.spec.replicas)

import enum
class ResourceConstraint(enum.Enum):
    request = "request"
    limit = "limit"
    both = "both"

class Resource(Setting):
    constraint: ResourceConstraint = ResourceConstraint.both

# TODO: Add support for units handling (str and repr)...
class CPU(Resource):
    """
    The CPU class models a Kubernetes CPU resource.
    """
    pass

# TODO: Fold into above? 2000m == 2 == 2.0
# def cpuunits(s):
#     """convert a string for CPU resource (with optional unit suffix) into a number"""
#     if s[-1] == "m":  # there are no units other than 'm' (millicpu)
#         return float(s[:-1]) / 1000.0
#     return float(s)

from pydantic import ByteSize
class Memory(Resource):
    """
    The Memory class models a Kubernetes Memory resource.
    """
    value: ByteSize

    def opsani_dict(self) -> dict:
        o_dict = super().opsani_dict()
        o_dict["memory"]["value"] = float(self.value) / Gi
        return o_dict

class Replicas(Setting):
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
        name = adjustment.setting_name
        if name in ("cpu", "memory"):
            # TODO: Add handling for limit/request constraints
            self.container.resources.requests[name] = adjustment.value
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

        for adjustment in adjustments:
            adjustable = self.get_adjustable(adjustment.component_name)
            adjustable.apply_adjustment(adjustment)
            changed.add(adjustable)
        
        patches = list(map(lambda a: a.deployment.patch(), changed))
        tasks = await asyncio.gather(*patches)

        # TODO: Wire in watch/wait for the adjustments

    class Config:
        arbitrary_types_allowed = True


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
    async def adjust(self, data: dict) -> dict:
        # TODO: What we will want to do is pass in updated settings + component ready to go
        # TODO: change the return value... can it just be none?
        adjustments = descriptor_to_adjustments(data["application"]["components"])
        await config.load_kube_config()
        state = await KubernetesState.read(self.config)
        await state.apply_adjustments(adjustments)

        # app_state = await self.raw_query(self.config.namespace, self.config.deployments)
        # debug(app_state, adjustments)
        # r = await self.update(self.config.namespace, app_state, adjustments)
        # debug(r)
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

    ####
    # Ported methods from global module namespace
    
    async def raw_query(self, appname: str, _components: List[Component]):
        """
        Read the list of deployments in a namespace and fill in data into desc.
        Both the input 'desc' and the return value are in the 'settings query response' format.
        NOTE only 'cpu', 'memory' and 'replicas' settings are filled in even if not present in desc.
        Other settings must have a description in 'desc' to be returned.
        """        

        # desc = copy.deepcopy(desc)

        # app = desc["application"]
        # comps = app["components"]

        monitoring = {}
        
        kubectl = Kubectl(self.config, self.logger)
        deployments = await kubectl.k_get(appname, DEPLOYMENT)
        deps_list = deployments["items"]
        if not deps_list:
            # NOTE we don't distinguish the case when the namespace doesn't exist at all or is just empty (k8s will return an empty list whether or not it exists)
            raise AdjustError(
                f"No deployments found in namespace '{appname}'",
                status="aborted", 
                reason="app-unavailable",
            )  # NOTE not a documented 'reason'
        deployments_by_name = {dep["metadata"]["name"]: dep for dep in deps_list}

        raw_specs = {}
        images = {}
        runtime_ids = {}
        components = _components.copy()

        for component in components:
            deployment_name = component.name
            container_name = None
            if "/" in deployment_name:
                deployment_name, container_name = component.name.split("/")
            if not deployment_name in deployments_by_name:
                raise ValueError(f'Could not find deployment "{dep_name}" defined for component "{full_comp_name}" in namespace "{appname}".')
            deployment = deployments_by_name[deployment_name]

            # Containers
            containers = deployment["spec"]["template"]["spec"]["containers"]
            if container_name is not None:
                container = next(filter(lambda c: c["name"] == container_name, containers))
                if not container:
                    raise ValueError('Could not find container with name "{}" in deployment "{}" '
                    'for component "{}" in namespace "{}".'
                    "".format(cont_name, dep_name, full_comp_name, appname))
            else:
                container = containers[0]

            # skip if excluded by label
            try:
                if bool(
                    int(deployment["metadata"].get("labels", {}).get(EXCLUDE_LABEL, "0"))
                ):  # string value of 1 (non-0)
                    self.logger.debug(f"Skipping component {deployment_name}: excluded by label")
                    continue
            except ValueError as e:  # int() is the only thing that should trigger exceptions here
                # TODO add warning to annotations to be returned
                self.logger.warning(
                    "failed to parse exclude label for deployment {}: {}: {}; ignored".format(
                        deployment_name, type(e).__name__, str(e)
                    ),
                    file=sys.stderr,
                )
                # pass # fall through, ignore unparseable label

            # selector for pods, NOTE this relies on having a equality-based label selector,
            # k8s seems to support other types, I don't know what's being used in practice.
            try:
                match_label_selectors = deployment["spec"]["selector"]["matchLabels"]
            except KeyError:
                # TODO: inconsistent errors
                raise AdjustError(
                    "only deployments with matchLabels selector are supported, found selector: {}".format(
                        repr(deployment["spec"].get("selector", {}))
                    ),
                    status="aborted",
                    reason="app-unavailable",
                )  # NOTE not a documented 'reason'
            # convert to string suitable for 'kubect -l labelsel'
            selector_args = ",".join(("{}={}".format(k, v) for k, v in match_label_selectors.items()))

            # list of pods, for runtime_id
            try:
                pods = await kubectl.k_get(appname, ["-l", selector_args, "pods"])
                runtime_ids[deployment_name] = [pod["metadata"]["uid"] for pod in pods["items"]]
            except subprocess.CalledProcessError as e:
                Adjust.print_json_error(
                    error="warning",
                    cl="CalledProcessError",
                    message="Unable to retrieve pods: {}. Output: {}".format(e, e.output),
                )

            # extract deployment settings
            # NOTE: generation, resourceVersion and uid can help detect changes
            # (also, to check PG's k8s code in oco)
            replicas = deployment["spec"]["replicas"]
            raw_specs[deployment_name] = deployment["spec"]["template"]["spec"]  # save for later, used to checksum all specs

            # name, env, resources (limits { cpu, memory }, requests { cpu, memory })
            # FIXME: what to do if there's no mem reserve or limits defined? (a namespace can have a default mem limit, but that's not necessarily set, either)
            # (for now, we give the limit as 0, treated as 'unlimited' - AFAIK)
            images[deployment_name] = container["image"]  # FIXME, is this always defined?
            cpu_setting = component.get_setting("cpu")
            mem_setting = component.get_setting("mem")
            replicas_setting = component.get_setting("replicas")

            container_resources = container.get("resources")
            # TODO: Push all of these defaults into the model.
            if container_resources:
                if mem_setting:
                    mem_val = self.get_rsrc(mem_setting, container_resources, "mem")
                    mem_setting.value = (memunits(mem_val) / Gi)
                    mem_setting.min = (mem_setting.min or MEM_STEP / Gi)
                    mem_setting.max = (mem_setting.max or MAX_MEM / Gi)
                    mem_setting.step = (mem_setting.step or MEM_STEP / Gi)
                    mem_setting.pinned = (mem_setting.pinned or None)

                if cpu_setting:
                    cpu_val = self.get_rsrc(cpu_setting, container_resources, "cpu")
                    cpu_setting.value = cpuunits(cpu_val)
                    cpu_setting.min = (cpu_setting.min or CPU_STEP)
                    cpu_setting.max = (cpu_setting.max or MAX_CPU)
                    cpu_setting.step = (cpu_setting.step or CPU_STEP)
                    cpu_setting.pinned = (cpu_setting.pinned or None)
                # TODO: adjust min/max to include current values, (e.g., increase mem_max to at least current if current > max)
            # set replicas: FIXME: can't actually be set for each container (the pod as a whole is replicated); for now we have no way of expressing this limitation in the setting descriptions
            # note: setting min=max=current replicas, since there is no way to know what is allowed; use override descriptor to loosen range
            if replicas_setting:
                replicas_setting.value = replicas
                replicas_setting.min = (replicas_setting.min or replicas)
                replicas_setting.max = (replicas_setting.max or replicas)
                replicas_setting.step = (replicas_setting.step or 1)
                replicas_setting.pinned = (replicas_setting.pinned or None)

            # current settings of custom env vars (NB: type conv needed for numeric values!)
            cont_env_list = container.get("env", [])
            # include only vars for which the keys 'name' and 'value' are defined
            cont_env_dict = {
                i["name"]: i["value"] for i in cont_env_list if "name" in i and "value" in i
            }

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

        if runtime_ids:
            monitoring["runtime_id"] = get_hash(runtime_ids)

        # app state data
        # (NOTE we strip the component names because our (single-component) 'reference' app will necessarily have a different component name)
        # this should be resolved by complete re-work, if we are to support 'reference' app in a way that allows multiple components
        raw_specs = [raw_specs[k] for k in sorted(raw_specs.keys())]
        images = [images[k] for k in sorted(images.keys())]
        monitoring.update(
            {
                "spec_id": get_hash(raw_specs),
                "version_id": get_hash(images),
                # "runtime_count": replicas_sum
            }
        )

        return AppState(
            components=components,
            deployments=deps_list,
            monitoring=monitoring
        )
    
    def get_rsrc(self, setting, cont_resources, sn):
        rn = RESOURCE_MAP[sn]
        selector = setting.selector or "both"
        if selector == "request":
            val = cont_resources.get("requests", {}).get(rn)
            if val is None:
                val = cont_resources.get("limits", {}).get(rn)
                if val is not None:
                    Adjust.print_json_error(
                        error="warning",
                        cl=None,
                        message='Using the non-selected value "limit" for resource "{}" as the selected value is not set'.format(
                            sn
                        ),
                    )
                else:
                    val = "0"
        else:
            val = cont_resources.get("limits", {}).get(rn)
            if val is None:
                val = cont_resources.get("requests", {}).get(rn)
                if val is not None:
                    if selector == "limit":
                        Adjust.print_json_error(
                            error="warning",
                            cl=None,
                            message='Using the non-selected value "request" for resource "{}" as the selected value is not set'.format(
                                sn
                            ),
                        )
                    # else: don't print warning for 'both'
                else:
                    val = "0"
        return val

    async def update(self, namespace: str, app_state: AppState, adjustments: List[Component]):

        # TODO: Needs to be ported
        # adjust_on = desc.get("adjust_on", False)

        # if adjust_on:
        #     try:
        #         should_adjust = eval(adjust_on, {"__builtins__": None}, {"data": data})
        #     except:
        #         should_adjust = False
        #     if not should_adjust:
        #         return {"status": "ok", "reason": "Skipped due to 'adjust_on' condition"}

        # NOTE: we'll need the raw k8s api data to see the container names (setting names for a single-container
        #       pod will include only the deployment(=pod) name, not the container name)
        # _, raw = raw_query(appname, desc)

        # convert k8s list of deployments into map
        # raw = {dep["metadata"]["name"]: dep for dep in raw}

        patchlst = {}
        # FIXME: NB: app-wide settings not supported
        
        cfg = {}
        # cfg = data.get("control", {})

        # FIXME: off-spec; step-down in data if a 'state' key is provided at the top.
        # if "state" in data:
        #     data = data["state"]

        for component in adjustments:
            if not component.settings:
                continue
            patches = {}
            replicas = None
            current_state = app_state.get_component(component.name)

            # find deployment name and container name, and verify it's existence
            container_name = None
            deployment_name = component.name
            if "/" in deployment_name:
                deployment_name, container_name = comp_name.split("/", 1)
            deployment = app_state.get_deployment(deployment_name)
            if not deployment:
                debug(app_state.deployments)
                raise AdjustError(
                    'Cannot find deployment with name "{}" for component "{}" in namespace "{}"'.format(deployment_name, component.name, namespace),
                    status="failed",
                    reason="unknown",
                )  # FIXME 'reason' code (either bad config or invalid input to update())
            container_name = (
                container_name
                or deployment["spec"]["template"]["spec"]["containers"][0]["name"]
            )  # chk for KeyError FIXME
            available_containers = set(
                c["name"] for c in deployment["spec"]["template"]["spec"]["containers"]
            )
            if container_name not in available_containers:
                raise AdjustError(
                    'Could not find container with name "{}" in deployment "{}" '
                    'for component "{}" in namespace "{}".'.format(
                        container_name, deployment_name, component.name, namespace
                    ),
                    status="failed",
                    reason="unknown",
                )  # see note above

            cont_patch = patches.setdefault(container_name, {})

            # TODO: Port this
            command = component.command
            if command:
                if command.get("encoder"):
                    cont_patch["command"], encoded_settings = encode_encoder(
                        settings, command["encoder"], expected_type=list
                    )

                    # Prevent encoded settings from further processing
                    for setting in encoded_settings:
                        del settings[setting]

            env = component.env
            if env:
                for en, ev in env.items():
                    if ev.get("encoder"):
                        val, encoded_settings = encode_encoder(
                            settings, ev["encoder"], expected_type=str
                        )
                        patch_env = cont_patch.setdefault("env", [])
                        patch_env.append({"name": en, "value": val})

                        # Prevent encoded settings from further processing
                        for setting in encoded_settings:
                            del settings[setting]
                    elif issetting(ev):
                        patch_env = cont_patch.setdefault("env", [])
                        patch_env.append({"name": en, "value": str(settings[en]["value"])})
                        del settings[en]

            # Settings and env vars
            # for name, value in settings.items():
            for setting in component.settings:
                name = setting.name
                value = _value(
                    setting.value
                )  # compatibility: allow a scalar, but also work with {"value": {anything}}
                cont_patch = patches.setdefault(container_name, {})
                if setting.name in ("mem", "cpu"):
                    set_rsrc(
                        cont_patch,
                        name,
                        value,
                        setting.selector or "both",
                    )
                    continue
                elif name == "replicas":
                    replicas = int(value)

            patch = patchlst.setdefault(deployment_name, {})
            if patches:  # convert to array
                cp = (
                    patch.setdefault("spec", {})
                    .setdefault("template", {})
                    .setdefault("spec", {})
                    .setdefault("containers", [])
                )
                for n, v in patches.items():
                    v["name"] = n
                    cp.append(v)
            if replicas is not None:
                patch.setdefault("spec", {})["replicas"] = replicas

        if not patchlst:
            raise Exception(
                "No components were defiend in a configuration file. Cannot proceed with an adjustment."
            )

        # NOTE: optimization possible: apply all patches first, then wait for them to complete (significant if making many changes at once!)

        # NOTE: it seems there's no way to update multiple resources with one 'patch' command
        #       (though -f accepts a directory, not sure how -f=dir works; maybe all listed resources
        #        get the *same* patch from the cmd line - not what we want)

        # execute patch commands
        patched_count = 0
        for n, v in patchlst.items():
            # ydump("tst_before_output_{}.yaml".format(n), k_get(appname, DEPLOYMENT + "/" + n))
            # run: kubectl patch deployment[.v1.apps] $n -p "{jsondata}"
            patchstr = json_enc(v)
            try:
                kubectl = Kubectl(self.config, self.logger)
                patch_r = await kubectl.k_patch(namespace, DEPLOYMENT, n, patchstr)
            except Exception as e:  # TODO: limit to expected errors
                raise AdjustError(str(e), status="failed", reason="adjust-failed")
            p, _ = test_dep_progress(patch_r)
            if test_dep_generation(patch_r, patch_r["metadata"]["generation"]) and p == 1.0:
                # patch made no changes, skip wait_for_update:
                patched_count = patched_count + 1
                continue

            # ydump("tst_patch_output_{}.yaml".format(n), patch_r)

            # wait for update to complete (and print progress)
            # timeout default is set to be slightly higher than the default K8s timeout (so we let k8s detect progress stall first)
            try:
                await self.wait_for_update(
                    namespace,
                    n,
                    patch_r["metadata"]["generation"],
                    patched_count,
                    len(patchlst),
                    cfg.get("timeout", 630),
                )
            except AdjustError as e:
                if e.reason != "start-failed":  # not undo-able
                    raise
                onfail = cfg.get(
                    "on_fail", "keep"
                )  # valid values: keep, destroy, rollback (destroy == scale-to-zero, not supported)
                if onfail == "rollback":
                    try:
                        # TODO: This has to be ported
                        subprocess.call(
                            kubectl(appname, "rollout", "undo", DEPLOYMENT + "/" + n)
                        )
                        print("UNDONE", file=sys.stderr)
                    except subprocess.CalledProcessError:
                        # progress msg with warning TODO
                        print("undo for {} failed: {}".format(n, e), file=sys.stderr)
                raise
            patched_count = patched_count + 1

        # spec_id and version_id should be tested without settlement_time, too - TODO

        # post-adjust settlement, if enabled
        testdata0 = await self.raw_query(namespace, app_state.components)
        settlement_time = cfg.get("settlement", 0)
        mon0 = testdata0.monitoring

        if "ref_version_id" in mon0 and mon0["version_id"] != mon0["ref_version_id"]:
            raise AdjustError(
                "application version does not match reference version",
                status="aborted",
                reason="version-mismatch",
            )

        # aborted status reasons that aren't supported: ref-app-inconsistent, ref-app-unavailable

        # TODO: This response needs to be modeled
        if not settlement_time:
            return {"monitoring": mon0, "status": "ok", "reason": "success"}

        # TODO: adjust progress accounting when there is settlement_time!=0

        # wait and watch the app, checking for changes
        # TODO: Port this...
        w = Waiter(
            settlement_time, delay=min(settlement_time, 30)
        )  # NOTE: delay between tests may be made longer than the delay between progress reports
        while w.wait():
            testdata = raw_query(namespace, app_state.components)
            mon = testdata.monitoring
            # compare to initial mon data set
            if mon["runtime_id"] != mon0["runtime_id"]:  # restart detected
                # TODO: allow limited number of restarts? (and how to distinguish from rejected/unstable??)
                raise AdjustError(
                    "component(s) restart detected",
                    status="transient-failure",
                    reason="app-restart",
                )
            # TODO: what to do with version change?
            #        if mon["version_id"] != mon0["version_id"]:
            #            raise AdjustError("application was modified unexpectedly during settlement", status="transient-failure", reason="app-update")
            if mon["spec_id"] != mon0["spec_id"]:
                raise AdjustError(
                    "application configuration was modified unexpectedly during settlement",
                    status="transient-failure",
                    reason="app-update",
                )
            if mon["ref_spec_id"] != mon0["ref_spec_id"]:
                raise AdjustError(
                    "reference application configuration was modified unexpectedly during settlement",
                    status="transient-failure",
                    reason="ref-app-update",
                )
            if mon["ref_runtime_count"] != mon0["ref_runtime_count"]:
                raise AdjustError("", status="transient-failure", reason="ref-app-scale")

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

def descriptor_to_components(descriptor: dict) -> List[Component]:
    components = []
    for component_name in descriptor:
        settings = []
        for setting_name in descriptor[component_name]["settings"]:
            setting_values = descriptor[component_name]["settings"][setting_name]
            # FIXME: Temporary hack
            if not setting_values.get("type", None):
                setting_values["type"] = "range"
            setting = Setting(name=setting_name, min=1, max=4, step=1, **setting_values)
            settings.append(setting)
        component = Component(name=component_name, settings=settings)
        components.append(component)
    return components

def descriptor_to_adjustments(descriptor: dict) -> List[Adjustment]:
    adjustments = []
    for component_name in descriptor:
        for setting_name, attrs in descriptor[component_name]["settings"].items():
            adjustment = Adjustment(
                component_name=component_name, 
                setting_name=setting_name,
                value=attrs["value"]
            )
            adjustments.append(adjustment)
    return adjustments
