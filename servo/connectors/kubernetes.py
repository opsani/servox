
from __future__ import print_function, annotations

import asyncio
import copy
import errno
import importlib
import json
import os
import subprocess
import sys
import time
from collections.abc import Iterable
from typing import List, Optional, Dict, Any

import yaml
from pydantic import BaseModel, Extra, validator

from servo import (
    BaseConfiguration,
    Connector,
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


# TODO: Temporary
class ConfigError(Exception):
    pass


# === constants
DESC_FILE = "./servo.yaml"
EXCLUDE_LABEL = "optune.ai/exclude"
Gi = 1024 * 1024 * 1024
MEM_STEP = 128 * 1024 * 1024  # minimal useful increment in mem limit/reserve, bytes
CPU_STEP = 0.0125  # 1.25% of a core (even though 1 millicore is the highest resolution supported by k8s)
MAX_MEM = 4 * Gi  # bytes, may be overridden to higher limit
MAX_CPU = 4.0  # cores
# MAX_REPLICAS = 1000 # arbitrary, TBD

# the k8s obj to which we make queries/updates:
DEPLOYMENT = "deployment"
# DEPLOYMENT = "deployment.v1.apps"  # new, not supported in 1.8 (it has v1beta1)
RESOURCE_MAP = {"mem": "memory", "cpu": "cpu"}

# TODO: Support as a plugin
def import_encoder_base():
    try:
        return importlib.import_module("encoders.base")
    except ImportError:
        raise ImportError(
            "Unable to import base for encoders when handling `command` section."
        )


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


def encoder_setting_name(setting_name, encoder_config):
    prefix = (
        encoder_config["setting_prefix"] if "setting_prefix" in encoder_config else ""
    )
    return "{}{}".format(prefix, setting_name)


def describe_encoder(value, config, exception_context="a describe phase of an encoder"):
    encoder_base = import_encoder_base()
    try:
        settings = encoder_base.describe(config, value or "")
        for name, setting in settings.items():
            yield (encoder_setting_name(name, config), setting)
    except BaseException as e:
        raise Exception("Error while handling {}: {}".format(exception_context, str(e)))


def encode_encoder(
    settings,
    config,
    expected_type=None,
    exception_context="an encode phase of an encoder",
):
    encoder_base = import_encoder_base()
    try:
        sanitized_settings = settings
        prefix = config.get("setting_prefix")
        if prefix:
            sanitized_settings = dict(
                map(
                    lambda i: (i[0].lstrip(prefix), i[1]),
                    filter(lambda i: i[0].startswith(prefix), settings.items()),
                )
            )
        encoded_value, encoded_settings = encoder_base.encode(
            config, sanitized_settings, expected_type=expected_type
        )
        encoded_settings = list(
            map(
                lambda setting_name: encoder_setting_name(setting_name, config),
                encoded_settings,
            )
        )
        return encoded_value, encoded_settings
    except BaseException as e:
        raise Exception("Error while handling {}: {}".format(exception_context, str(e)))


def islookinglikerangesetting(s):
    return "min" in s or "max" in s or "step" in s


def islookinglikeenumsetting(s):
    return "values" in s


def israngesetting(s):
    return s.get("type") == "range" or islookinglikerangesetting(s)


def isenumsetting(s):
    return s.get("type") == "enum" or islookinglikeenumsetting(s)


def issetting(s):
    return isinstance(s, dict) and (israngesetting(s) or isenumsetting(s))

# TODO: Moves into a utility class...


# ===

class Kubectl:
    def __init__(self, 
        config: KubernetesConfiguration,
        logger: Logger,
    ) -> None:
        self.config = config
        self.logger = logger

    # TODO: Becomes __call__
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

class KubernetesConfiguration(connector.BaseConfiguration):
    namespace: str
    server: Optional[str]
    token: Optional[str]
    insecure_skip_tls_verify: bool = False
    components: List[Component]

    @classmethod
    def generate(cls, **kwargs) -> "KubernetesConfiguration":
        return cls(
            namespace="default",
            description="Update the namespace, deployment, etc. to match your Kubernetes cluster",
            **kwargs
        )

    # TODO: Temporary...
    class Config:
        extra = Extra.allow

@connector.metadata(
    description="Kubernetes adjust connector",
    version="1.5.0",
    homepage="https://github.com/opsani/kubernetes-connector",
    license=License.APACHE2,
    maturity=Maturity.EXPERIMENTAL,
)
class KubernetesConnector(connector.Connector):
    config: KubernetesConfiguration

    @on_event()
    async def describe(self) -> Description:
        debug(self.config)
        result = await self.raw_query(self.config.namespace, self.config.components)
        debug(result)
        return Description(components=self.config.components)

    @on_event()
    def components(self) -> Description:
        return self.config.components

    @on_event()
    async def adjust(self, data: dict) -> dict:
        adjustments = descriptor_to_components(data["application"]["components"])
        app_state = await self.raw_query(self.config.namespace, self.config.components)
        debug(app_state, adjustments)
        r = await self.update(self.config.namespace, app_state, adjustments)
        debug(r)
        return r

    @on_event()
    def check(self) -> List[Check]:
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

        # TODO: Model will have components, deployments, 
        
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

        # TODO: Move to a model
        # NOTE: These three are used for hashing
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

            # TODO: Break into a method...
            # TODO: This is broken atm
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

    # TODO: This is actually doing the adjust...
    async def update(self, namespace: str, app_state: AppState, adjustments: List[Component]):

        # TODO: Seems like a config element? Disabled for now...
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
        # TODO: Model this
        # raw = {dep["metadata"]["name"]: dep for dep in raw}

        patchlst = {}
        # FIXME: NB: app-wide settings not supported
        
        # TODO: Wire support for serializing control section
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

            # TODO: Needs to be modeled
            cont_patch = patches.setdefault(container_name, {})

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