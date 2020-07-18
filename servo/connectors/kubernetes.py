from __future__ import print_function

import copy
import errno
import importlib
import json
import os
import subprocess
import sys
import time
from collections.abc import Iterable
from typing import List, Optional

import yaml
from pydantic import Extra

from servo import (
    Check,
    Component,
    Description,
    License,
    Maturity,
    Setting,
    connector,
    on_event
)

json_enc = json.JSONEncoder(separators=(",", ":")).encode


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


# === compute hash of arbitrary data struct
# (copied inline from skopos/.../plugins/spec_hash_helper.py)
import hashlib


# TODO: Why do we need this?
def _dbg(*data):
    with open("/skopos/plugins/dbg.log", "a") as f:
        print(data, file=f)


def get_hash(data):
    """md5 hash of Python data. This is limited to scalars that are convertible to string and container
    structures (list, dict) containing such scalars. Some data items are not distinguishable, if they have
    the same representation as a string, e.g., hash(b'None') == hash('None') == hash(None)"""
    # _dbg("get_hash", data)
    hasher = hashlib.md5()
    dump_container(data, hasher.update)
    return hasher.hexdigest()


def dump_container(c, func):
    """stream the contents of a container as a string through a function
    in a repeatable order, suitable, e.g., for hashing
    """
    #
    if isinstance(c, dict):  # dict
        func("{".encode("utf-8"))
        for k in sorted(c):  # for all repeatable
            func("{}:".format(k).encode("utf-8"))
            dump_container(c[k], func)
            func(",".encode("utf-8"))
        func("}".encode("utf-8"))
    elif isinstance(c, list):  # list
        func("[".encode("utf-8"))
        for k in c:  # for all repeatable
            dump_container(k, func)
            func(",".encode("utf-8"))
        func("]".encode("utf-8"))
    else:  # everything else
        if isinstance(c, type(b"")):
            pass  # already a stream, keep as is
        elif isinstance(c, str):
            # encode to stream explicitly here to avoid implicit encoding to ascii
            c = c.encode("utf-8")
        else:
            c = str(c).encode("utf-8")  # convert to string (e.g., if integer)
        func(c)  # simple value, string or convertible-to-string


# ===


def kubectl(namespace, *args):
    cmd_args = ["kubectl"]
    if not bool(int(os.environ.get("OPTUNE_USE_DEFAULT_NAMESPACE", "0"))):
        cmd_args.append("--namespace=" + namespace)
    # append conditional args as provided by env vars
    if os.getenv("OPTUNE_K8S_SERVER") is not None:
        cmd_args.append("--server=" + os.getenv("OPTUNE_K8S_SERVER"))
    if os.getenv("OPTUNE_K8S_TOKEN") is not None:
        cmd_args.append("--token=" + os.getenv("OPTUNE_K8S_TOKEN"))
    if bool(os.getenv("OPTUNE_K8S_SKIP_TLS_VERIFY", False)):
        cmd_args.append("--insecure-skip-tls-verify=true")
    dbg_txt = "DEBUG: ns='{}', env='{}', r='{}', args='{}'".format(
        namespace,
        os.environ.get("OPTUNE_USE_DEFAULT_NAMESPACE", "???"),
        cmd_args,
        list(args),
    )
    if args[0] == "patch":
        print(dbg_txt, file=sys.stderr)
    else:
        dbg_log(dbg_txt)
    return cmd_args + list(args)


def k_get(namespace, qry):
    """run kubectl get and return parsed json output"""
    if not isinstance(qry, list):
        qry = [qry]
    # this will raise exception if it fails:
    output = subprocess.check_output(kubectl(namespace, "get", "--output=json", *qry))
    output = output.decode("utf-8")
    output = json.loads(output)
    return output


def k_patch(namespace, typ, obj, patchstr):
    """run kubectl patch and return parsed json output"""

    # this will raise exception if it fails:
    cmd = kubectl(namespace, "patch", "--output=json", typ, obj, "-p", patchstr)
    output = subprocess.check_output(cmd)
    output = output.decode("utf-8")
    output = json.loads(output)
    return output


def read_desc():
    """load the user-defined descriptor, returning a dictionary of the contents under the k8s top-level key, if any"""
    # TODO: Eliminate this
    try:
        f = open(DESC_FILE)
        desc = yaml.safe_load(f)
    except IOError as e:
        if e.errno == errno.ENOENT:
            raise ConfigError("configuration file {} does not exist".format(DESC_FILE))
        raise ConfigError(
            "cannot read configuration from {}: {}".format(DESC_FILE, e.strerror)
        )
    except yaml.error.YAMLError as e:
        raise ConfigError("syntax error in {}: {}".format(DESC_FILE, str(e)))

    refer_tip = "You can refer to a sample configuration in README.md."
    assert bool(desc), "Configuration file is empty."
    driver_key = "kubernetes"

    # TODO: Everything below gets ported to settings class
    if os.environ.get("OPTUNE_USE_DRIVER_NAME", False):
        driver_key = os.path.basename(__file__)
    assert driver_key in desc and desc[driver_key], (
        "No configuration were defined for K8s driver in config file {}. "
        'Please set up configuration for deployments under key "{}". '
        "{}".format(DESC_FILE, refer_tip, driver_key)
    )
    desc = desc[driver_key]

    assert (
        "application" in desc and desc["application"]
    ), 'Section "application" was not defined in a configuration file. {}'.format(
        refer_tip
    )
    assert (
        "components" in desc["application"]
        and desc["application"]["components"] is not None
    ), 'Section "components" was not defined in a configuration file section "application". {}'.format(
        refer_tip
    )
    assert desc["application"]["components"], (
        "No components were defined in a configuration file. "
        "Please define at least one component. {}".format(refer_tip)
    )

    comps = desc["application"]["components"]
    replicas_tracker = {}
    for name, comp in comps.items():
        settings = comp.get("settings", {})
        if "replicas" in settings:
            dep_name = name.split("/")[0]  # if no '/', this just gets the whole name
            replicas_tracker.setdefault(dep_name, 0)
            replicas_tracker[dep_name] += 1

    if len(replicas_tracker) < sum(replicas_tracker.values()):
        rotten_deps = map(
            lambda d: d[0], filter(lambda c: c[1] > 1, replicas_tracker.items())
        )
        raise Exception(
            'Some components have more than one setting "replicas" defined. Specifically: {}. '
            'Please, keep only one "replicas" per deployment.'.format(
                ", ".join(rotten_deps)
            )
        )

    return desc


def numval(v, minv, maxv, step=1.0, pinn=None):
    """shortcut for creating linear setting descriptions"""
    ret = {"value": v, "min": minv, "max": maxv, "step": step, "type": "range"}
    if pinn is not None:
        ret["pinned"] = pinn == True
    return ret


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


def check_setting(name, settings):
    assert isinstance(
        settings, Iterable
    ), 'Object "settings" passed to check_setting() is not iterable.'
    assert name not in settings, (
        'Setting "{}" has been define more than once. '
        "Please, check other config sections for setting duplicates.".format(name)
    )


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


def get_rsrc(desc_settings, cont_resources, sn):
    rn = RESOURCE_MAP[sn]
    selector = desc_settings.get(sn, {}).get("selector", "both")
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


def raw_query(appname, desc):
    """
    Read the list of deployments in a namespace and fill in data into desc.
    Both the input 'desc' and the return value are in the 'settings query response' format.
    NOTE only 'cpu', 'memory' and 'replicas' settings are filled in even if not present in desc.
    Other settings must have a description in 'desc' to be returned.
    """
    desc = copy.deepcopy(desc)

    app = desc["application"]
    comps = app["components"]

    cfg = desc.pop(
        "control", {}
    )  # FIXME TODO - query doesn't receive data from remote, only the local cfg can be used; where in the data should the "control" section really be?? note, [userdata][deployment] sub-keys for specifying the 'reference' app means we have to have that 'reference' as a single deployment and it has to be excluded from enumeration as an 'adjustable' component, using the whitelist.
    refapp = cfg.get("userdata", {}).get("deployment", None)
    mon_data = {}
    if refapp:
        d2 = desc.copy()
        c2 = copy.deepcopy(cfg)
        c2["userdata"].pop("deployment", None)
        d2["control"] = c2
        if (
            len(comps) != 1
        ):  # 'reference app' works only with single-component (due to the use of deployment name as 'component name' and having both apps in the same namespace)
            raise AdjustError(
                "operation with reference app not possible when multiple components are defined",
                status="aborted",
                reason="ref-app-unavailable",
            )
        refcomps = {refapp: comps[list(comps.keys())[0]]}
        d2["application"] = {
            "components": refcomps
        }  # single component, renamed (so we pick the 'reference deployment' in the same namespace)
        try:
            refqry, _ = raw_query(appname, d2)
        except AdjustError as e:
            raise AdjustError(str(e), status="aborted", reason="ref-app-unavailable")
        # let other exceptions go unchanged

        # TODO: maybe something better than a sum is needed here, some multi-component scale events could end up modifying scale counts without changing the overall sum
        replicas_sum = sum(
            (
                c["settings"]["replicas"]["value"]
                for c in refqry["application"]["components"].values()
            )
        )
        refqry = refqry["monitoring"]  # we don't need other data from refqry any more
        mon_data = {
            "ref_spec_id": refqry["spec_id"],
            "ref_version_id": refqry["version_id"],
            "ref_runtime_count": replicas_sum,
        }
        if refqry.get("runtime_id"):
            mon_data["ref_runtime_id"] = refqry["runtime_id"]

    deployments = k_get(appname, DEPLOYMENT)
    # note d["Kind"] should be "List"
    deps_list = deployments["items"]
    if (
        not deps_list
    ):  # NOTE we don't distinguish the case when the namespace doesn't exist at all or is just empty (k8s will return an empty list whether or not it exists)
        raise AdjustError(
            "application '{}' does not exist or has no components".format(appname),
            status="aborted",
            reason="app-unavailable",
        )  # NOTE not a documented 'reason'
    deps_dict = {dep["metadata"]["name"]: dep for dep in deps_list}
    raw_specs = {}
    imgs = {}
    runtime_ids = {}
    # ?? TODO: is it possible to have an item in 'd' with "kind" other than "Deployment"? (likely no)
    #          is it possible to have replicas == 0 (and how do we represent that, if at all)
    for full_comp_name in comps.keys():
        dep_name = full_comp_name
        cont_name = None
        if "/" in dep_name:
            dep_name, cont_name = full_comp_name.split("/")
        assert dep_name in deps_dict, (
            'Could not find deployment "{}" defined for component "{}" in namespace "{}".'
            "".format(dep_name, full_comp_name, appname)
        )
        dep = deps_dict[dep_name]
        conts = dep["spec"]["template"]["spec"]["containers"]
        if cont_name is not None:
            contsd = {c["name"]: c for c in conts}
            assert cont_name in contsd, (
                'Could not find container with name "{}" in deployment "{}" '
                'for component "{}" in namespace "{}".'
                "".format(cont_name, dep_name, full_comp_name, appname)
            )
            cont = contsd[cont_name]
        else:
            cont = conts[0]

        # skip if excluded by label
        try:
            if bool(
                int(dep["metadata"].get("labels", {}).get(EXCLUDE_LABEL, "0"))
            ):  # string value of 1 (non-0)
                continue
        except ValueError as e:  # int() is the only thing that should trigger exceptions here
            # TODO add warning to annotations to be returned
            print(
                "failed to parse exclude label for deployment {}: {}: {}; ignored".format(
                    dep_name, type(e).__name__, str(e)
                ),
                file=sys.stderr,
            )
            # pass # fall through, ignore unparseable label

        # selector for pods, NOTE this relies on having a equality-based label selector,
        # k8s seems to support other types, I don't know what's being used in practice.
        try:
            sel = dep["spec"]["selector"]["matchLabels"]
        except KeyError:
            raise AdjustError(
                "only deployments with matchLabels selector are supported, found selector: {}".format(
                    repr(dep["spec"].get("selector", {}))
                ),
                status="aborted",
                reason="app-unavailable",
            )  # NOTE not a documented 'reason'
        # convert to string suitable for 'kubect -l labelsel'
        sel = ",".join(("{}={}".format(k, v) for k, v in sel.items()))

        # list of pods, for runtime_id
        try:
            pods = k_get(appname, ["-l", sel, "pods"])
            pods = pods["items"]
            runtime_ids[dep_name] = [pod["metadata"]["uid"] for pod in pods]
        except subprocess.CalledProcessError as e:
            Adjust.print_json_error(
                error="warning",
                cl="CalledProcessError",
                message="Unable to retrieve pods: {}. Output: {}".format(e, e.output),
            )

        # extract deployment settings
        # NOTE: generation, resourceVersion and uid can help detect changes
        # (also, to check PG's k8s code in oco)
        replicas = dep["spec"]["replicas"]
        tmplt_spec = dep["spec"]["template"]["spec"]
        raw_specs[dep_name] = tmplt_spec  # save for later, used to checksum all specs

        # name, env, resources (limits { cpu, memory }, requests { cpu, memory })
        # FIXME: what to do if there's no mem reserve or limits defined? (a namespace can have a default mem limit, but that's not necessarily set, either)
        # (for now, we give the limit as 0, treated as 'unlimited' - AFAIK)
        imgs[full_comp_name] = cont["image"]  # FIXME, is this always defined?
        comp = comps[full_comp_name] = comps[full_comp_name] or {}
        settings = comp["settings"] = comp.setdefault("settings", {}) or {}
        read_mem = not settings or "mem" in settings
        read_cpu = not settings or "cpu" in settings
        read_replicas = not settings or "replicas" in settings
        res = cont.get("resources")
        if res:
            if read_mem:
                mem_val = get_rsrc(desc_settings=settings, cont_resources=res, sn="mem")
                # (value, min, max, step) all in GiB
                settings["mem"] = numval(
                    v=memunits(mem_val) / Gi,
                    minv=(settings.get("mem") or {}).get("min", MEM_STEP / Gi),
                    maxv=(settings.get("mem") or {}).get("max", MAX_MEM / Gi),
                    step=(settings.get("mem") or {}).get("step", MEM_STEP / Gi),
                    pinn=(settings.get("mem") or {}).get("pinned", None),
                )
            if read_cpu:
                cpu_val = get_rsrc(desc_settings=settings, cont_resources=res, sn="cpu")
                # (value, min, max, step), all in CPU cores
                settings["cpu"] = numval(
                    v=cpuunits(cpu_val),
                    minv=(settings.get("cpu") or {}).get("min", CPU_STEP),
                    maxv=(settings.get("cpu") or {}).get("max", MAX_CPU),
                    step=(settings.get("cpu") or {}).get("step", CPU_STEP),
                    pinn=(settings.get("cpu") or {}).get("pinned", None),
                )
            # TODO: adjust min/max to include current values, (e.g., increase mem_max to at least current if current > max)
        # set replicas: FIXME: can't actually be set for each container (the pod as a whole is replicated); for now we have no way of expressing this limitation in the setting descriptions
        # note: setting min=max=current replicas, since there is no way to know what is allowed; use override descriptor to loosen range
        if read_replicas:
            settings["replicas"] = numval(
                v=replicas,
                minv=(settings.get("replicas") or {}).get("min", replicas),
                maxv=(settings.get("replicas") or {}).get("max", replicas),
                step=(settings.get("replicas") or {}).get("step", 1),
                pinn=(settings.get("replicas") or {}).get("pinned", None),
            )

        # current settings of custom env vars (NB: type conv needed for numeric values!)
        cont_env_list = cont.get("env", [])
        # include only vars for which the keys 'name' and 'value' are defined
        cont_env_dict = {
            i["name"]: i["value"] for i in cont_env_list if "name" in i and "value" in i
        }

        env = comp.get("env")
        if env:
            for en, ev in env.items():
                check_setting(en, settings)
                assert isinstance(
                    ev, dict
                ), 'Setting "{}" in section "env" of a config file is not a dictionary.'
                if "encoder" in ev:
                    for name, setting in describe_encoder(
                        cont_env_dict.get(en),
                        ev["encoder"],
                        exception_context="an environment variable {}" "".format(en),
                    ):
                        check_setting(name, settings)
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
            # Remove section "env" from final descriptor
            del comp["env"]

        command = comp.get("command")
        if command:
            if command.get("encoder"):
                for name, setting in describe_encoder(
                    cont.get("command", []),
                    command["encoder"],
                    exception_context="a command section",
                ):
                    check_setting(name, settings)
                    settings[name] = setting
                # Remove section "command" from final descriptor
            del comp["command"]

    if runtime_ids:
        mon_data["runtime_id"] = get_hash(runtime_ids)

    # app state data
    # (NOTE we strip the component names because our (single-component) 'reference' app will necessarily have a different component name)
    # this should be resolved by complete re-work, if we are to support 'reference' app in a way that allows multiple components
    raw_specs = [raw_specs[k] for k in sorted(raw_specs.keys())]
    imgs = [imgs[k] for k in sorted(imgs.keys())]
    mon_data.update(
        {
            "spec_id": get_hash(raw_specs),
            "version_id": get_hash(imgs),
            # "runtime_count": replicas_sum
        }
    )

    desc["monitoring"] = mon_data

    return desc, deps_list


# DEBUG:
def ydump(fn, data):
    f = open(fn, "w")
    yaml.dump(data, f)
    f.close()


def dbg_log(*args):
    from loguru import logger

    logger.debug(args)
    # if os.getenv("TDR_DEBUG_LOG"):
    #     print(*args, file=sys.stderr)


def query(appname, desc):
    r, _ = raw_query(appname, desc)
    return r


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


def wait_for_update(
    appname, obj, patch_gen, print_progress, c=0, t=1, wait_for_progress=40
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

    w = Waiter(wait_for_gen, 2)
    while w.wait():
        # NOTE: no progress prints here, this wait should be short
        r = k_get(appname, DEPLOYMENT + "/" + obj)
        # ydump("tst_wait{}_output_{}.yaml".format(rc,obj),r) ; rc = rc+1

        if test_dep_generation(r, patch_gen):
            break

    if w.expired:
        raise AdjustError(
            "update of {} failed, timed out waiting for k8s object update".format(obj),
            status="failed",
            reason="adjust-failed",
        )

    dbg_log("waiting for progress: deployment {}, generation {}".format(obj, patch_gen))

    p = 0.0  #

    m = "waiting for progress from k8s {}".format(obj)

    w = Waiter(wait_for_progress, 2)
    c = float(c)
    err = "(wait skipped)"
    while w.wait():
        r = k_get(appname, DEPLOYMENT + "/" + obj)
        print_progress(int((c + p) * part * 100), m)
        p, err = test_dep_progress(r)
        if p == 1.0:
            return  # all done
        if err:
            break

    # loop ended, timed out:
    raise AdjustError(
        "update of {} failed: timed out waiting for replicas to come up, status: {}".format(
            obj, err
        ),
        status="failed",
        reason="start-failed",
    )


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


def update(appname, desc, data, print_progress):

    # TODO: Seems like a config element?
    adjust_on = desc.get("adjust_on", False)

    if adjust_on:
        try:
            should_adjust = eval(adjust_on, {"__builtins__": None}, {"data": data})
        except:
            should_adjust = False
        if not should_adjust:
            return {"status": "ok", "reason": "Skipped due to 'adjust_on' condition"}

    # NOTE: we'll need the raw k8s api data to see the container names (setting names for a single-container
    #       pod will include only the deployment(=pod) name, not the container name)
    _, raw = raw_query(appname, desc)

    # convert k8s list of deployments into map
    # TODO: Model this
    raw = {dep["metadata"]["name"]: dep for dep in raw}

    patchlst = {}
    # FIXME: NB: app-wide settings not supported

    cfg = data.get("control", {})

    # FIXME: off-spec; step-down in data if a 'state' key is provided at the top.
    if "state" in data:
        data = data["state"]

    # TODO: Traverse model objects
    for comp_name, comp_data in (
        data.get("application", {}).get("components", {}).items()
    ):
        settings = comp_data.get("settings", {})
        if not settings:
            continue
        patches = {}
        replicas = None
        comp_desc = desc["application"]["components"].get(comp_name) or {}

        # find deployment name and container name, and verify it's existence
        cont_name = None
        dep_name = comp_name
        if "/" in dep_name:
            dep_name, cont_name = comp_name.split("/", 1)
        if dep_name not in raw:
            raise AdjustError(
                'Cannot find deployment with name "{}" for component "{}" in namespace "{}"'
                + "".format(dep_name, comp_name, appname),
                status="failed",
                reason="unknown",
            )  # FIXME 'reason' code (either bad config or invalid input to update())
        cont_name = (
            cont_name
            or raw[dep_name]["spec"]["template"]["spec"]["containers"][0]["name"]
        )  # chk for KeyError FIXME
        available_conts = set(
            c["name"] for c in raw[dep_name]["spec"]["template"]["spec"]["containers"]
        )
        if cont_name not in available_conts:
            raise AdjustError(
                'Could not find container with name "{}" in deployment "{}" '
                'for component "{}" in namespace "{}".'.format(
                    cont_name, dep_name, comp_name, appname
                ),
                status="failed",
                reason="unknown",
            )  # see note above

        cont_patch = patches.setdefault(cont_name, {})

        command = comp_desc.get("command")
        if command:
            if command.get("encoder"):
                cont_patch["command"], encoded_settings = encode_encoder(
                    settings, command["encoder"], expected_type=list
                )

                # Prevent encoded settings from further processing
                for setting in encoded_settings:
                    del settings[setting]

        env = comp_desc.get("env")
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
        for name, value in settings.items():
            value = _value(
                value
            )  # compatibility: allow a scalar, but also work with {"value": {anything}}
            cont_patch = patches.setdefault(cont_name, {})
            if name in ("mem", "cpu"):
                set_rsrc(
                    cont_patch,
                    name,
                    value,
                    comp_desc.get("settings", {}).get(name, {}).get("selector", "both"),
                )
                continue
            elif name == "replicas":
                replicas = int(value)

        patch = patchlst.setdefault(dep_name, {})
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
            patch_r = k_patch(appname, DEPLOYMENT, n, patchstr)
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
            wait_for_update(
                appname,
                n,
                patch_r["metadata"]["generation"],
                print_progress,
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
    testdata0, raw = raw_query(appname, desc)
    settlement_time = cfg.get("settlement", 0)
    mon0 = testdata0["monitoring"]

    if "ref_version_id" in mon0 and mon0["version_id"] != mon0["ref_version_id"]:
        raise AdjustError(
            "application version does not match reference version",
            status="aborted",
            reason="version-mismatch",
        )

    # aborted status reasons that aren't supported: ref-app-inconsistent, ref-app-unavailable

    if not settlement_time:
        return {"monitoring": mon0, "status": "ok", "reason": "success"}

    # TODO: adjust progress accounting when there is settlement_time!=0

    # wait and watch the app, checking for changes
    w = Waiter(
        settlement_time, delay=min(settlement_time, 30)
    )  # NOTE: delay between tests may be made longer than the delay between progress reports
    while w.wait():
        testdata, raw = raw_query(appname, desc)
        mon = testdata["monitoring"]
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


class KubernetesConfiguration(connector.BaseConfiguration):
    namespace: Optional[str]

    @classmethod
    def generate(cls, **kwargs) -> "KubernetesConfiguration":
        return cls(
            namespace="default",
            description="Update the namespace, deployment, etc. to match your Kubernetes cluster",
            **kwargs
        )

    class Config:
        # We are the base root of pluggable configuration
        # so we ignore any extra fields so you can turn connectors on and off
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
    progress: float = 0.0

    def print_progress(
        self, message=None, msg_index=None, stage=None, stageprogress=None
    ):

        data = dict(
            progress=self.progress,
            message=message if (message is not None) else self.progress_message,
        )

        if msg_index is not None:
            data["msg_index"] = msg_index
        if stage is not None:
            data["stage"] = stage
        if stageprogress is not None:
            data["stageprogress"] = stageprogress

        from devtools import pformat

        # print(json.dumps(data), flush=True)
        self.logger.info(pformat(json))
        # Schedule the next progress upd

    def _progress(self, progress, message):
        """adapter for the default base class implementation of progress message"""
        self.progress = progress
        self.print_progress(message=message)

    @on_event()
    def describe(self) -> Description:
        try:
            desc = read_desc()
        except ConfigError as e:
            raise AdjustError(
                str(e), reason="unknown"
            )  # maybe we should introduce reason=config (or even a different status class, instead of 'failed')
        desc.pop("driver", None)
        #        namespace = os.environ.get('OPTUNE_NAMESPACE', desc.get('namespace', self.app_id))
        # TODO: Replace the namespace
        namespace = "default"
        result = query(namespace, desc)
        components = descriptor_to_components(result["application"]["components"])
        return Description(components=components)

    @on_event()
    def components(self) -> Description:
        desc = read_desc()
        return descriptor_to_components(desc["application"]["components"])

    @on_event()
    def adjust(self, data: dict) -> dict:
        try:
            desc = read_desc()
            # debug(desc)
        except ConfigError as e:
            raise AdjustError(
                str(e), reason="unknown"
            )  # maybe we should introduce reason=config (or even a different status class, instead of 'failed')
        # all other exceptions: default handler - stack trace and sys.exit(1)
        # namespace = os.environ.get('OPTUNE_NAMESPACE', desc.get('namespace', self.app_id))
        namespace = "default"
        r = update(namespace, desc, data, self._progress)
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


def descriptor_to_components(descriptor: dict) -> List[Component]:
    components = []
    for component_name in descriptor:
        settings = []
        for setting_name in descriptor[component_name]["settings"]:
            setting_values = descriptor[component_name]["settings"][setting_name]
            # FIXME: Temporary hack
            if not setting_values.get("type", None):
                setting_values["type"] = "range"
            setting = Setting(name=setting_name, **setting_values)
            settings.append(setting)
        component = Component(name=component_name, settings=settings)
        components.append(component)
    return components
