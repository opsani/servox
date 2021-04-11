import json
import re
import time
from typing import List

import kubernetes_asyncio
from kubernetes_asyncio.client import CustomObjectsApi
from kubernetes_asyncio.client.api_client import ApiClient
from pydantic import BaseModel, parse_obj_as

import servo


class ContainerUsage(BaseModel):
    cpu: str
    memory: str


class ContainerMetrics(BaseModel):
    name: str
    usage: ContainerUsage


class KubePodMetrics(BaseModel):
    apiVersion: str = ''
    kind: str = ''
    containers: List[ContainerMetrics]
    metadata: dict
    timestamp: str = ''
    window: str = ''


class KubeNodeMetrics(BaseModel):
    apiVersion: str = ''
    kind: str = ''
    metadata: dict
    timestamp: str
    window: str
    usage: dict


class NodeInfo(BaseModel):
    ec2type: str
    cpu_allocatable: float
    memory_allocatable: float
    cpu_capacity: float
    memory_capacity: float


class Usage(BaseModel):
    cpu: float
    memory: float


class Allocation(BaseModel):
    cpu_request: float = 0
    cpu_limit: float = 0
    memory_request: float = 0
    memory_limit: float = 0


async def get_deployment(deploy, ns='default'):
    async with ApiClient() as api:
        appsv1 = kubernetes_asyncio.client.AppsV1Api(api)
        return await appsv1.read_namespaced_deployment(deploy, ns)


async def get_pod(podname, ns='default'):
    async with ApiClient() as api:
        v1 = kubernetes_asyncio.client.CoreV1Api(api)
        return await v1.read_namespaced_pod(podname, ns)


async def get_service(svcname, ns='default'):
    async with ApiClient() as api:
        apis_api = kubernetes_asyncio.client.CoreV1Api(api)
        return await apis_api.read_namespaced_service(svcname, ns)


def has_owner(obj, *owner_uids):
    owners = obj.metadata.owner_references
    if owners:
        for o in owners:
            if o.uid in owner_uids:
                return o.uid


async def get_rs_own_by_uid(uid, ns='default'):
    async with ApiClient() as api:
        apis_api = kubernetes_asyncio.client.AppsV1Api(api)
        resp = await apis_api.list_namespaced_replica_set(ns)
        return [rs for rs in resp.items if has_owner(rs, uid)]


async def get_replica_set(rs, ns='default'):
    async with ApiClient() as api:
        apis_api = kubernetes_asyncio.client.AppsV1Api(api)
        return await apis_api.read_namespaced_replica_set(rs, ns)


async def get_pods_own_by_uid(uid, ns='default'):
    async with ApiClient() as api:
        rslist = await get_rs_own_by_uid(uid, ns)
        # print("rslist", rslist)
        rs_uids = [r.metadata.uid for r in rslist]
        v1 = kubernetes_asyncio.client.CoreV1Api(api)
        ret = await v1.list_namespaced_pod(ns, watch=False)
        return [i for i in ret.items if has_owner(i, *rs_uids)]


async def list_deployment_by_selector(selector, namespace='default'):
    async with ApiClient() as api:
        apis = kubernetes_asyncio.client.AppsV1Api(api)
        body = await apis.list_namespaced_deployment(namespace=namespace, label_selector=selector)
        return body.items


def parse_selector(selector):
    if not selector:
        return {}
    return dict([tuple(assign.split('=', 1)) for assign in selector.split(',')])


async def list_deployment_by_template_selector(selector, namespace='default'):
    async with ApiClient() as api:
        labels = parse_selector(selector)
        apis = kubernetes_asyncio.client.AppsV1Api(api)
        body = await apis.list_namespaced_deployment(namespace=namespace)
        # python 3 can use dict view to test subset
        return [d for d in body.items if labels.items() <= d.spec.template.metadata.labels.items()]


async def list_custom_object_by_template_selector(api, plural, selector, namespace='default'):
    objs = list_namespaced_custom_object(*api, namespace, plural)
    labels = parse_selector(selector)
    # python 3 can use dict view to test subset
    return [o for o in objs
            if 'labels' in o['spec']['template']['metadata']
            and labels.items() <= o['spec']['template']['metadata']['labels'].items()]


async def list_deployment_by_field_selector(selector, namespace='default'):
    async with ApiClient() as api:
        apis = kubernetes_asyncio.client.AppsV1Api(api)
        body = await apis.list_namespaced_deployment(namespace=namespace, filed_selector=selector)
        return body.items


async def scale_deployment(deploy, replica, ns='default'):
    async with ApiClient() as api:
        apis_api = kubernetes_asyncio.client.AppsV1Api(api)
        body = await apis_api.read_namespaced_deployment_scale(deploy, ns)
        body.spec.replicas = replica
        await apis_api.patch_namespaced_deployment_scale(deploy, ns, body)


async def read_deployment_scale(deploy, ns='default'):
    async with ApiClient() as api:
        apis_api = kubernetes_asyncio.client.AppsV1Api(api)
        body = await apis_api.read_namespaced_deployment_scale(deploy, ns)
        return body.spec.replicas


async def list_pod_all_namespaces():
    async with ApiClient() as api:
        # XXX: handle continue read()
        apis_api = kubernetes_asyncio.client.CoreV1Api(api)
        body = await apis_api.list_pod_for_all_namespaces()
        return body.items


async def list_pod_in_node(nodename):
    # XXX: handle continue read()
    async with ApiClient() as api:
        v1 = kubernetes_asyncio.client.CoreV1Api(api)
        selector = 'spec.nodeName=%s,status.phase!=Failed,status.phase!=Succeeded' % nodename
        body = await v1.list_pod_for_all_namespaces(field_selector=selector)
        return body.items


async def list_pod_by_selector(selector, namespace='default'):
    async with ApiClient() as api:
        v1 = kubernetes_asyncio.client.CoreV1Api(api)
        body = await v1.list_namespaced_pod(namespace=namespace, label_selector=selector)
        return body.items


async def read_node(node_name):
    async with ApiClient() as api:
        v1 = kubernetes_asyncio.client.CoreV1Api(api)
        body = await v1.read_node(node_name)
        return body


def get_node_allocatable(node):
    return parse_cpu_value(node.status.allocatable['cpu'])


def get_node_instance_type(node):
    labels = node.metadata.labels
    if not labels:
        return ''
    return labels.get('beta.kubernetes_asyncio.io/instance-type', '')


def get_pod_cpu_allocation_limit(pod):
    return sum(parse_cpu_value(container.resources.limits['cpu'])
               for container in pod.spec.containers
               if container.resources.limits is not None
               and 'cpu' in container.resources.limits)


def get_pod_cpu_allocation_request(pod):
    return sum(parse_cpu_value(container.resources.requests['cpu'])
               for container in pod.spec.containers
               if container.resources.requests is not None
               and 'cpu' in container.resources.requests)


def get_pod_memory_allocation_limit(pod):
    return sum(parse_size_value(container.resources.limits['memory'])
               for container in pod.spec.containers
               if container.resources.limits is not None
               and 'memory' in container.resources.limits)


def get_pod_allocation(pod):
    cpu_request = get_pod_cpu_allocation_request(pod)
    cpu_limit = get_pod_cpu_allocation_limit(pod)
    mem_request = get_pod_memory_allocation_request(pod)
    mem_limit = get_pod_memory_allocation_limit(pod)

    return {'cpu_request': cpu_request, 'cpu_limit': cpu_limit,
            'memory_request': mem_request, 'memory_limit': mem_limit}


def get_pod_memory_allocation_request(pod):
    return sum(parse_size_value(container.resources.requests['memory'])
               for container in pod.spec.containers
               if container.resources.limits is not None
               and 'memory' in container.resources.requests)


def get_deployment_cpu_allocation_limit(dp):
    return sum(parse_cpu_value(container.resources.limits['cpu'])
               for container in dp.spec.template.spec.containers
               if container.resources.limits is not None
               and 'cpu' in container.resources.limits)


def get_deployment_cpu_allocation_request(dp):
    return sum(parse_cpu_value(container.resources.requests['cpu'])
               for container in dp.spec.template.spec.containers
               if container.resources.requests is not None
               and 'cpu' in container.resources.requests)


def get_deployment_memory_allocation_limit(dp):
    return sum(parse_size_value(container.resources.limits['memory'])
               for container in dp.spec.template.spec.containers
               if container.resources.limits is not None
               and 'memory' in container.resources.limits)


def get_deployment_memory_allocation_request(dp):
    return sum(parse_size_value(container.resources.requests['memory'])
               for container in dp.spec.template.spec.containers
               if container.resources.requests is not None
               and 'memory' in container.resources.requests)


def get_deployment_allocation(dp):
    cpu_request = get_deployment_cpu_allocation_request(dp)
    cpu_limit = get_deployment_cpu_allocation_limit(dp)
    mem_request = get_deployment_memory_allocation_request(dp)
    mem_limit = get_deployment_memory_allocation_limit(dp)

    return {'cpu_request': cpu_request, 'cpu_limit': cpu_limit,
            'memory_request': mem_request, 'memory_limit': mem_limit}


def get_node_cpu_allocated_limit(nodename, pods):
    pods = [get_pod_cpu_allocation_limit(po)
            for po in pods if is_pod_ready(po)]
    return sum(pods) if pods else 0


def get_node_cpu_allocated_request(nodename, pods):
    pods = [get_pod_cpu_allocation_request(po)
            for po in pods if is_pod_ready(po)]
    return sum(pods) if pods else 0


def get_node_memory_allocated_limit(nodename, pods):
    pods = [get_pod_memory_allocation_limit(po)
            for po in pods if is_pod_ready(po)]
    return sum(pods) if pods else 0


def get_node_memory_allocated_request(nodename, pods):
    pods = [get_pod_memory_allocation_request(po)
            for po in pods if is_pod_ready(po)]
    return sum(pods) if pods else 0


def get_node_allocation(nodename):
    pods = list_pod_in_node(nodename)
    cpu_r = get_node_cpu_allocated_request(nodename, pods)
    cpu_l = get_node_cpu_allocated_limit(nodename, pods)
    mem_r = get_node_memory_allocated_request(nodename, pods)
    mem_l = get_node_memory_allocated_limit(nodename, pods)
    return {'cpu_request': cpu_r, 'cpu_limit': cpu_l,
            'memory_request': mem_r, 'memory_limit': mem_l}


def parse_node_timestamp(ts):
    return time.mktime(time.strptime(ts, "%Y-%m-%dT%H:%M:%SZ"))


async def list_prometheus_scrape_pod():
    # XXX: handle continue read()
    return [p for p in await list_pod_all_namespaces()
            if p.metadata.annotations.get("prometheus.io/scrape") == 'true']


async def read_autoscaler_status(name, ns='default'):
    async with ApiClient() as api:
        apis_api = kubernetes_asyncio.client.AutoscalingV1Api(api)
        body = await apis_api.read_namespaced_horizontal_pod_autoscaler_status(name, ns)
        return body


async def read_autoscaler(name, ns='default'):
    async with ApiClient() as api:
        apis_api = kubernetes_asyncio.client.AutoscalingV1Api(api)
        body = await apis_api.read_namespaced_horizontal_pod_autoscaler(name, ns)
        return body


async def patch_autoscaler(name, ns, hpa):
    async with ApiClient() as api:
        apis_api = kubernetes_asyncio.client.AutoscalingV1Api(api)
        ret = await apis_api.patch_namespaced_horizontal_pod_autoscaler(name, ns, body=hpa)
        return ret


def is_pod_ready(pod):
    if pod.metadata.deletion_timestamp:
        # schedule for deletion.
        return False
    status = pod.status.container_statuses
    if not status:
        return False
    for c in status:
        if not (c.ready and c.state.running):
            return False
    return True


def get_pod_node(pod):
    # use pod.spec.node_name.split('.')[0] to obtain private IP address only
    return pod.spec.node_name


async def custom_api_call(url, method='GET', auth_settings=['BearerToken'], response_type='json', _preload_content=False, **kwargs):
    async with ApiClient() as api:
        return await kubernetes_asyncio.client.ApiClient(api) \
            .call_api(url, method, auth_settings=auth_settings,
                      response_type=response_type, _preload_content=_preload_content, **kwargs)


async def json_api_call(url, method='GET', response_type='json', **kwargs):
    response, code, header = await custom_api_call(url, method, response_type, **kwargs)
    j = json.loads(response.data)
    return j, code, header


def parse_cpu_value(value):
    m = re.split(r'([mun])', value)
    v = float(m[0])
    if m[1:]:
        unit = {'n': 0.000000001,
                'u': 0.000001,
                'm': 0.001}
        factor = unit[m[1]]
        v *= factor
    return v


def parse_size_value(value):
    m = re.split(r'([KMGTm]i?)', value)
    v = float(m[0])
    if m[1:]:
        unit = {'Ki': 1024,
                'K': 1000,
                'Mi': 1048576,
                'M': 1000000,
                'Gi': 1073741824,
                'G': 1000000000,
                'Ti': 1099511627776,
                'T': 1000000000000,
                'm': 0.001,
                }
        factor = unit[m[1]]
        v *= factor
    return v


def get_pm_cpu(pm, container):
    if pm is None:
        return float('nan')
    for c in pm.containers:
        if c.name == container:
            return parse_cpu_value(c.usage.cpu)
    return float('nan')


async def get_cluster_custom_object(*args, **kwargs):
    async with ApiClient() as api:
        cust = CustomObjectsApi(api)
        try:
            return await cust.get_cluster_custom_object(*args, **kwargs)
        except Exception:
            servo.logger.exception(f"Exception in get_cluster_custom_object({args},{kwargs})", exc_info=True)


async def get_namespaced_custom_object(*args, **kwargs):
    async with ApiClient() as api:
        cust = CustomObjectsApi(api)
        try:
            return await cust.get_namespaced_custom_object(*args, **kwargs)
        except Exception:
            servo.logger.exception(f"Exception in get_namespaced_custom_object({args},{kwargs})", exc_info=True)


async def list_namespaced_custom_object(*args, **kwargs):
    async with ApiClient() as api:
        cust = CustomObjectsApi(api)
        try:
            body = await cust.list_namespaced_custom_object(*args, **kwargs)
            return body['items']
        except Exception:
            servo.logger.exception(f"Exception in get_namespaced_custom_object({args},{kwargs})", exc_info=True)


async def patch_namespaced_custom_object(*args, **kwargs):
    async with ApiClient() as api:
        cust = CustomObjectsApi(api)
        try:
            return await cust.patch_namespaced_custom_object(*args, **kwargs)
        except Exception:
            servo.logger.exception(f"Exception in patch_namespaced_custom_object({args},{kwargs})", exc_info=True)


async def get_namespaced_custom_object_scale(*args, **kwargs):
    async with ApiClient() as api:
        cust = CustomObjectsApi(api)
        try:
            return await cust.get_namespaced_custom_object_scale(*args, **kwargs)
        except Exception:
            servo.logger.exception(f"Exception in get_namespaced_custom_object_scale({args},{kwargs})", exc_info=True)


async def patch_namespaced_custom_object_scale(*args, **kwargs):
    async with ApiClient() as api:
        cust = CustomObjectsApi(api)
        try:
            return await cust.patch_namespaced_custom_object_scale(*args, **kwargs)
        except Exception:
            servo.logger.exception(f"Exception in patch_namespaced_custom_object_scale({args},{kwargs})", exc_info=True)


async def get_node_metrics(nodename, cache={}):
    if not nodename:
        return
    if nodename in cache:
        return cache[nodename]
    nm = await get_cluster_custom_object('metrics.k8s.io', 'v1beta1', 'nodes', nodename)
    cache[nodename] = x = nm and KubeNodeMetrics.parse_obj(nm)
    return x


async def list_all_node_metrics():
    async with ApiClient() as api:
        cust = CustomObjectsApi(api)
        nodes = await cust.list_cluster_custom_object('metrics.k8s.io', 'v1beta1', 'nodes')
        return parse_obj_as(List[KubeNodeMetrics], nodes['items']) if nodes else []


async def get_node_usage(nodename, nm_cache={}):
    nm = await get_node_metrics(nodename, nm_cache)
    return nm and parse_usage(nm.usage)


def parse_usage(usage):
    # pprint.pprint(usage)
    cpu = parse_cpu_value(usage['cpu'])
    memory = parse_size_value(usage['memory'])
    return dict(cpu=cpu, memory=memory)


async def get_pod_metrics(podname, ns):
    pm = await get_namespaced_custom_object('metrics.k8s.io', 'v1beta1', ns, 'pods', podname)
    if pm:
        if not pm['timestamp']:
            pm['timestamp'] = ''
        # pprint.pprint(pm)
        return KubePodMetrics.parse_obj(pm)


async def get_pod_usage(podname, ns):
    pm = await get_pod_metrics(podname, ns)
    if pm:
        cpu = sum([parse_cpu_value(c.usage.cpu) for c in pm.containers])
        memory = sum([parse_size_value(c.usage.memory) for c in pm.containers])
        return dict(cpu=cpu, memory=memory)


def get_node_info(node):
    ec2type = get_node_instance_type(node)
    allocatable = node.status.allocatable
    capacity = node.status.capacity
    cpu_a = parse_cpu_value(allocatable['cpu'])
    mem_a = parse_size_value(allocatable['memory'])
    cpu_c = parse_cpu_value(capacity['cpu'])
    mem_c = parse_size_value(capacity['memory'])
    return dict(name=ec2type, cpu_allocatable=cpu_a, memory_allocatable=mem_a,
                cpu_capacity=cpu_c, memory_capacity=mem_c)


def get_envoy_url(pod):
    ann = pod.metadata.annotations
    # print(pod.metadata.name, ann)
    path = ann.get('prometheus.io/path')
    if not path:
        return ''
#    if path.startswith('/stats/'):
#        path = '/stats'
    port = ann.get('prometheus.io/port', '')
    if port:
        port = ":" + port
    url = "http://%s%s%s" % (pod.status.pod_ip, port, path)
    return url
