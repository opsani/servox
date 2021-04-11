import asyncio
import enum
import json
import math
import pprint
import time
from collections import defaultdict
from statistics import mean

import plyvel

import servo
from servo.connectors.olas import ast_formula
from servo.connectors.olas import client
from servo.connectors.olas import configuration
from servo.connectors.olas import envoy_stats
from servo.connectors.olas import eventqueue
from servo.connectors.olas import kube
from servo.connectors.olas import prometheus
from servo.connectors.olas import server_classes
from servo.connectors.olas.fastpath import Fastpath

ERR_ENVOY_NOT_RD = "pod not ready"
ERR_ENVOY_NO_URL = "no metrics url"

DEFAULT_CPU_COST = 0.0175
DEFAULT_MEM_COST = 0.0125    # per GiB


async def get_pod_envoy_stats(p):
    if not kube.is_pod_ready(p):
        return None, ERR_ENVOY_NOT_RD
    url = kube.get_envoy_url(p)

    return await envoy_stats.get_envoy_stats(url)


class mode(enum.Enum):
    CPU_DETECT  = enum.auto()
    CAP_MONITOR = enum.auto()
    CPU_PREDICT = enum.auto()
    CPU_TARGET  = enum.auto()
    RATE_TARGET = enum.auto()
    HPA         = enum.auto()
    NOOP        = enum.auto()


class PrometheusPlugin:
    must_metrics = [
        'rq_total',
        'rq_time_bucket',
        'rq_error',
    ]

    def __init__(self, plugincfg):
        self.query = []
        self.extra_metrics = []
        self.extra = {}
        self.parse_config(plugincfg)

    def parse_config(self, config):
        self.query = []
        self.config = configuration.PrometheusConfig.parse_obj(config.config)
        syms = {}

        for m in config.metrics:
            syms[m.name] = m.query
        for name in self.must_metrics:
            self.query.append(("query", dict(query=syms[name])))
            del syms[name]
        for name in syms:
            self.query.append(("query", dict(query=syms[name])))
            self.extra_metrics.append(name)
        servo.logger.info(f'Prometheus query {self.query}')
        servo.logger.info(f'Prometheus extra query name {" ".join(self.extra_metrics)}')


class ContourPlugin:
    def __init__(self, cfg):
        self.cfg = cfg
        for m in cfg.metrics:
            if m.name == 'envoy_cluster_name':
                self.envoy_cluster_name = m.query
                return
        msg = 'Contour plugin: expect metrics name "envoy_cluster_name" not found'
        raise ValueError(msg)


class OLASController:
    def __init__(self, cfg_dict, backend_dict):
        self.config_dict = cfg_dict
        self.client = client.OLASClient(backend_dict['url'],
                                        backend_dict['account'],
                                        backend_dict['app_id'],
                                        backend_dict['auth_token'])
        self.base_ts = self.ts = time.time()

    def boot(self, upload_cfg=True):
        servo.logger.info(f"olas boot cfg {self.config_dict}")
        self.upload_cfg = upload_cfg

        self.cfg = cfg = configuration.OLASConfiguration.parse_obj(self.config_dict)

        self.time_slo = self.find_name_in_values("rq_time", self.cfg.objectives.metrics)
        cpu_slo_percent = self.find_name_in_values("cpu", self.cfg.objectives.metrics)
        if self.time_slo is None or cpu_slo_percent is None:
            msg = "Either 'rq_time' or 'cpu' is not configured under " +\
                  "metrics key of objective field of ConfigMap."
            raise ValueError(msg)

        if not cfg.scaleTargetRef.kind or cfg.scaleTargetRef.kind == 'Deployment':
            self.target_api = []
        else:
            # custom target scale object
            self.target_api = cfg.scaleTargetRef.apiVersion.split('/')

        self.cpu_slo = cpu_slo_percent / 100.0
        servo.logger.info(f"time slo: {self.time_slo} cpu slo {self.cpu_slo}")

        self.model_id = 0
        self.ps = None
        if self.cfg.config.devel:
            self.logfd = open("olas.txt", "a")
        self.output_array = []
        self.count: int = 0
        self.last_control_ts = 0.0
        self.stats_pod_rate = {}
        self.pod_buckets = {}
        self.total_rate = 0
        self.cap = 0
        self.pod_ts = {}
        self.cpu = 0
        self.target = 0
        self.cpu_history = []
        self.cpu_window = []  # transitional cpu observation window.
        self.cpu_window_size = self.cfg.config.sloWindow  # window size
        self.cpu_target = 0   # cpu_target is a value between 0 - 1
        servo.logger.info(f"mode: {self.cfg.config.mode}")
        self.mode = getattr(mode, self.cfg.config.mode.upper())
        self.exlabels = kube.parse_selector(self.cfg.scaleTargetRef.exclude)
        self.last_violation = None
        self.mode_table = {
            mode.CPU_DETECT: self.scale_by_cpu_slo,
            mode.CAP_MONITOR: self.scale_by_capacity_monitor,
            mode.CPU_PREDICT: self.scale_by_cpu_predict,
            mode.CPU_TARGET: self.scale_by_cpu,
            mode.RATE_TARGET: self.scale_by_rate,
            mode.HPA: self.scale_by_hpa_observe_only,
            mode.NOOP: self.scale_by_noop,
        }
        self.cached_result = {}
        if upload_cfg:
            self.db = plyvel.DB("metrics", create_if_missing=True)
            self.cached_key = self.get_last_cached_key()
        self.traffic_history = defaultdict(list)
        if self.cfg.config.resolution < 5:
            self.fastpath_resolution = self.cfg.config.resolution
        else:
            self.fastpath_resolution = 5
        self.prometheus = None
        self.contour = None
        self.reload_mtime = None
        self.last_node_ts = 0
        self.ms_sync_gap = 0
        self.load_plugins()
        self.deployment = ''
        self.service_selector = ''
        self.deployments = []
        self.excluded_dps = []
        self.pod_cost = 0

    async def get_deployment_name(self):
        ref = self.cfg.scaleTargetRef
        return ref.name if not ref.service else await self.get_deployment_by_service(ref.service, ref.namespace)

    async def get_service_label_selector(self, service, ns):
        svc = await kube.get_service(service, ns)
        if not svc:
            servo.logger.error(f"No service {service} found")
            return ''
        selector = svc.spec.selector
        if not selector:
            servo.logger.error(f"Service {service} no selector found")
            return ''
        return ",".join([f'{k}={v}' for k, v in selector.items()])

    async def get_deployment_by_service(self, service, namespace):
        # lookup deployment from service spec.
        self.service_selector = selector = await self.get_service_label_selector(service, namespace)
        if not selector:
            servo.logger.error(f"Service {service} has no selector")
            return ''
        if self.target_api:
            dps = await kube.list_custom_object_by_template_selector(self.target_api, self.cfg.scaleTargetRef.kind.lower() + 's', selector, namespace)
            self.excluded_dps = [d['metadata']['name'] for d in dps
                                 if self.exlabels
                                 and 'labels' in d['spec']['template']['metadata']
                                 and self.exlabels.items() <= d['spec']['template']['metadata']['labels'].items()]
            self.deployments = [d['metadata']['name'] for d in dps if d['metadata']['name'] not in self.excluded_dps]
        else:
            dps = await kube.list_deployment_by_template_selector(selector, namespace)
            self.excluded_dps = [d.metadata.name for d in dps
                                 if self.exlabels
                                 and self.exlabels.items() <= d.spec.template.metadata.labels.items()]
            self.deployments = [d.metadata.name for d in dps if d.metadata.name not in self.excluded_dps]

        servo.logger.info(f"get deployment by selector {self.service_selector} return {self.deployments}"
                          f" excluding {self.excluded_dps} by selector {self.cfg.scaleTargetRef.exclude}")
        if len(self.deployments) != 1:
            servo.logger.error(f"Service {service} has {len(self.deployments)} deployments, expect 1")
            return ''
        return self.deployments[0]

    async def get_pods(self):
        pods = []
        replicas = uid = None
        ref = self.cfg.scaleTargetRef
        self.deployment = name = await self.get_deployment_name()
        if name:
            if not self.target_api:
                obj = await kube.get_deployment(name, ref.namespace)
                self.replicas = replicas = obj.spec.replicas
                uid = obj.metadata.uid
            else:
                obj = await kube.get_namespaced_custom_object(*self.target_api, ref.namespace, ref.kind.lower() + 's', name)
                self.replicas = replicas = obj['spec']['replicas']
                uid = obj['metadata']['uid']
                # servo.logger.info(f'get obj {pprint.pformat(obj)}')

        if not ref.service:
            if uid:
                pods = await kube.get_pods_own_by_uid(uid, ref.namespace)
        elif self.service_selector:
            pods = await kube.list_pod_by_selector(self.service_selector, ref.namespace)

        if not replicas:
            self.replicas = len(pods)
        return pods

    def get_last_cached_key(self):
        with self.db.iterator(reverse=True, include_value=False) as it:
            for k in it:
                servo.logger.info(f"last key is {int(k)}")
                return int(k)
        return 0

    def find_name_in_values(self, name, values):
        for value in values:
            if value.name == name:
                return value.value

    def load_plugins(self):
        for plugin in self.cfg.metricSource:
            if plugin.plugin == 'prometheus':
                self.prometheus = PrometheusPlugin(plugin)
            elif plugin.plugin == 'contour':
                return
                self.contour = ContourPlugin(plugin)

    async def get_kube_node_metrics(self, pods):
        count = 0
        nodes = set(p.spec.node_name for p in pods)
        nodes = [n for n in nodes if n]
        while 1:
            nm_cache = {}
            nms = [await kube.get_node_metrics(n, nm_cache) for n in nodes]
            ts_list = [kube.parse_node_timestamp(nm.timestamp) for nm in nms if nm]
            if ts_list and min(ts_list) > self.last_node_ts:
                last_ts = max(ts_list)
                if count:
                    # Observe the update just happen. Update the time sync value
                    ts = time.time()
                    self.ms_sync_gap = ts - last_ts
                    servo.logger.info(f'Update metric server time sync gap {self.ms_sync_gap}')
                self.last_node_ts = last_ts
                return nm_cache
            count += 1

    async def sleep_loop_interval(self):
        sleep = self.last_node_ts + 60 + self.ms_sync_gap - time.time()
        if sleep > 0 and sleep < 65:
            servo.logger.info(f'sleeping {sleep} sync {self.ms_sync_gap:.1f}')
            await asyncio.sleep(sleep)
        self.ts = time.time()

    def stats_get_rate(self, stats, name, default_interval):
        prev_ts, prev_total = self.stats_pod_rate.get(name, (0, 0))
        rate, total = stats.get_rate('envoy_cluster_upstream_rq_total',
                                     {'envoy_cluster_name': 'service'},
                                     self.ts - prev_ts if prev_ts else default_interval,
                                     prev_total)
        self.stats_pod_rate[name] = (self.ts, total)
        return rate

    def stats_get_hist(self, stats, name):
        prev_buckets = self.pod_buckets.get(name, [])
        his, bucket = stats.get_histogram('envoy_cluster_upstream_rq_time_bucket', {}, prev_buckets)
        self.pod_buckets[name] = bucket
        return his

    async def collect_prometheus_by_url(self, m, url):
        p = prometheus.PrometheusQuery(url)
        prom = self.prometheus
        res = await p.querys(prom.query)
        values = p.extract_vector_list(res)
        total, raw_buckets, errors = values[:len(prom.must_metrics)]
        total_rate = p.merge_rate(total)
        his = prometheus.Histogram(p.merge_buckets(raw_buckets))
        rq_time = his.get_pn(prom.config.rq_time_histogram_quantile)

        syms = dict(total_rate=total_rate, rq_time=rq_time, errors=errors)
        extra = ''
        for n, v in zip(prom.extra_metrics, values[len(prom.must_metrics):]):
            syms[n] = v
            extra += f" {n} {v:.1f}"

        servo.logger.info(f'\tProm  his {his}\ttotal_rate {total_rate:.1f}\trq_time {rq_time:.1f} error {errors} extra {extra}')
        m.metrics.update(syms)
        return syms

    async def collect_contour(self, m):
        envoys = await kube.list_pod_by_selector('app=envoy', 'projectcontour')
        total_rate = p50 = p90 = 0
        merge_bucket = his = None
        for p in envoys:
            name = p.metadata.name
            stats, err = await get_pod_envoy_stats(p)
            # servo.logger.info(f'{stats} {name}')
            prev_ts, prev_total = self.stats_pod_rate.get(name, (0, 0))
            rate, total = stats.get_rate('envoy_cluster_upstream_rq_total',
                                         {'envoy_cluster_name': self.contour.envoy_cluster_name},
                                         self.ts - prev_ts if prev_ts else 60,
                                         prev_total)
            self.stats_pod_rate[name] = (self.ts, total)
            # servo.logger.info(f'{name} {total}')
            total_rate += rate

            prev_buckets = self.pod_buckets.get(name)
            bucket, total = stats.get_delta_bucket('envoy_cluster_upstream_rq_time_bucket', {'envoy_cluster_name': self.contour.envoy_cluster_name}, prev_buckets)
            self.pod_buckets[name] = total
            # servo.logger.info(f'{name} {bucket}')

            if not merge_bucket:
                merge_bucket = bucket
            else:
                merge_bucket = stats.merge_bucket(merge_bucket, bucket)
        if merge_bucket:
            his = envoy_stats.Histogram(merge_bucket)
            p50 = his.get_pn(50)
            p90 = his.get_pn(90)

        servo.logger.info(f'\tEnvoy his {his}\ttotal_rate {total_rate:.1f}\tp90 {p90:.1f} p50 {p50:.1f}')
        syms = dict(total_rate=total_rate, p50=p50, p90=p90)
        m.metrics.update(syms)
        return syms

    def is_pod_warmup(self, name):
        ts = self.pod_ts.get(name, self.ts)
        return self.ts - ts < self.cfg.config.warmUpDelay * 60

    def has_node_metrics(self):
        return self.mode in [mode.CAP_MONITOR]

    async def collect_pod_metrics(self, p, m, nm_cache, allocation_map, nodetype_map, node_map, cpus):
        if not kube.is_pod_ready(p):
            return
        name = p.metadata.name
        podusage = await kube.get_pod_usage(name, p.metadata.namespace)
        if not podusage:
            return

        cpu = podusage['cpu']
        allocation = kube.get_pod_allocation(p)
        key = tuple(allocation.values())
        if key not in allocation_map:
            allocation_map[key] = len(m.allocations)
            m.allocations.append(server_classes.Allocation.parse_obj(allocation))
        alloc_index = allocation_map[key]

        self.pod_ts[name] = p.status.start_time.timestamp()

        if not self.is_pod_warmup(name):
            cpus.append(cpu)

        node_index = 0
        if self.has_node_metrics():
            nodename = p.spec.node_name
            if nodename not in node_map:
                node = await kube.read_node(nodename)
                ec2type = kube.get_node_instance_type(node)

                if ec2type not in nodetype_map:
                    nodetype_map[ec2type] = len(nodetype_map)
                    m.nodetypes.append(server_classes.NodeType.parse_obj(kube.get_node_info(node)))
                nodetype_index = nodetype_map[ec2type]
                nodeusage = kube.get_node_usage(nodename, nm_cache)
                nodealloc = kube.get_node_allocation(nodename)
                node_map[nodename] = len(m.nodes)
                m.nodes.append(server_classes.NodeMetrics(name=nodename, nodetype=nodetype_index,
                                                          usage=nodeusage, allocation=nodealloc)
                               )
            node_index = node_map[nodename]

        return server_classes.PodMetrics(name=name, metrics=podusage,
                                         allocation=alloc_index, node=node_index)

    async def collect_envoy_sidecar(p, self, m, pod_metrics):
        name = p.metadata.name
        stats, err = await get_pod_envoy_stats(p)
        if err:
            servo.logger.error("Error fetcthing pod %s envoy metric: %s", name, err)
            return
        # servo.logger.info(f'{name} {stats}')
        rate = self.stats_get_rate(stats, name, self.ts - self.last_control_ts)
        his = self.stats_get_hist(stats, name)
        p90 = his.get_pn(90) if his else 0
        p50 = his.get_pn(50) if his else 0
        syms = dict(rate=rate, p90=p90, p50=p50, rq_time=p90)
        pod_metrics.metrics.update(syms)
        return syms

    async def collect_metrics(self, pods):
        m = server_classes.Metrics(ts=self.ts, deployments=self.deployments,
                                   excluded_deployments=self.excluded_dps)
        m.replicas = len(pods)
        exclude = [p for p in pods
                   if self.exlabels
                   and self.exlabels.items() <= p.metadata.labels.items()]
        pods = [p for p in pods if p not in exclude]
        nm_cache = await self.get_kube_node_metrics(pods)
        if not self.deployment:
            servo.logger.info("Can't resolve deployment.")

        if self.contour:
            contour_syms = await self.collect_contour(m)
        if self.prometheus:
            url = self.prometheus.config.url
            prom_syms = await self.collect_prometheus_by_url(m, url)
        if self.cfg.config.envoyContour:
            assert False  # XXX not implemented.
            m.metrics.update(contour_syms)

        self.pod_ts = {}
        allocation_map = {}     # tuple(allocation.values()) -> index
        nodetype_map = {}       # ec2 name -> index
        node_map = {}
        cpus = []

        for p in pods + exclude:
            pod_metrics = await self.collect_pod_metrics(p, m, nm_cache, allocation_map, nodetype_map, node_map, cpus)
            if not pod_metrics:
                continue
            if p not in exclude:
                m.pods.append(pod_metrics)
            else:
                m.excluded_pods.append(pod_metrics)
            if self.cfg.config.envoySideCar:
                await self.collect_envoy_sidecar(p, m, pod_metrics)
        m.metrics['total_cpu'] = sum(cpus) if cpus else math.nan
        m.metrics['cpu'] = mean(cpus) if cpus else math.nan
        self.load_metrics(m)
        servo.logger.info(f'Collected metrics {pprint.pformat(m.dict())}')
        return m

    def load_metrics(self, m):
        # TODO: aggrigate pod metrics for envoy_sidecar
        symbols = m.metrics
        self.total_rate = symbols['total_rate']
        self.rq_time = symbols['rq_time']
        self.p50 = symbols['p50']
        self.cpu = symbols['cpu']
        self.total_cpu = symbols['total_cpu']
        cpu_requests = [a.cpu_request for a in m.allocations]
        self.cpu_request = cpu_requests[0] if cpu_requests else 0.0
        if not self.cpu_request:
            servo.logger.error("Please define cpu request in target deployment.")
        mem_requests = [a.memory_request for a in m.allocations]
        self.mem_request = mem_requests[0] if mem_requests else 0
        if not self.mem_request:
            servo.logger.error("Please define memory request in target deployment.")

    def caculate_pod_cost(self, metrics):
        expr = self.cfg.config.costFormula
        if not expr:
            expr = 'cpu * DEFAULT_CPU_COST + mem * DEFAULT_MEM_COST'

        if not metrics.allocations:
            return None

        allocation = metrics.allocations[0]
        cpu = allocation.cpu_request

        memory = allocation.memory_request / (1024 * 1024 * 1024)
        syms = dict(cpu=cpu, mem=memory, DEFAULT_CPU_COST=DEFAULT_CPU_COST, DEFAULT_MEM_COST=DEFAULT_MEM_COST)
        try:
            cost = ast_formula.evaluate(expr, syms)
        except Exception as e:
            servo.logger.info(f"Failed to calculate pod cost using '{expr}' with '{syms}': {e}")
            return None

        servo.logger.info(f"Calculated pod cost: {cost} using '{expr}' with '{syms}'")
        return cost

    def add_scale_request(self, desired, reason, raw=False):
        self.scale_request.append((desired, raw, reason))

    def scale_pods_by_ratio(self, scale, reason):
        desired = scale * self.replicas
        reason = f"{reason} scale {scale:.3f} desired {desired:.2f} roundup {math.ceil(desired)}"
        if scale < 1 + self.cfg.config.tolerance and scale > 1 - self.cfg.config.tolerance:
            reason += f" skip by tolerance {self.cfg.config.tolerance:.1%}"
            desired = self.replicas
        self.add_scale_request(desired, reason)

    def constrain_desired_scale(self, desired):
        if self.pod_cost and self.cfg.objectives.maxCost:
            max_replicas = int(self.cfg.objectives.maxCost / self.pod_cost)
            if desired > max_replicas:
                servo.logger.info(f"Trying to scale to {desired} instances, constrained by cost to {max_replicas}")
                desired = max_replicas
        if self.cfg.objectives.maxCPU and self.cpu_request:
            max_replicas = int(self.cfg.objectives.maxCPU / self.cpu_request)
            if desired > max_replicas:
                servo.logger.info(f"Trying to scale to {desired} instances, constrained by maxCPU to {max_replicas}")
                desired = max_replicas

        maxMem = self.cfg.objectives.maxMem and kube.parse_size_value(self.cfg.objectives.maxMem)
        if maxMem and self.mem_request:
            max_replicas = int(maxMem / self.mem_request)
            if desired > max_replicas:
                servo.logger.info(f"Trying to scale to {desired} instances, constrained by maxMem to {max_replicas}")
                desired = max_replicas
        if self.cfg.objectives.maxReplicas != 0 and desired > self.cfg.objectives.maxReplicas:
            servo.logger.info(f"Trying to scale to {desired} instances, constrained by maxReplicas to {self.cfg.objectives.maxReplicas}")
            desired = self.cfg.objectives.maxReplicas
        if desired < self.cfg.objectives.minReplicas:
            servo.logger.info(f"Trying to scale to {desired} instances, constrained by minReplicas to {self.cfg.objectives.minReplicas}")
            desired = self.cfg.objectives.minReplicas
        return desired

    async def output_scale_with_constrain(self, desired, raw=False):
        if not desired:
            return
        # servo.logger.info(f"scale number: {desired}")
        desired = self.constrain_desired_scale(int(math.ceil(desired)))
        cooldown = self.cfg.config.coolDown
        if cooldown > 30:
            cooldown += 15
        ts = self.ts - cooldown
        while self.output_array and self.output_array[0][0] < ts:
            self.output_array.pop(0)
        self.output_array.append((self.ts, desired))
        servo.logger.info(f"out_array {[x [1] for x in self.output_array]}")
        if not raw:
            desired = max([desired for ts, desired in self.output_array])

        if desired > self.replicas and self.cpu_request and (self.cpu / self.cpu_request) < (self.cfg.objectives.minScaleCPU / 100):
            servo.logger.info(f"Trying to scale from {self.replicas} to {desired}, constrainged by minScaleCPU {self.cpu / self.cpu_request:.1%}")
            desired = self.replicas

        if desired != self.replicas and self.deployment:
            servo.logger.info(f"\tscaling from {self.replicas} to {desired} ==============")
            await self.output_scale(desired)
            self.replicas = desired

    async def output_scale(self, desired):
        target = self.cfg.scaleTargetRef
        if not self.target_api:
            return await kube.scale_deployment(self.deployment, desired, ns=target.namespace)
        obj = await kube.get_namespaced_custom_object_scale(*self.target_api, target.namespace, target.kind.lower() + 's', self.deployment)
        if not obj:
            servo.logger.error(f"Invalid scale object {obj} for kind {target.kind}, apiVersion {target.apiVersion}")
            return
        # servo.logger.info(f"Get deployment like object scale {obj}")
        obj['spec']['replicas'] = desired
        res = await kube.patch_namespaced_custom_object_scale(*self.target_api, target.namespace, target.kind.lower() + 's', self.deployment, obj)
        # servo.logger.info(f"Patch output scale res {res}")

    def scale_by_cpu(self, metrics):
        if math.isnan(self.cpu) or not self.cpu_request:
            return
        scale = (self.cpu / self.cpu_request) / self.cpu_target
        reason = f"cpu: {self.cpu:.1%} target {self.target:.1f}"
        self.scale_pods_by_ratio(scale, reason)

    def get_predict(self, predict, current):
        if self.cfg.config.enablePrediction and predict and predict > current:
            return predict
        return current

    def scale_by_rate(self, metrics):
        total = self.get_predict(self.total_predict, self.total_rate)
        rate = total / self.replicas
        scale = rate / self.target
        reason = f"rate: {rate:.1f} target {self.target:.1f}"
        self.scale_pods_by_ratio(scale, reason)

    def scale_by_cpu_predict(self, metrics):
        if math.isnan(self.cpu) or not self.cpu_request:
            return
        self.check_slo_violation()

        if not self.cfg.config.enablePrediction:
            return self.scale_by_cpu_slo(metrics)

        total_cpu = self.get_predict(self.total_predict, self.total_cpu)
        if self.total_predict:
            self.target = 0.75 * self.cpu_request
        else:
            self.target = 0.6 * self.cpu_request
        cpu = total_cpu / self.replicas
        scale = cpu / self.target
        reason = f"cpu_p: cpu {cpu :.3f} target {self.target:.3f}"
        self.scale_pods_by_ratio(scale, reason)

    def scale_by_capacity_monitor(self, metrics):
        if math.isnan(self.cpu):
            return
        if self.ps is None:
            return self.scale_by_cpu_slo(metrics)
        cap, err = self.ps.detect(metrics, self)
        if err and not self.cap:
            servo.logger.info("Unable to detect pod capacity. PodSensor Exception: %s", err)
            return self.scale_by_cpu_slo(metrics)
        elif err:
            servo.logger.info("Use prior detected pod capacity. PodSensor Exception: %s", err)
        else:
            self.cap = cap
        self.target = self.cap * .85
        servo.logger.info(f"cap-qps {self.cap:.1f} new target {self.target}")
        if not self.check_slo_violation():
            self.scale_by_rate(metrics)

    def scale_by_cpu_slo(self, metrics):
        # cpu, p90, pod_rate):
        if math.isnan(self.cpu):
            return
        servo.logger.info(f"scale_by_cpu_slo, cpu {self.cpu:.1%} p90 {self.rq_time}, pod_rate {self.pod_rate}"
                          f" mode {self.mode} cpu_target {self.cpu_target} {self.cpu_history[-2:]}")

        violation = self.check_slo_violation()
        if not violation:

            level = self.cpu_history[-1][0] if self.cpu_history else 0
            better = [(cpu, p90, rate) for cpu, p90, rate in self.cpu_window
                      if p90 < self.time_slo / 3 and cpu > level]
            servo.logger.info(f"level {level} better {better}")
            if len(better) == self.cpu_window_size:
                better.sort()
                self.cpu_history.append(better[0])
                servo.logger.info(f"Adding good cpu sample {better[0]}")

            if self.last_violation != violation:
                servo.logger.info("SLO recovered.")
                if self.cpu_history and self.cpu_request:
                    self.cpu_target = self.cpu_history[-1][0] * .85 / self.cpu_request
                    servo.logger.info(f"Detect cpu target {self.cpu_target}")
        self.last_violation = violation

        if self.cpu_target:
            self.scale_by_cpu(metrics)
        else:
            servo.logger.info(f"No cpu_target. Pass this round, mode {self.mode} cpu_target {self.cpu_target}")

    async def scale_by_hpa_observe_only(self, metrics):
        if not self.deployment:
            servo.logger.info("deployment not found, abort hpa observe")
            return

        status = await kube.read_autoscaler_status(self.deployment, self.cfg.scaleTargetRef.namespace)
        hpacpu = status and status.status and status.status.current_cpu_utilization_percentage
        servo.logger.info(f"HPA cpu {hpacpu}")

    def scale_by_noop(self, metrics):
        pass

    def check_slo_violation(self):
        self.cpu_window.append((self.cpu, self.rq_time, self.pod_rate))
        while len(self.cpu_window) > self.cpu_window_size:
            self.cpu_window.pop(0)

        violations = [p90 for cpu, p90, rate in self.cpu_window if p90 > self.time_slo]

        reason = ''
        if len(violations) >= self.cpu_window_size:
            reason += " slo window"
        if self.rq_time > self.time_slo * 10:
            reason += f" 10x {self.rq_time/self.time_slo:.1f}"
        if self.cpu_request and self.cpu > self.cpu_slo * self.cpu_request:
            reason += f" cpu > {self.cpu_slo}"
        if reason:
            scale_num = self.replicas * 1.15
            self.add_scale_request(scale_num, reason)
            return True

    async def cached_upload(self, name, obj):
        '''self.cached_upload(name,obj).
            Name is the name of the object.
            Object is dictionary type'''

        if obj:
            self.cached_key += 1
            key = str(self.cached_key).zfill(20).encode()
            pack = json.dumps((name, obj)).encode()
            servo.logger.info(f'Putting {name} at {self.cached_key}  {len(pack)} bytes')
            self.db.put(key, pack)
        with self.db.iterator() as it:
            for key, data in it:
                k = int(key)
                n, obj = json.loads(data)
                servo.logger.info(f"Uploading {n} at {k} {len(data)} bytes")
                ret = await getattr(self.client, f'upload_{n}')(obj)
                if not ret:
                    # something is wrong in the API call.
                    servo.logger.error(f"Upload {k} {n} failed {ret}. Skip this round.")
                    return ret
                self.cached_result[n] = ret
                self.db.delete(key)
        return self.cached_result.get(name)

    async def upload_metrics(self, metrics):
        # input metrics are dict() type.
        model_id = await self.cached_upload("metrics", metrics)
        # TODO: enable CAP_MONITOR mode
        # if self.mode.name == 'CAP_MONITOR' and model_id and model_id.id > self.model_id:
        #     servo.logger.info(f"Fetching new model {model_id.id} from the back end.")
        #     model = self.client.get_pod_model()
        #     servo.logger.info(f"Fetched model_id {model.model_id} model size {len(model.model)}")
        #     if model and model.model_id > self.model_id:
        #         servo.logger.info(f'Updating new model "{model.model[:10]}..." from the back end.')
        #         try:
        #             self.ps = PodSensor(self.time_slo, model.model)
        #         except Exception as e:
        #             servo.logger.info(f"Initialize model id: {model.model_id} exception {e}")
        #         else:
        #             self.model_id = model.model_id

    async def try_disable_hpa(self):
        if self.mode in (mode.HPA, mode.NOOP) or not self.deployment:
            return
        try:
            hpa = await kube.read_autoscaler(self.deployment, self.cfg.scaleTargetRef.namespace)
        except Exception:
            return

        if not hpa:
            return
        name = hpa.spec.scale_target_ref.name
        if not name.endswith("-disabled-by-olas"):
            servo.logger.info(f"Found HPA {hpa.spec}")
            hpa.spec.scale_target_ref.name = name + "-disabled-by-olas"
            servo.logger.info(f"Disabling HPA {hpa.spec.scale_target_ref}")
            await kube.patch_autoscaler(self.deployment, self.cfg.scaleTargetRef.namespace, hpa)

    async def try_predict(self, datum, source):
        # local traffic buffer
        self.traffic_history[source].append(dict(ts=self.ts, total=datum))

        if self.cfg.config.enablePrediction:
            total_predict = await self.client.predict(source)
            if total_predict is None:
                try:
                    # instantiate Fastpath object with recent traffic data from traffic buffer
                    standby = Fastpath(self.traffic_history[source],
                                       self.fastpath_resolution,
                                       self.cfg.config.fastpathWindow)
                    # predict on the latest traffic data
                    total_predict = standby.predict()

                    # discard the earliest record in traffic buffer
                    self.traffic_history[source].pop(0)
                except Exception as err:
                    servo.logger.info("Fastpath is not invoked.")
                    servo.logger.error(err)
            elif ((self.traffic_history[source][-1]['ts'] - self.traffic_history[source][0]['ts'])
                    >= 4.5 * self.cfg.config.fastpathWindow * (60 * self.fastpath_resolution)):
                # discard the earliest record in traffic buffer
                self.traffic_history[source].pop(0)
            return total_predict

    async def control_with_metrics(self, metrics):
        if self.cfg.config.disableHPA:
            await self.try_disable_hpa()

        self.scale_request = []  # list of (scale_number, raw, reason)

        numpods = len(metrics.pods)
        self.replicas = metrics.replicas

        if not numpods:
            return

        await self.upload_metrics(metrics.dict())

        # use current per replica traffic as default for prediction, e.g.,
        # predictive feature is not enabled or predictive models are not
        # functioning due to either outage or insufficient data
        self.pod_rate = self.total_rate / numpods

        self.total_predict = 0
        if self.mode.name in ('CAP_MONITOR', 'RATE_TARGET'):
            self.total_predict = await self.try_predict(self.total_rate, 'rate')
        elif self.mode.name == 'CPU_PREDICT':
            self.total_predict = await self.try_predict(self.total_cpu, 'cpu')

        stats = (f"ts {self.ts - self.base_ts:.1f} cpu {self.cpu/self.cpu_request if self.cpu_request else self.cpu:.1%}"
                 f" p90 {self.rq_time:.1f} total_rate {self.total_rate:.1f} pod_rate {self.pod_rate:.1f}"
                 f" prediction {self.total_predict/self.replicas if self.total_predict else 0:.1f} "
                 f" target {self.target:.1f} pods {numpods} request replicas {self.replicas}"
                 )
        servo.logger.info(stats)
        if self.cfg.config.devel:
            self.logfd.write(f"{self.ts-self.base_ts:.0f} {self.cpu:.1%} {self.pod_rate/self.cpu if self.cpu else math.nan:.1f} {self.total_rate:.1f} {self.pod_rate:.1f} {self.total_predict/self.replicas if self.total_predict else 0:.1f}"
                             f" {self.target:.1f} {self.p50:.1f} {self.rq_time:.1f} {numpods} {self.replicas}\n")
            self.logfd.flush()

        fn = self.mode_table[self.mode]
        if asyncio.iscoroutinefunction(fn):
            await fn(metrics)
        else:
            fn(metrics)

        if self.cfg.objectives.maxCost:
            self.pod_cost = self.caculate_pod_cost(metrics)
        servo.logger.info(f"scaler action {self.scale_request}")
        if self.scale_request:
            self.scale_request.sort()
            replicas, raw, reason = self.scale_request[-1]
            await self.output_scale_with_constrain(replicas, raw)
        self.last_control_ts = self.ts

    async def process_event_task(self):
        queue = self.eventqueue
        while 1:
            event_type, metrics = await queue.get()
            servo.logger.info("Process event {metrics}")
            await self.control_with_metrics(metrics)

    async def control_task(self):

        self.boot()
        self.eventqueue = eventqueue.eventQueue.subscribe_event("OLASMetrics")
        asyncio.create_task(self.process_event_task())

        if self.upload_cfg:
            await self.cached_upload("config", self.cfg.dict())

        if self.cfg.config.devel:
            text = "ts cpu cpucap total-qps pod-qps prediction target p50 p90 n replicas"
            print(text, file=self.logfd)

        pods = await self.get_pods()
        await self.collect_metrics(pods)

        while 1:
            await self.sleep_loop_interval()
            if self.cfg.config.devel:
                self.try_reload()
            pods = await self.get_pods()
            metrics = await self.collect_metrics(pods)
            eventqueue.eventQueue.put_event("OLASMetrics", metrics)

    def try_reload(self):
        import importlib
        import os
        import sys
        mtime = os.path.getmtime("olas/controller.py")
        if self.reload_mtime and mtime != self.reload_mtime:
            servo.logger.info(f"=====Reloading module controller classid: {id(self.__class__)} =======")
            module = sys.modules['controller']
            ctl = importlib.reload(module)
            self.__class__ = ctl.OLASController
            self.mode_table[mode.CPU_PREDICT] = getattr(self, "scale_by_cpu_predict")
            servo.logger.info(f"{module} new classid: {id(self.__class__)}")

            importlib.reload(prometheus)
            importlib.reload(envoy_stats)
            importlib.reload(client)
            importlib.reload(kube)
        self.reload_mtime = mtime
