from __future__ import annotations
import enum
import os
from servo.events import on_event
from typing import List, Optional, Tuple
import servo
from servo import connector, configuration, checks
from servo.connectors import kubernetes, prometheus
from servo import Check, Filter, Severity

class Formations(str, enum.Enum):
    opsani_dev = "opsani_dev"

class DevFormationChecks(checks.BaseChecks):
    @checks.warn("Prometheus sidecar")
    async def check_prometheus_sidecar(self) -> Tuple[bool, str]:
        if not os.environ.get("KUBERNETES_SERVICE_HOST", False):
            return False, "Not running under Kubernetes"

        # Read our own Pod
        pod_name = os.environ.get("POD_NAME", None)
        pod_namespace = os.environ.get("POD_NAMESPACE", None)
        if pod_name and pod_namespace:
            pod = await kubernetes.Pod.read(pod_name, pod_namespace)
            container = pod.get_container("prometheus")
            if container:
                return True, f"Found Prometheus sidecar running {container.obj.image} in Pod {pod_name}"
        else:
            return False, f"No Prometheus sidecar found in Pod {pod_name}"

    # TODO: Dispatch metrics and check the names/queries
    # @checks.multicheck("Prometheus queries")
    # async def check_metrics(self) -> None:
    #     ...

    # TODO: Fetch and look at the Prometheus targets
    @checks.check("Envoy proxies are being scraped")
    async def check_envoy_sidecar_metrics(self) -> str:
        # TODO: Ask Prometheus? Get its config or do I ask it to do something
        ...

    # TODO: Find the Envoy proxies and make sure they are all reporting metrics

    # TODO: Cycle the canary up and down and make sure that it gets traffic
    # # What we may want to do is run an adjust and then re-run all the checks.
    # # Actually we can just bring up the canary and then re-check...
    # @check("New canary Pods receive traffic")
    # async def check_pod_load_balancing(self) -> str:
    #     ...

class FormationConfiguration(servo.AbstractBaseConfiguration):
    __root__: Optional[Formations] = None

    # TODO: Add a generate command

@connector.metadata(
    description="Run connectors in a specific formation",
    version="0.0.1",
    homepage="https://github.com/opsani/servox",
    license=connector.License.APACHE2,
    maturity=connector.Maturity.EXPERIMENTAL,
)
class FormationConnector(connector.BaseConnector):
    config: FormationConfiguration

    @servo.on_event()
    async def check(
        self,
        filter_: Optional[Filter],
        halt_on: Optional[Severity] = checks.Severity.critical
    ) -> List[Check]:
        return await DevFormationChecks.run(servo.BaseConfiguration(), filter_=filter_, halt_on=halt_on)

#     # TODO: require kubernetes -- can we depend on other connectors?
#     # TODO: inspect the prometheus targets
#     # TODO: dispatch event to get metrics from Prometheus, check thresholds???