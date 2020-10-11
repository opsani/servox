import socket
import aioprometheus
import servo
from typing import List

service: aioprometheus.Service = aioprometheus.Service()
measure_counter: aioprometheus.Counter = aioprometheus.Counter(
    "measurements", "Number of events.", const_labels={"host": socket.gethostname()}
)
adjust_counter: aioprometheus.Counter = aioprometheus.Counter(
    "adjusts", "Number of events.", const_labels={"host": socket.gethostname()}
)
service.register(measure_counter)
service.register(adjust_counter)

@servo.metadata(
    description="Tracks metrics for servo events.",
    version="0.5.0",
    homepage="https://github.com/opsani/servox",
    license=servo.License.APACHE2,
    maturity=servo.Maturity.EXPERIMENTAL,
)
class MetricsConnector(servo.BaseConnector):
    @servo.on_event()
    async def startup(self) -> None:
        await service.start(addr="127.0.0.1")
        self.logger.info(f"serving prometheus metrics on: {service.metrics_url}")

    @servo.on_event()
    async def shutdown(self) -> None:
        await service.stop()

    @servo.after_event("measure")
    async def after_measure(self, results: List[servo.EventResult], **kwargs) -> None:
        self.logger.info("after_measure")
        measure_counter.inc({"servo": self.optimizer.id})

    @servo.after_event("adjust")
    async def after_adjust(self, results: List[servo.EventResult]) -> None:
        self.logger.info("after_adjust")
        adjust_counter.inc({"servo": self.optimizer.id})

    class Config:
        arbitrary_types_allowed = True
