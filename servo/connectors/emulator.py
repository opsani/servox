import random
from typing import List, Union
import servo

METRICS = [
    servo.Metric("throughput", servo.Unit.REQUESTS_PER_MINUTE),
    servo.Metric("error_rate", servo.Unit.PERCENTAGE),
]

COMPONENTS = [
    servo.Component(
        "fake-app",
        settings=[
            servo.CPU(
                min=1,
                max=5
            ),
            servo.Memory(
                min=0.25,
                max=8.0,
                step=0.125
            ),
            servo.Replicas(
                min=1,
                max=10
            )
        ]
    )
]

@servo.metadata(
    description="An emulator that pretends take measurements and make adjustments.",
    version="0.5.0",
    homepage="https://github.com/opsani/servox",
    license=servo.License.APACHE2,
    maturity=servo.Maturity.EXPERIMENTAL,
)
class EmulatorConnector(servo.BaseConnector):
    @servo.on_event()
    async def describe(self) -> servo.Description:
        components = await self.components()
        metrics = await self.metrics()

        components_ = await self.components()
        for component_ in components_:
            for setting in component_.settings:
                setting.value = _random_value_for_setting(setting)

        return servo.Description(metrics=metrics, components=components)

    @servo.on_event()
    async def metrics(self) -> List[servo.Metric]:
        return METRICS.copy()

    @servo.on_event()
    async def components(self) -> List[servo.Component]:
        return COMPONENTS.copy()

    @servo.on_event()
    async def measure(
        self, *, metrics: List[str] = None, control: servo.Control = servo.Control()
    ) -> servo.Measurement:
        wait_duration = _random_duration()
        progress = servo.DurationProgress(wait_duration)
        notifier = lambda p: self.logger.info(
            p.annotate(f"sleeping for {wait_duration} to simulate measurement aggregation...", False),
            progress=p.progress,
        )
        await progress.watch(notifier)

        metrics_ = await self.metrics()
        readings = list(map(lambda m: servo.DataPoint(metric=m, value=random.uniform(10, 2000)), metrics_))

        return servo.Measurement(readings=readings)

    @servo.on_event()
    async def adjust(
        self, adjustments: List[servo.Adjustment], control: servo.Control = servo.Control()
    ) -> servo.Description:
        wait_duration = _random_duration()
        progress = servo.DurationProgress(wait_duration)
        notifier = lambda p: self.logger.info(
            p.annotate(f"sleeping for {wait_duration} to simulate adjustment rollout...", False),
            progress=p.progress,
        )
        await progress.watch(notifier)

        components_ = await self.components()
        for component_ in components_:
            for setting in component_.settings:
                setting.value = _random_value_for_setting(setting)

        return servo.Description(components=components_)


def _random_duration() -> servo.Duration:
    seconds = random.randrange(30, 600)
    return servo.Duration(seconds=seconds)

def _random_value_for_setting(setting: servo.Setting) -> Union[str, servo.Numeric]:
    if isinstance(setting, servo.RangeSetting):
        max = int((setting.max - setting.min) / setting.step)
        return random.randint(0, max) * setting.step + setting.min
    elif isinstance(setting, servo.EnumSetting):
        return random.choice(setting.values)
    else:
        raise ValueError(f"unexpected setting: {repr(setting)}")
