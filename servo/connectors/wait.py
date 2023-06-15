# Copyright 2023 Cisco Systems, Inc. and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import servo


class WaitConfiguration(servo.BaseConfiguration):
    measure_enabled: bool

    @classmethod
    def generate(cls, **kwargs) -> "WaitConfiguration":
        return cls(measure_enabled=False)


@servo.metadata(
    description="Wait (no-op) Connector for Opsani",
    version="0.0.1",
    homepage="https://github.com/opsani/servox",
    license=servo.License.apache2,
    maturity=servo.Maturity.stable,
)
class WaitConnector(servo.BaseConnector):
    """
    The wait connector is a dummy no-op connector which supports externally driven operation (currently measure only)
    by waiting for the specified duration and reporting progress in the same fashion as any other connector without
    interacting with any external systems
    """

    config: WaitConfiguration

    @servo.on_event()
    def describe(self, control: servo.Control = servo.Control()) -> servo.Description:
        return servo.Description(metrics=[])

    @servo.on_event()
    async def measure(
        self, *, metrics: list[str] = None, control: servo.Control = servo.Control()
    ) -> servo.Measurement:
        if self.config.measure_enabled:
            measurement_duration = servo.Duration(
                control.warmup + control.duration + control.delay
            )
            self.logger.info(
                f"Waiting for {measurement_duration} for external measurement to complete"
            )

            progress = servo.EventProgress(
                timeout=measurement_duration, settlement=None
            )
            await progress.watch(self.log_progress)

        return servo.Measurement(readings=[])

    async def log_progress(self, progress: servo.EventProgress) -> None:
        """Report a progress message (heart beat) to the CO api with percent of completion (also logs to stdout)"""
        return self.logger.info(
            progress.annotate(
                f"Waiting for {progress.timeout} for external measurement to complete",
                False,
            ),
            progress=progress.progress,
        )
