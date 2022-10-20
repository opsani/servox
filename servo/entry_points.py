# Copyright 2022 Cisco Systems, Inc. and/or its affiliates.
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

# Entry points for executing functionality defined in other modules
# This wrapping is necessary for keeping Docker builds fast because
# the entry points must be present on the filesystem during package
# installation in order to be registered on the path and fast moving
# sources will unnecessarily bust the Docker cache and trigger full
# reinstalls of all package dependencies.
# Do not implement meaningful functionality here. Instead import and
# dispatch the intent into focused modules to do the real work.
# noqa

import dotenv
import uvloop

import servo
import servo.cli


def run_cli() -> None:
    """Run the Servo CLI."""
    uvloop.install()
    dotenv.load_dotenv(dotenv.find_dotenv(usecwd=True))

    # NOTE: We load connectors here because waiting until assembly
    # is too late for registering CLI commands
    try:
        for connector in servo.connector.ConnectorLoader().load():
            servo.logger.debug(f"Loaded {connector.__qualname__}")
    except Exception:
        servo.logger.exception(
            "failed loading connectors via discovery", backtrace=True, diagnose=True
        )

    cli = servo.cli.ServoCLI()
    cli()
