# CHANGELOG

Servo is an Open Source framework supporting Continuous Optimization of infrastructure
and applications via the Opsani optimization engine. Opsani provides a software as a
service platform that optimizes the resourcing and configuration of cloud native
applications to reduce operational costs and increase performance. Servo instances are
responsible for connecting the optimizer service with the application under optimization
by linking with the metrics system (e.g. Prometheus, Thanos, DataDog, etc) and the
orchestration system (e.g. Kubernetes, CloudFormation, etc) in order to apply changes
and evaluate their impact on cost and performance.

Servo is distributed under the terms of the Apache 2.0 license.

This changelog catalogs all notable changes made to the project. The format
is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/). Releases are
versioned in accordance with [Semantic
Versioning](https://semver.org/spec/v2.0.0.html).

## [0.8.0] "pass the calimari" - Unreleased

### Added
- `servo.__codename__` constant now contains the release codename.
- Extensive docstring comments for all members of the `servo.types` module.
- Kubernetes Containers can be aliased to set explicit Compomnent names rather
  rather than automatically deriving names from Deployment/Container.
- Kubernetes Optimization Strategy classes can now accept options from the
  config file (currently supports `alias` for canaries).

### Removed
- Subprocess methods have been removed from `servo.connector.Connector` in
  favor of directly importing the subprocess module from the utilities module.

### Changed
- The `servo.logging` module has been generalized for use outside of the
  `servo.connectors.Connector` inheritance hierarchy.
- The active connector is now managed via a `ContextVar` just as the active
  event is. This enables logging to correctly be attributed to the active
  connector without having to pass a specific logger object around everywhere.
- The `servo.types.Setting` class has been significantly overhauled:
  - Setting is now an abstract base class
  - RangeSetting models range settings
  - EnumSetting models enum settings
  - Introduce CPU, Memory, Replicas, and InstanceType settings for special
    optimizer settings
  - Validate numerous behaviors (range inclusion, enum inclusion, type
    agreement, etc)


### Fixed
- Progress tracking now handled zero length durations appropriately (e.g., in warmup, settlement, etc).

## [0.7.0] "nice for what?" - 2020-09-09

### Added
- `servo run --check` can now be controlled via the `SERVO_RUN_CHECK`
environment variable.
- The `servo.logging` module now provides the `set_colors` function for programmatically enabling or disabling coloring.
- The CLI accepts a `—no-colors` argument that will explicitly disable coloring.
- The `SERVO_NO_COLOR` and `NO_COLOR` environment variables are respected to
  disable coloring.
- The API URL can be now be overridden via the hidden `--url` CLI option or the `OPSANI_URL` environment variable.
- Introduce the `multicheck` decorator for use in checks implementations. A 
  multicheck is method that returns an iterable collection of checkable objects
  and a `CheckHandler` callable that can evaluate each item. Each item in the 
  iterable collection is wrapped into an individual check and run independently.
  This provides a simple mechanism for checking configurations that have a mix
  of settings that need to be handled specifically and homogenous collections
  that can be handled iteratively. The generated checks are filterable and fully
  integrated with the CLI.
- Checks and multichecks now support templated string inputs. The `self` and
  `item` arguments are made available as format variables, enabling the names
  and descriptions given to the decorators to produce dynamic, contextual values
  from the configuration. This enhances the readability and diagnostic context
  of the checks output.
- The Prometheus connector now exposes a rich set of checks.
- The Prometheus connector now accepts an optional list of targets that are
  expected to be scraped.

### Changed
- Log coloring is now conditionally enabled via TTY auto-detection.

### Fixed
- Handle measure command responses that include metric units (`oco-e`
  compatibility).
- Prometheus can now connect to localhost URLs.
- The `get_instance_methods` utility function now returns instance methods that
  are bound to a specific object instance

## [0.6.2] - 2020-09-03

### Changed
- Switched Docker base image to `python:3.8-slim`.

## [0.6.1] - 2020-09-03

### Enhanced
- Logging when connecting via a proxy.

### Fixed
- Handled null annotations and labels when cloning a Deployment in order to create a Pod.
- Servo runner now honors proxy settings (previously only honored within connectors).
- `servo check servo` now works as expected (previously not handled as a connector name).

### Changed
- Updated to httpx v0.14.3.

## [0.6.0] "woke up like this" - 2020-08-30

### Enhanced
- The checks subsystem has been rearchitected to streamline the development of
checks and provide a better operational experience for running and debugging checks.

### Added
- `servo run --check` will run all configured checks before starting the servo runloop.
- `servo check` now supports filtering by name, id, and tags. Failure mode handling is configurable via `--halt-on-failed=[requirement,check,never]`
- `servo run` now applies exponential backoff and retry to recover from transient HTTP errors.
- Introduce associations mixin for managing off-model support objects that don't make sense to model as attributes.
- `ServoConfiguration` class for applying settings to the servo itself.
- HTTP proxy support (configured on `ServoConfiguration`).
- Timeout configuration (configured on `ServoConfiguration`).
- Support for configuring backoff and retry behaviors (configured on `ServoConfiguration`).
- Baseline set of checks on the Kubernetes Connector.

### Fixed
- Attempting to connect to an invalid or unavailable optimizer backend now triggers exponential backoff and retry rather than crashing.
- Encountering an unexpected event error from the optimizer now aborts the operation in progress and resets rather than waiting for the operation to complete.

## [0.5.1] - 2020-08-23

### Removed
- Migrated servo-webhooks into a standalone project (see https://github.com/opsani/servo-webhooks)

## [0.5.0] - 2020-08-22

### Added
- The capabilities of the logging module have been significantly enhanced with supporting for
changing the logging levels, performing timed logging around methods, and attaching backtraces.
- Log messages are now annotated with the event context when logging occurs during an event
handler.
- The metadata decorator now accepts a tuple for the `name` parameter for providing an explicit
default name for connectors of the decorated type, overriding the name inference.
- Progress can be automatically reported to the Opsani API by annotating log messages with a "progress" key.
- Introduced general utilities for executing subprocesses including streaming output, timeouts, etc.
- Added integration testing infrastructure for testing the servo in Docker, Minikube, and EKS.
- Integrated complete implementation of Kubernetes connector on top of kubernetes_asyncio.

### Changed
- Enable parsing of extended Golang duration strings in `Duration` (days, weeks, months, years).
- The base connector class has been renamed from `Connector` to `BaseConnector` to align with other inheritance patterns in the library.
- Redesigned the Kuberenetes configuration to adopt familiar naming conventions and structure from Kubernetes.

## [0.4.2] - 2020-07-24

### Fixed
- The init command output formatting was broken when accepting defaults.

## [0.4.1] - 2020-07-24

### Fixed
- The `respx` mocking library was in the main project dependencies instead of dev dependencies.

## [0.4.0] - 2020-07-24

The headline feature of this release is the adoption of asyncio
as a core part of the architecture.

### Fixed
- Docker image builds now correctly support excluding development dependencies via the `SERVO_ENV` build argument.

### Added
- Docker images are now published to Docker Hub with README content from `docs/README-DOCKERHUB.md`.
- Introduced Duration class for modeling time durations as Golang style duration strings.
- BaseConfiguration now provides a `yaml` method for easily to YAML.
- BaseConfiguration now provides a `json_encoders` static method for easily accessing the default encoders.
- The Vegeta connector now validates the `target` and `targets` settings.
- The check command now supports a verbose and non-verbose modes for outputting multiple checks from a connector.
- The version command can now output version data for connectors.
- The connectors command now outputs names for displaying aliased configurations.
- The servo now runs asynchronously on top of asyncio.
- Added start-up banner to run command.

### Removed
- The `durationpy` package has been removed in favor of a local implementation.

### Changed
- Updated Docker and Docker Compose configurations to use `/servo/opsani.token` as the default path for API tokens mounted as a file.
- Docker images pushed to Docker Hub are now built with `SERVO_ENV=production` to exclude development packages.
- The reporting interval is now configurable `VegetaConfiguration`.
- The check event now returns a list of checks instead of one result.
- Removed the 'Connector` suffix from the default connector name attribute and introduced `full_name`.
- The `__config_key__` attribute has been renamed to `__default_name__`.
- The identifier for connectors has been renamed to `name` for simplicitly and clarity.
- The `ServoAssembly` class has been renamed to `Assembly`.

## [0.3.1] - 2020-07-16

### Fixed
- Eliminated secondary connectors package to eliminate Docker & PyPI distribution issues.

## [0.3.0] - 2020-07-15

### Added
- Config files can be outputted in Kubernetes ConfigMap format (`servo config -f configmap`).
- All Connectors can now dispatch events (previously only available to the `Servo` class).
- `Optimizer` now includes an `api_url` property.
- Event accessor `Connector.events` and `Connector.get_event`.
- Project automation via GitHub Actions (Dependabot, release-drafter, PyPI release, Docker builds).

### Removed
- Removed `ServoAssembly.default_routes` from the API (usage eliminated).

### Changed
- Normalized naming of miscellaneous config methods.
- Renamed `ServoAssembly.all_connectors` to `ServoAssembly.all_connector_types` for clarity.

### Fixed
- Pydantic will no longer see self-references between connectors (avoids recursion sharp edge).

## [0.2.5] - 2020-07-13

### Fixed
- Dotenv was non-functional in PyPI release package.

## [0.2.1] - 2020-07-13

### Fixed
- Worked around package naming conflict.

## [0.2.0] - 2020-07-13

Initial public release.

There is quite a bit of functionality available. Please consult the README.md at the
root of the repository for details. The major limitations at the moment are around porting
of connectors to the new architecture. At the moment a connectfor for the Vegeta load
generator and Kubernetes resource management are bundled in the distribution. The connector
catalog will expand rapidly but porting does involve some effort in order to leverage the
full value of the new system.

### Added
- Configuration management. Generate, validate, and document configuration via JSON Schema.
- Support for dispatching events for communicating between connectors.
- Initial support for check, describe, measure, and adjust operations.
- Vegeta and Kubernetes connectors for testing load generation and adjustments.
- Init command for setting up an assembly.
- Informational commands (`servo show [metrics | events | components]`).
- Foundational documentation within the code and in the README.md at the root of the repository.
- Assets for running a containerized servo under Kubernetes or Docker / Docker Compose.

[Unreleased]: https://github.com/opsani/servox/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/opsani/servox/releases/tag/v0.2.0
