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
versioned in accordance with [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Added
- Introduced Duration class for modeling time durations as Golang style duration strings.
- BaseConfiguration now provides a `yaml` method for easily to YAML.
- BaseConfiguration now provides a `json_encoders` static method for easily accessing the default encoders.
- The Vegeta connector now validates the `target` and `targets` settings.
- The check command now supports a verbose and non-verbose modes for outputting multiple checks from a connector.
- The version command can now output version data for connectors.

### Changed
- The reporting interval is now configurable `VegetaConfiguration`.
- The check event now returns a list of checks instead of one result.
- Removed the 'Connector` suffix from the default connector name attribute and introduced `full_name`.

### Removed
- The `durationpy` package has been removed in favor of a local implementation.

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
