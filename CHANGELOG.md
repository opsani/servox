# CHANGELOG

Servo is an Open Source framework supporting Continuous Optimization of
infrastructure and applications via the Opsani optimization engine. Opsani
provides a software as a service platform that optimizes the resourcing and
configuration of cloud native applications to reduce operational costs and
increase performance. Servo instances are responsible for connecting the
optimizer service with the application under optimization by linking with the
metrics system (e.g. Prometheus, Thanos, DataDog, etc) and the orchestration
system (e.g. Kubernetes, CloudFormation, etc) in order to apply changes and
evaluate their impact on cost and performance.

Servo is distributed under the terms of the Apache 2.0 license.

This changelog catalogs all notable changes made to the project. The format is
based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/). Releases are
versioned in accordance with [Semantic
Versioning](https://semver.org/spec/v2.0.0.html).

## [2.2.7] "genesis" - 2023-07-27

### Changed

- Progress reporting queue is cleared on command completion to prevent stale progress from being sent when pileup occurs [544](https://github.com/opsani/servox/pull/544)
- Memory parsing logic treats unitless decimal strings as GiB (aligns with existing float parsing) per the servo protocol [546](https://github.com/opsani/servox/pull/546)

## [2.2.6] "genesis" - 2023-07-10

### Changed

- A unique identifier for each servo deployment is now supported in configuration and sent with every request to the backend
- A unique identifier for each command received from the backend is now retrieved from the response to the WHATS_NEXT request and included on all servo requests for the given command (progress and completion)

## [2.2.5] "genesis" - 2023-06-15

### Changed

- 404 no longer considered fatal status code
- Checks now support exponential backoff
- Checks delay configuration now defaults to exponential backoff
- Diagnostics reporting disabled by default
- Connector discovery now done with importlib.metadata.entry_points instead of deprecated pkg_resources.iter_entry_points

### Fixed

- Measurement delay (wait driver only)

## [2.2.4] "genesis" - 2023-04-14

### Changed

- Default interval of progress updates changed from 5 seconds to 60
- Updated API path used for servo endpoint of optimize solution

## [2.2.3] "genesis" - 2023-03-31

### Changed

- Update ID of appD config to use optimizerID instead of workloadID

## [2.2.2] "genesis" - 2023-03-20

### Changed

- Updated third party libraries


## [2.2.1] "genesis" - 2023-02-17

### Added
- Modify Opsani dev connector to work with Appd configuration
- Support for agent management sidecar

### Fixed
- Fix servo crash on connection error


## [2.2.0] "subtle rhapsody" - 2023-01-24

### Added

- Appdynamics authentication
- Add helm chart


## [2.1.1] "subtle rhapsody" - 2022-08-23

### Added

- Servox preliminary support for optimization of StatefulSets [439](https://github.com/opsani/servox/pull/439)

### Changed

- README.md and checks documentation updated to reflect changes to CLI executions of checks [445](https://github.com/opsani/servox/pull/445)

### Fixed

- Kubernetes metrics server connector returning tuning memory request/limit from main workload [449](https://github.com/opsani/servox/pull/449)
- Invalid operation error when comparing NaN values against configured SLO mimimums [450](https://github.com/opsani/servox/pull/450)

### Security

- httpx updated to version 0.23.0 [446](https://github.com/opsani/servox/pull/446)

## [2.1.0] "subtle rhapsody" - 2022-07-25

### Added

- Native MacOS support [422](https://github.com/opsani/servox/pull/422)
- Container logs are now included in rejections caused by restarting pods [383](https://github.com/opsani/servox/pull/383)
  - Test coverage for container logs [394](https://github.com/opsani/servox/pull/394/files)
- Added [Black](https://github.com/psf/black) code formatter [398](https://github.com/opsani/servox/pull/398)
- Added [MyPy](https://github.com/python/mypy) code scans and inline linting [402](https://github.com/opsani/servox/pull/402)
- `no_tuning` configuratibility to allow external drivers to leverage Opsani Dev orchestration without being required to launch a Saturation or Tuning optimization [404](https://github.com/opsani/servox/pull/404)
- Combined CI coverage reports from Unit tests and Integration tests [425](https://github.com/opsani/servox/pull/425)

### Changed

- Pinned certain dependencies for improved stability:
  - Kubectl [425](https://github.com/opsani/servox/pull/425)
  - aws-iam-authenticator [425](https://github.com/opsani/servox/pull/425)
- Response bodies are now included in the logging of unretryable HTTP responses [400](https://github.com/opsani/servox/pull/400)
- Updated dependencies:
  - pyenv-action [403](https://github.com/opsani/servox/pull/403)
  - black [415](https://github.com/opsani/servox/pull/415), [422](https://github.com/opsani/servox/pull/422)
  - typer [422](https://github.com/opsani/servox/pull/422)
  - pydantic [422](https://github.com/opsani/servox/pull/422)
- Remedies from multiple failing checks are now applied in the same check iteration. Previous behavior was to apply a single remedy from the first failing check then rerun all checks, rinse and repeat until passing [411](https://github.com/opsani/servox/pull/411)
- Fast Fail metric thresholds that are close to 0 will be skipped instead of counting as failures [412](https://github.com/opsani/servox/pull/412)
- Updated kubernetes connector container `resources` logic to return `limits` in place of `requests` in cases where `requests` are not set [426](https://github.com/opsani/servox/pull/426)

### Fixed

- Improved CI test stability [388](https://github.com/opsani/servox/pull/388)
- Resolved edge case where timeouts would run indefinitely if set to a duration of 0 [392](https://github.com/opsani/servox/pull/392)
- Resolved error in connector logic that could have caused the wrong target resource to be referenced as the target or failed to locate the target resource altogether [422](https://github.com/opsani/servox/pull/422)


### Removed

- Private repository mirroring [389](https://github.com/opsani/servox/pull/389/files)
- Eager Metrics Observer [397](https://github.com/opsani/servox/pull/397)

## [2.0.0] "electric orchestra" - 2021-12-18

### Added

- Kubernetes metrics connector for reporting Pod metrics from kubernetes metrics server [#369](https://github.com/opsani/servox/pull/369)
- Ability to upload logs and configuration for tech support [#373](https://github.com/opsani/servox/pull/373)
- Support for describing and adjusting kubernetes container environment variables [#375](https://github.com/opsani/servox/pull/375)

### Changed

- Updated project to python 3.9.7 [#367](https://github.com/opsani/servox/pull/367)
- No longer using custom kubernetes_asyncio package [#368](https://github.com/opsani/servox/pull/368)
- The Opsani Dev connector attaches the Kubernetes Metrics connector if we're able to list pod metrics [#379](https://github.com/opsani/servox/pull/379)

### Fixed

- Relaxed python version dependency requirements [#372](https://github.com/opsani/servox/pull/372)

## [0.11.1] "preposterous ports" - 2021-11-19

### Fixed

- Kubernetes connector logical error in check for argo rollout permissions [#358](https://github.com/opsani/servox/pull/358)
- Opsani Dev connector configuration mathematical error in prometheus queries for success_rate and error_rate [#357](https://github.com/opsani/servox/pull/357)

## [0.11.0] "preposterous ports" - 2021-11-01

### Added

- Prometheus connector fast-fail support; short-circuit long running measurements when SLO violations are detected [#330](https://github.com/opsani/servox/pull/330)
- Kubernetes connector Argo Rollouts WorkloadRef support [#337](https://github.com/opsani/servox/pull/337)
- Opsani Dev connector support for configuration of image and tag used for envoy sidecar injection [#341](https://github.com/opsani/servox/pull/341)
- Kubernetes connector support for injecting static environment variables into the tuning Pod [#343](https://github.com/opsani/servox/pull/343)

### Changed

- Kubernetes Deployment optimizations (Saturation Mode) `destroy` error behavior changed to `shutdown`; the Deployment
resource is no longer destroyed and is scaled to zero replicas instead [#317](https://github.com/opsani/servox/pull/317)
it to zero replicas instead
- Kubernetes adjustment values no longer raise validation errors when outside the configured range (eg. falling back to
initially observed/baseline values) [#279](https://github.com/opsani/servox/pull/279)

### Fixed

- Checks running on multiple connector assemblies were only processing checks for the first connector to respond [#351](https://github.com/opsani/servox/pull/351)
- Prometheus connector reliance on implicit importing of servo.cli [#355](https://github.com/opsani/servox/pull/355)

## [0.10.7] "baseless allegation" - 2021-09-02

### Changed

- Updated opsani Dev connector metrics to remove confusing _avg metric name [#331](https://github.com/opsani/servox/pull/331)

## [0.10.6] "baseless allegation" - 2021-08-20

### Changed
* Update forked kubernetes_asyncio dependency to use PyPI package source instead of github [#315](https://github.com/opsani/servox/pull/315)

## [0.10.5] "baseless allegation" - 2021-08-16

### Changed

- Improved messaging of unschedulable adjustment rejections [#285](https://github.com/opsani/servox/pull/285)
- Test suites now treat warnings as errors [#288](https://github.com/opsani/servox/pull/288)
- Relaxed validation of account names to support accounts that don't adhere to DNS naming convention [#302](https://github.com/opsani/servox/pull/302)
- Updated calculation and naming of Opsani Dev prometheus metrics to use averages [#305](https://github.com/opsani/servox/pull/305)

### Added

- Beta support for optimizing [Argo Rollouts](https://argoproj.github.io/argo-rollouts/) [#303](https://github.com/opsani/servox/pull/303)
- Simple telemetry to report run environment details to the backend [#261](https://github.com/opsani/servox/pull/261)
- Existing kubernetes resource requests/limits are now validated [#282](https://github.com/opsani/servox/pull/282)

### Removed

- Opsani Dev p99 latency metrics [#306](https://github.com/opsani/servox/pull/306)
- Opsani Dev main_pod_avg_request_rate redundant metric [#305](https://github.com/opsani/servox/pull/305)

## [0.10.4] "baseless allegation" - 2021-07-09

### Fixed

- The Optimizer `base_url` is now normalized to strip a trailing slash to ensure
  that computed paths are deterministically correct.
- Cancellation requests from the optimizer were not being properly respected,
  resulting in unnecessary delays to cancel operations from the console.
- Enforce settlement time user-config parameter.

## [0.10.3] "baseless allegation" - 2021-06-06

### Enhanced

- The `port` option is now respected by the sidecar injection remedy.
- Introduced the `scripts` connector that supports attaching arbitrary shell
  commands to be run before, on, or after an event is dispatched by the servo. [#245](https://github.com/opsani/servox/pull/225)
- Add ImagePullBackOff error for target service (tuning or mainline) [#248](https://github.com/opsani/servox/pull/248)
- Improve K8s Deployment updates and error handling. [#253](https://github.com/opsani/servox/pull/253)

### Fixed

- Dynamically named ports are now supported by resolving the port name against
  the Pod spec template [#246](https://gihtub.com/opsani/servox/pull/246)
- Use correct process for determining current active pods [#247](https://gihtub.com/opsani/servox/pull/247)
- Support settlement command correctly in ServoX k8s connector [#240](https://gihtub.com/opsani/servox/pull/240)
- Remove token exposure in TRACE log [#239](https://gihtub.com/opsani/servox/pull/239)
- When checks fail and are rerun, the tuning pod is no longer rebuilt
  unnecessarily.
- Eliminated cases where the `KubernetesConnector` could fail to report progress
  due to Kubernetes API availability, timeouts, etc. resulting in errant Servo
  disconnected events being emitted by the optimizer.
- Improved test resiliancy and removed name collisions in automated test namespaces

## [0.10.2] "baseless allegation" - 2021-05-20

### Fixed

- Port was not being accepted as input on inject_sidecar cli command [#241](https://gihtub.com/opsani/servox/pull/241)


## [0.10.1] "baseless allegation" - 2021-05-11

### Enhanced

- Flexible support for pod template resource specifications [#182](https://github.com/opsani/servox/pull/182)
- Aligned memory unit display with the expected Kubernetes output

### Fixed

- Allowed min=max in range settings
- Correctly use selector labels to locate deployment [#202](https://github.com/opsani/servox/pull/202)

## [0.10.0] "baseless allegation" - Unreleased

### Enhanced

- Opsani Dev checks for traffic flows are faster.
- Service check errors now include the missing labels.
- Container resources are now checked against the optimizable range before
  optimization begins.
- Introduced pub/sub transformers for filtering, splitting, and aggregating
  messages across channels. [#191](https://github.com/opsani/servox/pull/191)
- Added `--no-poll` and `--interactive` options to the `servo run` command.
  [#192](https://github.com/opsani/servox/pull/192)
- Enjoy a random start-up banner in a random color palette.
  [#193](https://github.com/opsani/servox/pull/193)
- `TRACE` logging from the `servo.api` module now includes cURL commands.
  [#194](https://github.com/opsani/servox/pull/194)

### Fixed

- Traffic checks no longer require a 2xx status code to pass.
- Load testing hints using Vegeta now include `kubectl exec` stanza to run
  remotely rather than on the local workstation.
- Resource requirements now output a sensible error message rather than raising
  a `KeyError` when `cpu` or `memory` are not defined.

### Changed

- Updated to httpx v0.17.0
- Updated uvloop to v0.15.2
- Optimizer is now a member of the Configuration object.

### Fixed

- HTTP connection errors could result in unbound references to `response` in the
  `servo.api` module. (SOL-292)

## [0.9.5] "serenity now" - 2021-02-24

### Enhanced

- Added support for deploying Opsani Dev on Kubernetes `NodePort` Services.
- Range setting that are out of step alignment now suggest alternative values
  to consider.
- Normal operational logging is less verbose.

### Fixed

- Container restarts due to `CancellationError` in response to Kubernetes
  adjustment failures are now avoided.
- Kubernetes `ContainersNotReady` status upon timeout are now handled as
  adjustment failures.
- HTTP status code 4xx responses are no longer retried.

## [0.9.4] "serenity now" - 2021-02-17

### Fixed

- Use the bound logger for reporting Prometheus query errors in publisher. refs
  SOL-238

## [0.9.3] "serenity now" - 2021-02-16

### Fixed

- Fixed an asyncio crash in the `ServoRunner`.
- Gracefully handle query errors from Prometheus.
- Support asyncio cancellation within pub/sub publisher decorator.

## [0.9.2] "serenity now" - 2021-02-16

### Fixed

- Include colorama package in release builds.

## [0.9.1] "serenity now" - 2021-02-16

### Fixed

- Include toml package in release builds.

## [0.9.0] "serenity now" - 2021-02-16

### Added

- Opsani Dev v2.0 integrated for rapid service optimization.
- Incorporated [uvloop](https://github.com/MagicStack/uvloop) for faster async
  event loops.
- Initial release of Wavefront Connector.
- Support for marking adjustments as failed or rejected via exceptions.
- Multiple servos can now be run within a single assembly. If the config file is
  a compound YAML document, multiple servos will be instantiated allowing the
  concurrent optimization of multiple applications.
- Introduced emulator connector, which pretends to take measurements and make
  adjustments with randomly sampled data but does not do any real work.
- New servo configurations can be generated and added to the assembly via
  `servo generate --append`.
- New command `servo list` for viewing the active servos in the assembly.
- Introduced new top-level option `--name`/`-n` for targeting a specific servo
  in the assembly when running in multi-servo mode.
- In multi-servo configurations, concurrency can be constrained via the new
  top-level `--limit` option.
- Connector details for a particular servo instance can now be displayed via the
   `servo show connectors` CLI command.
- Extensive new development and testing tooling.
- Introduced `servo.pubsub` module providing publisher/subscriber capabilities.
- The test suite now runs extensive integration and system tests under CI.
- All aspects of the Opsani Dev installation experience are now covered by
  checks.
- Introduced the `attach` and `detach` lifeycle events for handling setup and
  teardown concerns that need to execute when a Servo or Connector is added or
  removed from am Assembly/Servo.
- Added automated checks.
- Introduced pub/sub module.

### Enhanced

- The Kubernetes connector now recovers from and reports on numerous failure
  modes such as unschedulable configurations.
- The Prometheus connector now exposes a Prometheus HTTP API client library.
- The Envoy sidecar can now be automatically injected via the CLI.
- The Opsani Dev connector now exposes a very simple configuration surface.

### Changed

- The `__codename__` support has been generalized as `cryptonym` attribute for
  all connectors.
- Version output now includes the cryptonym.
- Updated Pydantic to v1.7.3
- Updated httpx to v0.16.1
- Updated orjson to v3.4.6
- Updated the `servo.errors` module with numerous new error types.
- The `servo.events.run_event_handlers` method now always returns a list of
  `EventResult` objects when `return_exceptions` is True. Exceptions are caught
  and embedded in the `value` attribute.
- Exceptions raised by an event handler are decorated with a
  `servo.events.EventResult` object on the `__event_result__` attribute.
- When an event is cancelled by a before event handler by raising a
  `servo.errors.EventCancelledError`, an empty result list is now returned.
- The `servo.api.Mixin` class is now an abstract base class and requires the
  implementation of the `api_client_options` method.
- Configuration of backoff/retry behaviors has been reimplemented for clarity
  and simplicity.
- The `servo.assembly.Assembly` class now maintains a collection of servos
  rather than a singleton.
- The optimizer settings are now part of the configuration.
- All CLI commands are now multi-servo aware and enabled.
- The `servo.Runner` class has been split into `servo.AssemblyRunner` and
  `servo.ServoRunner` to support multi-servo configurations.
- The Docker image entry point is now multi-servo aware.
- Servos are now named. The default name is adopted from the Optimizer ID if
  one is not directly configured.
- The top level `servo connectors` command now displays info about all available
  connectors rather than those that are currently active. `servo show
  connectors` now reports instance specific connector info.
- Migrated the `current*` family of methods off of model classes and into module
  level functions be more Pythonic.

### Removed

- The `duration` attribute of the Vegeta Connector configuration is now private
  as the optimizer or operator always provide the value.
- The `servo.events.broadcast_event` method was removed as it was seldom used
  and the functionality is easily replicated in downstream code.
- API client options including base URL, proxies, and timeouts are no longer
  duplicated across connectors as an extra attribute.

### Fixed

- Local versions are now resolved via the `pyproject.toml` file to accurately
  reflect pre-release versioning and local work in progress.
- Exceptions are now chained within the Kubernetes Connector, ensuring that
  traceback context is not lost.
- An invalid key was referenced during adjustment of Kubernetes container memory
  request/limits.
- Scalar `servo.types.DataPoint` objects are now serialized for processing by
  the optimizer service rather than raising an exception.
- The `ConnectorCLI` class now supports aliased connector instances.
- Test coverage gaps have been plugged throughout the CLI module.
- Scalar data points can now be handled by the CLI.
- Invalid keys in the `connectors` field of a config file will no longer
  trigger an unhandled exception.
- Step values of range settings are now validated to ensure that a step of zero
  is not configured.
- Setting values are now validated appropriately upon being changed. This
  prevents invalid values from being externally applied to a running
  optimization (e.g., an external deployment or manual change is made).
- Fixed exception in Prometheus mutltichecks due to unescpaed format
  characters in interpolated Prometheus queries.
- Fixed connector lifecycle issue with Opsani Dev connector preventing use in
  several of the CLI commands.
- Corrected an entry points discovery issue affecting the latest versions of
  Python.

## [0.8.4] "pass the calamari" - 2021-02-05

### Fixed

- Containers are no longer accessed positionally by index instead of by name at
  any time. This was resulting in broken adjustments when the Deployment
  contained an init container, the main container was not the first container,
  or during installation the Envoy proxy was injected at the beginning of the
  container list instead of the end.

## [0.8.3] "pass the calamari" - 2020-10-21

### Fixed

- Initialize Kubernetes optimizations before taking a measurement. This ensures
  that the Kubernetes Connector has an opportunity to set up the environment
  before a measurement is taken that is reliant on the setup (e.g. all canaries
  have been created).

## [0.8.2] "pass the calamari" - 2020-10-16

### Added

- Connectors can now be dynamically added and removed from the servo via the
  `servo.servo.Servo` methods `add_connector` and `remove_connector`.
- Individual checks can now be run by ID or name via the
  `servo.checks.BaseChecks.run_one` method.
- Developer Makefile tasks `format`, `lint`, and `test`.
- Introduced pre-commit hooks for enforcing style guides and engineering
  standards.

### Changed

- Code across the library is now referenced by package and module rather than
  importing individual classes and functions. This reduces the amount of
  boilerplate code in any given module and makes the code more accessible by
  making implicit references explicit and unambiguous.
- Simplified `servo.logging.ProgressHandler` implementation with an asyncio
  queue and producer/consumer pattern.
- Renamed `servo.checks.BaseChecks.run_` method to
  `servo.checks.BaseChecks.run_all`.
- Renamed the `filter_` argument of check runners to `matching`.
- Millicore values are now serialized to strings as simple integers when evenly
  divisible by 1000 (e.g., `str(Millicore(3000)) == "3"`).
- The canonical coding style for the project has been evolved to directly import
  packages and modules instead of class and functions.

### Fixed

- `servo show components` now includes the setting names instead of naked
  values.
- Type resolution (used in eventing and checks) is now able to flexibly handle
  arbitrary imports path and type aliases.
- Settlement time can now be supplied to adjust operations by the optimizer via
  the `servo.types.Control` type.
- Before and after event handlers are no longer invoked with extraneous
  arguments from the on event definition.

## [0.8.1] "pass the calamari" - 2020-10-02

Maintenance release to remove development packages from Docker images.

### Changed

- Reworked Docker build automation to decouple from release process.
- Fixed breakage in production builds.
- Updated Poetry dependency to v1.1.0 in Dockerfile.

## [0.8.0] "pass the calamari" - 2020-10-02

### Added

- `servo.__codename__` constant now contains the release codename.
- Extensive docstring comments for all members of the `servo.types` module.
- Kubernetes Containers can be aliased to set explicit Compomnent names rather
  rather than automatically deriving names from Deployment/Container.
- Kubernetes Optimization Strategy classes can now accept options from the
  config file (currently supports `alias` for canaries).
- Integrated orjson to gain control over JSON/YAML serialization for classes
  that inherit from built-in types (e.g., str, int, float).
- The `ProgressHandler` now handles exceptions and optionally notifies an
  external exception handler.
- Servo will now interrupt operations when it detects losing sync with the
  backend by encountering unexpected operation errors.
- Critical checks can be declared via the `require` decorator.
- Added the `warn` decorator for creating checks that emit warnings rather than
  failing.

### Removed

- Subprocess methods have been removed from `servo.connector.Connector` in favor
  of directly importing the subprocess module from the utilities module.
- The `required` attribute from the `servo.checks` module in favor of
  `severity`.

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
- JSON and YAML serializations now favor human readable representations by
  default whenever possible.
- Multicheck methods now yield more readable IDs based off the parent multicheck
  method name (e.g., `check_resource_requirements_item_0`).
- Checks now have a severity described by the `servo.checks.Severity`
  enumeration, replacing required.
- Required check nomenclature has been replaced with the `critical` severity
  level to clarify expectations and eliminate ambiguity in check behavior.

### Fixed

- Progress tracking now handles zero length durations appropriately (e.g., in
  warmup, settlement, etc).
- Model objects that inherit from builtin classes can now be serialized to
  custom representations.
- Kubernetes configuration values now serialize to human readable values instead
  of numerics.
- Multicheck expanded methods are now filterable and taggable.
- Progress logging and reporting will no longer trigger unhandled exceptions.
- Adjust operations now return a state descriptor rather than parroting back the
  requested state.
- Kubernetes Connector is now aware of out of band changes such as those made by
  Horizontal Pod Autoscalers.

## [0.7.0] "nice for what?" - 2020-09-09

### Added

- `servo run --check` can now be controlled via the `SERVO_RUN_CHECK`
  environment variable.
- The `servo.logging` module now provides the `set_colors` function for
  programmatically enabling or disabling coloring.
- The CLI accepts a `—no-colors` argument that will explicitly disable coloring.
- The `SERVO_NO_COLOR` and `NO_COLOR` environment variables are respected to
  disable coloring.
- The API URL can be now be overridden via the hidden `--url` CLI option or the
  `OPSANI_URL` environment variable.
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

- Handled null annotations and labels when cloning a Deployment in order to
  create a Pod.
- Servo runner now honors proxy settings (previously only honored within
  connectors).
- `servo check servo` now works as expected (previously not handled as a
  connector name).

### Changed

- Updated to httpx v0.14.3.

## [0.6.0] "woke up like this" - 2020-08-30

### Enhanced

- The checks subsystem has been rearchitected to streamline the development of
  checks and provide a better operational experience for running and debugging
  checks.

### Added

- `servo run --check` will run all configured checks before starting the servo
  runloop.
- `servo check` now supports filtering by name, id, and tags. Failure mode
  handling is configurable via `--halt-on-failed=[requirement,check,never]`
- `servo run` now applies exponential backoff and retry to recover from
  transient HTTP errors.
- Introduce associations mixin for managing off-model support objects that don't
  make sense to model as attributes.
- `ServoConfiguration` class for applying settings to the servo itself.
- HTTP proxy support (configured on `ServoConfiguration`).
- Timeout configuration (configured on `ServoConfiguration`).
- Support for configuring backoff and retry behaviors (configured on
  `ServoConfiguration`).
- Baseline set of checks on the Kubernetes Connector.

### Fixed

- Attempting to connect to an invalid or unavailable optimizer backend now
  triggers exponential backoff and retry rather than crashing.
- Encountering an unexpected event error from the optimizer now aborts the
  operation in progress and resets rather than waiting for the operation to
  complete.

## [0.5.1] - 2020-08-23

### Removed

- Migrated servo-webhooks into a standalone project (see
  [opsani/servo-webhooks](https://github.com/opsani/servo-webhooks))

## [0.5.0] - 2020-08-22

### Added

- The capabilities of the logging module have been significantly enhanced with
  supporting for changing the logging levels, performing timed logging around
  methods, and attaching backtraces.
- Log messages are now annotated with the event context when logging occurs
  during an event handler.
- The metadata decorator now accepts a tuple for the `name` parameter for
  providing an explicit default name for connectors of the decorated type,
  overriding the name inference.
- Progress can be automatically reported to the Opsani API by annotating log
  messages with a "progress" key.
- Introduced general utilities for executing subprocesses including streaming
  output, timeouts, etc.
- Added integration testing infrastructure for testing the servo in Docker,
  Minikube, and EKS.
- Integrated complete implementation of Kubernetes connector on top of
  kubernetes_asyncio.

### Changed

- Enable parsing of extended Golang duration strings in `Duration` (days, weeks,
  months, years).
- The base connector class has been renamed from `Connector` to `BaseConnector`
  to align with other inheritance patterns in the library.
- Redesigned the Kuberenetes configuration to adopt familiar naming conventions
  and structure from Kubernetes.

## [0.4.2] - 2020-07-24

### Fixed

- The init command output formatting was broken when accepting defaults.

## [0.4.1] - 2020-07-24

### Fixed

- The `respx` mocking library was in the main project dependencies instead of
  dev dependencies.

## [0.4.0] - 2020-07-24

The headline feature of this release is the adoption of asyncio as a core part
of the architecture.

### Fixed

- Docker image builds now correctly support excluding development dependencies
  via the `SERVO_ENV` build argument.

### Added

- Docker images are now published to Docker Hub with README content from
  `docs/README-DOCKERHUB.md`.
- Introduced Duration class for modeling time durations as Golang style duration
  strings.
- BaseConfiguration now provides a `yaml` method for easily to YAML.
- BaseConfiguration now provides a `json_encoders` static method for easily
  accessing the default encoders.
- The Vegeta connector now validates the `target` and `targets` settings.
- The check command now supports a verbose and non-verbose modes for outputting
  multiple checks from a connector.
- The version command can now output version data for connectors.
- The connectors command now outputs names for displaying aliased
  configurations.
- The servo now runs asynchronously on top of asyncio.
- Added start-up banner to run command.

### Removed

- The `durationpy` package has been removed in favor of a local implementation.

### Changed

- Updated Docker and Docker Compose configurations to use `/servo/opsani.token`
  as the default path for API tokens mounted as a file.
- Docker images pushed to Docker Hub are now built with `SERVO_ENV=production`
  to exclude development packages.
- The reporting interval is now configurable `VegetaConfiguration`.
- The check event now returns a list of checks instead of one result.
- Removed the 'Connector' suffix from the default connector name attribute and
  introduced `full_name`.
- The `__config_key__` attribute has been renamed to `__default_name__`.
- The identifier for connectors has been renamed to `name` for simplicitly and
  clarity.
- The `ServoAssembly` class has been renamed to `Assembly`.

## [0.3.1] - 2020-07-16

### Fixed

- Eliminated secondary connectors package to eliminate Docker & PyPI
  distribution issues.

## [0.3.0] - 2020-07-15

### Added

- Config files can be outputted in Kubernetes ConfigMap format (`servo config -f
  configmap`).
- All Connectors can now dispatch events (previously only available to the
  `Servo` class).
- `Optimizer` now includes an `api_url` property.
- Event accessor `Connector.events` and `Connector.get_event`.
- Project automation via GitHub Actions (Dependabot, release-drafter, PyPI
  release, Docker builds).

### Removed

- Removed `ServoAssembly.default_routes` from the API (usage eliminated).

### Changed

- Normalized naming of miscellaneous config methods.
- Renamed `ServoAssembly.all_connectors` to `ServoAssembly.all_connector_types`
  for clarity.

### Fixed

- Pydantic will no longer see self-references between connectors (avoids
  recursion sharp edge).

## [0.2.5] - 2020-07-13

### Fixed

- Dotenv was non-functional in PyPI release package.

## [0.2.1] - 2020-07-13

### Fixed

- Worked around package naming conflict.

## [0.2.0] - 2020-07-13

Initial public release.

There is quite a bit of functionality available. Please consult the README.md at
the root of the repository for details. The major limitations at the moment are
around porting of connectors to the new architecture. At the moment a connectfor
for the Vegeta load generator and Kubernetes resource management are bundled in
the distribution. The connector catalog will expand rapidly but porting does
involve some effort in order to leverage the full value of the new system.

### Added

- Configuration management. Generate, validate, and document configuration via
  JSON Schema.
- Support for dispatching events for communicating between connectors.
- Initial support for check, describe, measure, and adjust operations.
- Vegeta and Kubernetes connectors for testing load generation and adjustments.
- Init command for setting up an assembly.
- Informational commands (`servo show [metrics | events | components]`).
- Foundational documentation within the code and in the README.md at the root of
  the repository.
- Assets for running a containerized servo under Kubernetes or Docker / Docker
  Compose.

[Unreleased]: [View Compare](https://github.com/opsani/servox/compare/v0.2.0...HEAD) [0.2.0]:
[tag v0.2.0](https://github.com/opsani/servox/releases/tag/v0.2.0)
