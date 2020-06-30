# Opsani ServoX
![Run Tests](https://github.com/opsani/ServoX/workflows/Run%20Tests/badge.svg)
[![license](https://img.shields.io/github/license/opsani/servox.svg)](https://github.com/opsani/servox/blob/master/LICENSE)
[![release](https://img.shields.io/github/release/opsani/servox.svg)](https://github.com/opsani/servox/releases/latest)
[![GitHub release date](https://img.shields.io/github/release-date/opsani/servox.svg)](https://github.com/opsani/servox/releases)
[![Dependabot Status](https://api.dependabot.com/badges/status?host=github&repo=opsani/servox)](https://dependabot.com)

This repository contains the source code of the next generation Opsani Servo technlogy.

A servo is a deployable unit of software that connects an application or service to the
Opsani cloud optimization engine in order to identify cost savings and performance enhancements
by applying machine learning technology. Servos are lightweight Python applications and are 
typically deployed as stanadlone containers or on a Kubernetes cluster.

Servos are composed of connectors, which prvode the core functionality for integrating with metrics,
orchestration, and load generation systems/utilities. The ServoX codebase provides core functionality
shared by all servos and a rich library supporting the development of connectors.

## Quick Start

ServoX is a modern Python application distributed as an installable Python package. Development is done
in a Python install managed with [Pyenv](https://github.com/pyenv/pyenv) and a virtual environment managed by [Poetry](https://python-poetry.org/). 
This is the path of least resistance but any Python package management system should work.

* Clone the repo: `git clone git@github.com:opsani/servox`
* Install required Python: `cd servox && pyenv install`
* Install Poetry: `curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python`
* Install dependencies: `poetry install`
* Activate the venv: `poetry shell`
* Start interacting with the servo: `servo --help`
* Generate a config file: `servo generate`

## Overview

### Project Status

ServoX is currently in beta. Not all Opsani connectors are currently supported. The current focus is on Kubernetes, Prometheus, and
Vegeta. The connectors are bundled within the repository in the `connectors` directory. These packages will be migrated out to 
standalone modules as development progresses.

Putting these caveats aside, ServoX is fairly mature and provides some significant advantages and new capabilities above the 
production Servo module. If your target system is supported by the available connectors you may want to explore a ServoX 
deployment.

ServoX will be released as Servo 2.0.0 during the summer of 2020.

### Getting Started with Opsani

Access to an Opsani optimizer in required to deploy the servo and run the end to end integration tests. If you do not currently
have access to an Opsani environemnt but are otherwise interested in working with the CO engine and Servo, please reach out to us
at info@opsani.com and we will get back with you,

## Architecture

ServoX has been designed to provide a delightful experience for engineers integrating cloud optimization into their systems and 
workflow. Developer ergonomics and operator efficiency are primary concerns as integrating and orchestrating disparate components
can quickly become tiresome and complex. As a library, ServoX aspires to be as "batteries included" as possible and support developers
with well designed, implemented, and tested solutions for common concerns. As a tool, ServoX strives to support system operators and
devops engineers with strong support for common tasks and a high-velocity workflow.

There are a few key components that form the foundation of the architecture:

* **Connectors** - Connectors are pluggable components that enable the servo to interact with external systems such as metrics providers
(Prometheus, Datadog, New Relic, etc), orchestration technologies (Kubernetes, cloud provider APIs, etc), or load generators. Every major 
funtional component (including the servo itself) is a connector that inherits from the `Connector` base class. Connectors can process events
dispatched from the servo (see Events below), provide services to the user (see CLI below), and interact with other connectors.
* **Servo** - The Servo class models the active set of connectors and configuration that is executing. The servo handles connectivity with
the Opsani Optimizer API (see Optimizer below) and is responsible for the primary concerns of connectivity management and event handling.
* **Settings** - Configuration is a major shared concern in tools such as Opsani that are designed to integrate with arbitrary systems.
Ensuring that configuration is valid, complete, and functional is a non-trivial task for any component with more than a few knobs and levers.
ServoX provides a rich configuration subsystem built on Pydantic that makes modeling and processing configuration very straightforward. Out
of the box support is provided for common needs such as environment variables and dotenv files. Configuration is strongly validated using
JSON Schema and support is provided for generating config files directly from the connectors.
* **Optimizer** - The Optimizer class represents an Opsani optimization engine that the servo interacts with via an API. The optimizer can
be configured via CLI arguments, from the environment, or via assets such as Kubernetes secrets.
* **Events** - The Event subsystem provides the primary interaction point between the Servo and Connectors in a loosely coupled manner. Events
are simple string values that have connector defined semantics and can optionally return a result. The Servo base class defines the primary
events of `DESCRIBE`, `MEASURE`, `ADJUST`, and `PROMOTE` which correspond to declaring the metrics & components that the connector is interested in,
taking measurements and returning normalized scalar or time series data points, making changes to the application under optimization, or promoting
an optimized configuration to the broader system.
* **Assembly** - The Servo Assembly models the runtime environment of the servo outside of a particular configuration. The assembly is the parent
of the servo and is responsible for "assembling" it by instantiating connectors as configured by the operator. Connectors can be used multiple times
(e.g. you may want to connect to multiple discrete Prometheus services) or may not be used at all (e.g. you have a New Relic connector in the 
container image but aren't using it).
* **CLI** - The CLI provides the primary interface for interacting with the servo. The CLI is modular and contains a number of root level commands
and a registered subcommand for each active connector. Common actions include `schema`, `validate`, `describe`, `version`, etc. Each action is
implemented as an Event and are mounted contextually at the root CLI and as nested subcommands. For example, if you invoke `servo schema` you
will get an output of the JSON Schema for the entire servo with each connector registered under its config key (more on this later). If you
had the Prometheus connectors active, then `servo prometheus schema` would display the JSON Schema specific to the Prometheus connector.

### Environment Variables & Dotenv

Pay attention to the output of `servo --help` and `servo schema` to identify environment variables that can be used for configuration.
The servo handles configuration of deeply nested attibutes by building the environment variable mapping on the fly at assembly time.

For convenience, the `servo` CLI utility automatically supports `.env` files for loading configuration and is already in the `.gitignore`.
Interacting with the CLI is much cleaner if you drop in a dotenv file to avoid having to deal with the options to configure the optimizer.

### Connector Discovery

Connectors are set up to be auto-discovered using setuptools entry point available from the Python standard library. When a new connector
is installed into the assembly, it will be automatically discovered and available for interaction.

The specific of how this mechanism works is discussed in detail on the [Python Packaging Guide](https://packaging.python.org/guides/creating-and-discovering-plugins/).

The bundled connectors are registered and discovered using this mechanism via entries in the `pyproject.toml` file under the
`[tool.poetry.plugins."servo.connectors"]` stanza.

### Advanced Connector Configuration

ServoX is designed to support assemblies that contain an arbitrary number of connectors that may or may not be active and enable
the use of multiple instances of a connector with different settings. This introduces a few modes of configuration.

The servo looks to a `connectors` configuration key that explicitly declares which connectors are active within the assembly. If
a `connectors` key is not present in the config file, then all available connectors become optionally available based on the presence
or absence of their default configuration key. For example, an assembly that includes New Relic, Datadog, and SignalFX connectors
installed as Python packages with the following configuration would only activate Datadog due to the presence of its configuration
stanza:

```yaml
datadog:
  setting_1: some value
  setting_2: another value
```

This mode supports the general case of utilizing a small number of connectors in "off the shelf" configurations.

From time to time, it may become necessary to connect to multiple instances of a given service -- we have seen this a few times with
Prometheus in canary mode deployments where metrics are scattered across a few instances. In these cases, it can become necessary to
explicitly alias a connector and utilize it under two or more configurations. In such cases, the `connectors` key becomes required in
order to disambiguate aliases from configuation errors. In such cases, the `connectors` key can be configured as a dictionary where the
key identifies the alias and the value identifies the connector:

```yaml
connectors:
  prom1: prometheus
  prom2: prometheus

prom1:
  setting_1: some value
  setting_2: another value

prom2:
  setting_1: some value
  setting_2: another value
```

It is also possible to utilize the `connectors` key in order to exclude connectors from the active set. This can be done with the
dictionary syntax referenced above or using an array syntax if aliasing is not being utilized. For example, given a configuration
with New Relic and Prometheus active but some sort of issue warranting the isolation of Prometheus from the active set, the config
file might be configured like:

```yaml
connectors:
  - new_relic

prometheus:
  setting_1: some value
  setting_2: another value

new_relic:
  setting_1: some value
  setting_2: another value
```

### Requirements & Dependencies

ServoX is implemented in Python and supported by a handful of excellent libraries from the Python Open Source community. Additional
dependencies in the form of Python packages or system utilities are imposed by connectors (see below).

* [Python](https://www.python.org/) 3.6+ - ServoX makes liberal use of type hints to annotate the code and drive some functionality.
* [Pydantic](https://pydantic-docs.helpmanual.io/) - Pydantic is a fantastic parsing and validation library that underlies most classes within ServoX. It enables the
strong modeling and validation that forms the core of the configuration module.
* [Typer](https://typer.tiangolo.com/) - Typer provides a nice, lightweight enhancement on top of [Click](https://click.palletsprojects.com/en/7.x/) for building CLIs in Python. The CLI is built out on top of Typer.
* [httpx](https://www.python-httpx.org/) - httpx is a (mostly) requests compatible HTTP library that provides support for HTTP/2, is type annotated, has extensive test coverage, and supports async interactions on top of asyncio.
* [loguru](https://loguru.readthedocs.io/en/stable/index.html) - A rich Python logging library that builds on the foundation of the standard library
logging module and provides a number of enhancements.

## Development

### Contributing to ServoX

Open Source contributions are always appreciated. Should you wish to get involved, drop us a line via GitHub issues or
email to coordinate efforts.

It is expected that most Open Source contributions will come in the form of new connectors. Should you wish to develop
a connector, reach out to us at Opsani as we have connector developer guides that are in pre-release while ServoX matures.

### Docker & Compose

`Dockerfile` and `docker-compose.yaml` configurations are available in the repository and have been designed to support both
development and deployment workflows. Configuration file mounts and environment variables can be used to influence the behavior
of the servo within the container. The `SERVO_ENV` environment variable controls whether development packages are installed into
the image (hint: set it to `SERVO_ENV=production` to slim down your images and speed up builds).

### Work in Progress

The following is a non-exhaustive list of work in progress on the road to shipping v2.0.0

* [ ] Create connectors for vegeta, k8s, and prometheus
* [ ] Build config classes for all connectors
* [ ] Package builds
* [ ] Implement testing API for exercising API dependent components (e.g. `ServoRunner`)
* [ ] Produce config fixtures from existing projects
* [ ] Include servo-exec in core library
* [ ] Encoders

## Testing

Tests are implemented using [Pytest](https://docs.pytest.org/en/stable/) and live in the `tests` subdirectory. Tests can be executed directly via the `pyenv` CLI
interface (e.g. `pytest tests`) or via the developer module of the CLI via `servo dev test`.

## License

ServoX is distributed under the terms of the Apache 2.0 Open Source license.

A copy of the license is provided in the [LICENSE](LICENSE) file at the root of the repository.
