# Opsani ServoX

This repository contains the source code of the next generation Opsani Servo architecture.

Details about the design parameters of the implementation are detailed on the ServoX document available within the
Opsani wiki resoures.

This readme kinda sucks at the moment but the code is solid. As the last details of the design solidify this doc will evolve. But in the meantime...

Quick install via `poetry install [GITHUB URL]` and then work with the CLI `servo --help`. 

## Rundown of cool things...

The way that ServoX works is pretty hot. There are two fundamental models: Settings and Connectors, which are designed and implemented in matching pairs.
Settings classes are [Pydantic models](https://pydantic-docs.helpmanual.io/usage/models/) that describe the configuration of the connector class that they are paired with. 

The settings classes strongly specify the configuration surface area of a connector in a declarative, data driven fashion. Connectors are initialized with their
declared settings class as a required input. This has the effect of making it annoying and difficult to shoot yourself in the foot unless you really, really want to.

Connectors are auto-discovered via Python setuptools entry-points and magically become available once you install with Poetry (or Pip if you are a hoodlum). Each Connector (including the Servo) has a namesake settings class that models its configuration within the YAML file (or any other source in principle).

Connectors are pulled together into a routing table where in each root level key is either "owned" by a particular connector and subject to validation against its schema or are treated as foreign data to either be ignored (in the case of third party development) or rejected (if ypu are not lookibg for model extensibility and prioritize stability
and control).

You can either have implicit or explicit activation of connectrs. In the simple case where you don't have a `connectors` key in your YAML, all discovered connectors get optionally mounted at their default routes (that is to say, it will leave you alone until you begin coniguring a tkey that it lives at).

So if in your container/virtual env you have VegetaConnector, KubernetesConnector, PrometheusConnector, and DatadogConnector you can now activate them by adding the `vegeta`, `kubernetes`, `prometheus`, and `datadog` keys respectively at the root level of the config YAML..

But you can also manage them explicity via the `connectors` key like so:

```
connectors:
 - vegeta
 - datadog
```

if you only want to enable a subset of connectors or are debugging something and want to tag some in and out while working on something. In the explicit configuration form, you must have a properly configured stanza for the connector or it won't boot up.

But then it gets really fun because you can mount the same connector multiple times by using a key/value synatax:

```
connectors:
 vegeta: VegetaConnector
 staging_datadog: DatadogConnector
 production_datadog: DatadogConnector
 prom1: prometheus
 prom2: prometheus
```

In this form,when the Servo is assembled, it resolves the `connectors` key into a canonical map of paths to connector classes and then dynamically builds a Pydantic model by  creating a copy of the base settings model for each connector and attaching it to the dynamic model.

Connectors are initialized with a Settings instance of the type they declare and literally cannot tell the difference between running standalone or in a herd. They don't even know where the settings came from -- they just know that the model is valid and they were initialized with it.

When I assemble the dynamic servo settings model, I use the config paths to build environment variable names of the form `SERVO_[CONNECTOR]_[ATTRIBUTE]` so you get environment variables controls for every modeled setting, for free, with type safety and validation (you can trivially implement handlers for arbitrary types). You can also override multiple properties at once by doing `SERVO_VEGETA='{ "rate": "50/1s", "duration": "20m" }' and as long as the env var parses as clean JSON into a dictionary it will be merged onto the model before validation

## TODO:

* Create connectors for vegeta, k8s, and prometheus
* Build config classes for all connectors
* Support validation, generation
* Get servo going connecting to test server (local)
* Produce config fixtures from existing repos
* Include servo-exec in core library
* Dockerize
* Logging
* Event bus (broadcast measure/adjust)
* Metrics
* Encoders

## Overview

### ServoX Layout

* README.md
* pyproject.toml
* connector.py
* connector_test.py
* docs
* LICENSE

dotenv support

### Configuration

immutable


### Architecture

1. Components: settings, connector, cli
2. Connector routes (id => class)
3. Assembly
4. ENV vars overrides and aliased configs

config path, command name, settings, env vars

### Connector Discovery



import pkg_resources

discovered_plugins = {
    entry_point.name: entry_point.load()
    for entry_point
    in pkg_resources.iter_entry_points('myapp.plugins')
}

https://packaging.python.org/guides/creating-and-discovering-plugins/

In pyproject.toml:
```
[tool.poetry.plugins."servo.connectors"]
".rst" = "some_module:SomeClass"
```

### Modules

* config
* logging
* http
* testing

## Installation

## CLI

## Testing

## Packaging

## Contributing

## License

