
## TODO:

* Create connectors for vegeta, k8s, and prometheus
* Setup CLI
* Get dynamic registration going
* Setup tasks for tests, linting, console, etc
* Setup base config classes
* Build config classes for all connectors
* Support validation, generation
* Get servo going connecting to test server (local)
* Produce config fixtures from existing repos
* Includes aliasing support (replace aggregation)
* Include servo-exec in core library
* Dockerize
* Logging
* Config API
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

