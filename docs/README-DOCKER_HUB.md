# Opsani ServoX

[![Images
Badge](https://images.microbadger.com/badges/image/opsani/servox.svg)](https://microbadger.com/images/opsani/servox)

Docker image for Opsani [ServoX](https://github.com/opsani/servox).

The image includes connectors for Kubernetes, Prometheus, and Vegeta. Additional
connectors can be added by inheriting from the image and using
[Poetry](https://python-poetry.org/) to add connector packages to the assembly
(see below).

## Supported tags and respective `Dockerfile` links

- [`latest`](https://github.com/opsani/servox/blob/main/Dockerfile)
- [`edge`](https://github.com/opsani/servox/blob/main/Dockerfile)

For other versions see [releases](https://github.com/opsani/servox/releases) on
GitHub and the available [tags on Docker
Hub](https://hub.docker.com/r/opsani/servox/tags/).

## Configuration

The servo requires an optimizer, an API token, and a config file to run. The
image is configured to accept these parameters through a combination of
environment variables and mounts.

The API token can be configured through an environment variable *or* a mount.
When the `OPSANI_TOKEN` environment variable is not set, the servo will search
for a token file mounted at `/servo/opsani.token`.

### Environment Variables

| Name | Description |
|------|-------------|
| **`OPSANI_OPTIMIZER`** | Configures the Opsani optimizer for the servo (required). Format is `example.com/app`. |
| `OPSANI_TOKEN` | Configures the Opsani API token for authenticating with the optimizer service (optional). |

Servo connectors support setting values through environment variables for every
attribute of their configuration class. See `servo schema` for details.

### Mounts

| Path | Description |
|------|-------------|
| **`/servo/servo.yaml`** | The servo configuration file (required). |
| `/servo/opsani.token` | A file containing the Opsani API token for authenticating with the optimizer service (optional). |

## Usage

To display help:

```bash
docker run --rm -i opsani/servox --help
```

Generating a config file:

```bash
docker run --rm -i -v $(pwd):$(pwd) opsani/servox \
generate -f $(pwd)/servo.yaml
```

Running a servo:

```bash
docker run --rm -i -v $(pwd)/servo.yaml:/servo/servo.yaml \
-e OPSANI_OPTIMIZER=example.com/app -e OPSANI_TOKEN=123456 opsani/servox
```

For full documentation see [ServoX](https://github.com/opsani/servox) on GitHub.

## Usage in Kubernetes

To display help:

```bash
kubectl run servo --rm --attach --restart=Never --image="opsani/servox" -- servo --help
```

Running a servo:

```bash
kubectl run servo --rm --attach --restart=Never --image="opsani/servox" -- \
--optimizer example.com/app --token 123456 run
```

## Usage in Docker Compose

This `docker-compose.yaml` file supports the configuration of the Opsani API
token directly from the `OPSANI_TOKEN` environment variable or via a file mount
where the source file is configured via the `OPSANI_TOKEN_FILE` environment
variable.

```yaml
version: '3.8'

services:
  servo:
    image: opsani/servox
    restart: always
    environment:
      - OPSANI_OPTIMIZER=${OPSANI_OPTIMIZER:?Opsani Optimizer must be configured}
      - OPSANI_TOKEN=${OPSANI_TOKEN}
    volumes:
      - type: bind
        source: ./servo.yaml
        target: /servo/servo.yaml
        read_only: true
      - type: bind
        source: ${OPSANI_TOKEN_FILE:-/dev/null}
        target: /servo/opsani.token
        read_only: true
```

## Adding connectors via inheritance

Additional connectors can be included into a servo assembly image by using
`opsani/servox` as a parent image. The servo image uses the
[Poetry](https://python-poetry.org/) package manager for Python and connectors
can be installed through standard package management. The servo uses Python
setuptools entrypoints to auto-discover connectors that are available in the
environment.

```dockerfile
FROM opsani/servox

RUN poetry add servo-notifiers
```

## License

Apache 2.0 - see the
[LICENSE](https://github.com/opsani/servox/blob/main/LICENSE) file for details.
