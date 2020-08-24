#!/usr/bin/env bash
set -e

# Allow literal or volume mounted tokens
exec servo \
    --optimizer ${OPSANI_OPTIMIZER:?must be configured} \
    --config-file ${SERVO_CONFIG_FILE:-/servo/servo.yaml} \
    $(if [ ! -z ${OPSANI_TOKEN} ]; then \
        echo "--token ${OPSANI_TOKEN}"; \
      else \
        echo "--token-file /servo/opsani.token"; \
      fi) \
    "$@"
