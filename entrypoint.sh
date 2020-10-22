#!/usr/bin/env bash
set -e

# Allow literal or volume mounted tokens
exec servo \
    --config-file ${SERVO_CONFIG_FILE:-/servo/servo.yaml} \
    $(if [ ! -z ${OPSANI_TOKEN} ]; then \
        echo "--token ${OPSANI_TOKEN}"; \
      else \
        echo "--token-file ${OPSANI_TOKEN_FILE:-/servo/opsani.token}"; \
      fi) \
    "$@"
