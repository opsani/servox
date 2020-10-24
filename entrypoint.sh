#!/usr/bin/env bash
set -e

if [ $(grep -c '\-\-\-' ${SERVO_CONFIG_FILE:-/servo/servo.yaml}) -ne 0 ]; then
  # In multi-servo config everything has to be in the file
  exec servo \
    --config-file ${SERVO_CONFIG_FILE:-/servo/servo.yaml} \
    $(if [ ! -z ${OPSANI_TOKEN} ]; then \
        echo "--token ${OPSANI_TOKEN}"; \
      else \
        echo "--token-file ${OPSANI_TOKEN_FILE:-/servo/opsani.token}"; \
      fi) \
    "$@"
else
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
fi
