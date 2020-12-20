#!/usr/bin/env bash
set -e

# Allow literal or volume mounted tokens based on env
# In multi-servo mode, the config file contains optimizer + token details
OPSANI_TOKEN_FILE=${OPSANI_TOKEN_FILE:-/servo/opsani.token}
exec servo \
    --config-file ${SERVO_CONFIG_FILE:-/servo/servo.yaml} \
    $(if [ ! -z ${OPSANI_OPTIMIZER} ]; then \
        echo "--optimizer ${OPSANI_OPTIMIZER}"; \
      fi) \
    $(if [ ! -z ${OPSANI_TOKEN} ]; then \
        echo "--token ${OPSANI_TOKEN}"; \
      elif [ -f ${OPSANI_TOKEN_FILE} ]; then \
        echo "--token-file ${OPSANI_TOKEN_FILE}"; \
      fi) \
    "$@"
