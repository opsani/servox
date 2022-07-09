#!/usr/bin/env bash
set -e

# Allow literal or volume mounted tokens based on env
# In multi-servo mode, the config file contains optimizer + token details
OPSANI_TOKEN_FILE=${OPSANI_TOKEN_FILE:-/servo/opsani.token}

cat << EOF > /servo/servo.yaml
optimizer:
  workloadId: ${WORKLOAD_ID}
  endpoint: ${BASE_URL}
  oauth2client:
    clientId: ${OAUTH_CLIENT_ID}
    clientSecret: ${OAUTH_SECRET}
    tenantId: ${TENANT_ID}
EOF

exec servo \
    --config-file ${SERVO_CONFIG_FILE:-/servo/servo.yaml} $@

