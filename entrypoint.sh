#!/usr/bin/env bash
# Copyright 2022 Cisco Systems, Inc. and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
