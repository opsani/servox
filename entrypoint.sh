#!/usr/bin/env bash
set -e

# In multi-servo mode, the config file contains optimizer + token details
exec servo "$@"
