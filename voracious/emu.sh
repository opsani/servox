#!/bin/bash
APP=${APP-app00000}
#rm emulator.state
# The following descriptor exists in the master branch of oco
./servo --no-auth --delay 0.5 --url http://localhost:8080/accounts/a/applications/${APP}/servo ../env/emulator/test_apps/kubecon2018
