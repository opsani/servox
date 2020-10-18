# Voracious: multi-oco backend emulator for testing the phase 2 load generator

This utility emulates large number of oco backends for the purpose of testing the phase 2 load generator (multi-servo emulator). It can also be used to test the multi-app support in ServoX.

Voracious responds to two OCO API endpoints plus its own statistics endpoint:

* list of applications: returns a preconfigured number of applications (change N_APPS)
* servo: follows a very strict sequence of events (change ADJUST_PAYLOAD to adapt to emulator descriptor)
* statistics: returns statistics across all applications (this is just for human observation, not expected to be called by the multi-servo emulator)

It consumes a reasonably high number of servo streams (keep in mind that state is kept in memory for each application; the intended number of backends to be emulated is 5000)

To see the exact endpoints, see the swagger documentation/exerciser for Voracious at http://localhost:8080/docs#/

## Installation

Requires (see [requirements.txt](requirements.txt)):
- fastapi
- uvicorn
- PyYAML
- pydantic

Tested with Python 3.6.9 on Ubuntu.

Note: not tested in a clean environment using the above requirements.txt

## Configuration

There are 3 items that need to be adjusted in `voracious.py` to match the servo requests:

* application name forming template (the default is `appNNNNN` where NNNNN is a zero-padded integer)
* number of applications to return when asked to list (the default is 5)
* adjust payload with setting values (needs to match the descriptor sent by the servo emulator) (the default is the kubecon2018 demo configuration)

Voracious runs on port 8080 by default. Modify `run.sh` and `run-reload.sh` to modify.

## Usage

Run `uvicorn voracious:app --port=8080` or simply `./run.sh` (if modifying voracious.py, usr `./run-reload.sh` as it will reload the app whever the source file changes).

Note that Voracious is very strict on the expected sequence of events from each app's servo (much stricter than the protocol, as it is trying to test the emulator). After restarting voracious, all servos need to start with HELLO (this can be fixed if too limiting)

## Statistics

Voracious keeps count of how many adjust/measure cycles each application has gone through. It can return statistics over the 
counts so that the operation and evenness of the distribution of requests across apps can be verified.

Voracious returns the statistics on its root endpoint: e.g., `curl localhost:8080`. It can be watched, for example like this: 
`watch -n 3 curl -s localhost:8080`.

The payload looks like this: `{"n":4,"average":49.5,"min":47,"max":52}`

As long as the average, min and max are close, this means the application requests were reasonably evenly distributed. Check the number of apps, `n`, to confirm all expected applications are being reported.

## Test

### Servo-emulator
While Voracious is intended to be used with the phase 2 load generator (multi-servo emulator), you can see it work against one or a few traditional servo emulators. Use the attached `emu.sh` script to run the servo emulator, using the `master` branch of the `oco` repository. (Don't forget to download or symlink `servo` in the `servo-emulator` directory). To run multiple servo emulators, run each in a separate shell and set the application name like this:

`APP=app01234 ./emu.sh`

Note that the servo-emulator has a lot of 3rd party package requirements (requests, numpy, scipy, json_merge_patch, etc.);
it makes sense to test with it only if you have an oco environment set up.

Voracious was tested with up to 4 servo-emulators working concurrently.

### Performance

I was unable to get 100 of them working well but that seems to be a problem with the servo emulator, not voracious.

Voracious was tested with the `hey` load generator and it reached 812 req/sec with median response time of 120 msec. 

Test setup:

* laptop with 8-core Intel i5 CPU
* running Windows 10 and Ubuntu in the WSL subsystem (essentially, in a VM)
* running both voracious and hey on the same host
* single application emulated, with the app state manually advanced to measurement progress (so testing the full state machine but not the indexing into large set of applications)

Setup commands:
```
curl -X POST -d '{ "event" : "HELLO" }' http://localhost:8080/accounts/a/applications/app00000/servo
curl -X POST -d '{ "event" : "DESCRIPTION" }' http://localhost:8080/accounts/a/applications/app00000/servo
curl -X POST -d '{ "event" : "MEASUREMENT", "param": { "progress": 10 } }' http://localhost:8080/accounts/a/applications/app00000/servo
```

Test command:
```
hey -c 100 -n 10000 -m POST -d '{ "event" : "MEASUREMENT", "param": { "progress": 10 } }' http://localhost:808
0/accounts/a/applications/app00000/servo
```

## Potential improvements

* Verify operation in a clean environment
* Allow voracious to be restarted without restarting the servo emulators (sync state with all servos, without requiring HELLO)
* Determine if the `state` variable needs to be guarded against concurrent access (generally not expected to be needed)
* Build a container image (Dockerfile and docker-compose)