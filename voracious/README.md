# Voracious - multi-oco backend emulator for phase 2 load test

This utility emulates large number of oco backends for the purpose of testing the phase 2 load generator (multi-servo emulator).

Voracious responds to two endpoints:

* list of applications: returns a preconfigured number of applications (change N_APPS)
* servo: follows a very strict sequence of events (change ADJUST_PAYLOAD to adapt to emulator descriptor)

It consumes a reasonably high number of servo streams (keep in mind that state is kept in memory for each application; the intended number of backends to be emulated is 5000)

## Installation

Requires:
- fastapi
- uvicorn
- yaml
- pydantic

Tested with Python 3.6.9 on Ubuntu

## Configuration

There are 3 items that need to be adjusted in `voracious.py` to match the servo requests:

* application name forming template (the default is `appNNNNN` where NNNNN is a zero-padded integer)
* number of applications to return when asked to list (the default is 5)
* adjust payload with setting values (needs to match the descriptor sent by the servo emulator) (the default is the kubecon2018 demo configuration)

## Usage

Run `uvicorn voracious:app --port=8080`

Note that it is very strict on the expected sequence of events from each app's servo (much stricter than the protocol, as it is trying to test the emulator). After restarting voracious, all servos need to start with HELLO (this can be fixed if too limiting)

## Test

While Voracious is intended to be used with the phase 2 load generator (multi-servo emulator), you can see it work against one or a few traditional servo emulators. Use the attached `emu.sh` script to run the servo emulator, using the `master` branch of the `oco` repository. (Don't forget to download or symlink `servo` in the `servo-emulator` directory). To run multiple servo emulators, run each in a separate shell and set the application name like this:

`APP=app01234 ./emu.sh`

## Potential improvements

* Create requirements.txt and test in a clean environment
* Allow voracious to be restarted without restarting the servo emulators (sync state with all servos, without requiring HELLO)
* Determine if the `state` variable needs to be guarded against concurrent access (generally not expected to be needed)
* Collect and display statistics to demonstrate smooth and even load
* Package to be used without uvicorn (standalone) (TBD)