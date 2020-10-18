from typing import Dict, List, Optional, Union
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import yaml

# -------------------
# Responses - EDIT ME

N_APPS=5    # number of applications to return on list
ADJUST_PAYLOAD = yaml.safe_load(    # payload to send for all adjust commands - must match emulator
    """
    state:
        application:
            components:
                web:
                    settings:
                        cpu: { value: 1 }
                        replicas: { value: 3 }
                java:
                    settings:
                        mem: { value: 2 }
                        GCTimeRatio: { value: 99 }
                db:
                    settings:
                        cpu: { value: 1 }
                        commit_delay: { value: 0 }
    """
)

# -----------------------------------------------
def app_name(no: int):
    return f"app{no:05}"

def app_index(name: str):
    if not name.startswith("app"):
        raise ValueError(f"Invalid application name {name} (missing prefix)")
    try:
        n = int(name[3:])
    except Exception as e:
        raise ValueError(f"Invalid application name {name} (parsing: {e})")
    if n >= N_APPS:
        raise ValueError(f"Invalid application name {name} ({n} > {N_APPS-1})")
    return n

class AppListItem(BaseModel):
    mode: Optional[str]
    mode_limit: Optional[str]
    state: str

class MosocResponse(BaseModel):
    status: str
    message: str
    data: Union[Dict[str, AppListItem]]

class ServoEvent(BaseModel):
    event: str
    param: Union[None, Dict]

class ServoNotifyResponse(BaseModel):
    status: str
    message: Optional[str]

class ServoCommandResponse(BaseModel):
    cmd: str
    param: Dict

class VoraciousStatistics(BaseModel):
    n: int
    average: Optional[float]
    min: Optional[int]
    max: Optional[int]


class SeqError(Exception):
    pass

class App:
    name: str
    state: str = "initial"
    next_cmd: Optional[str] = None  # if None, WHATS_NEXT is invalid
    curr_cmd: Optional[str] = None
    curr_cmd_progress: Optional[int] = None
    n_cycles: int = 0

    fsmtab = {
        # state             event           progress  new state          next command  cycle++
        ("initial"        , "HELLO"       , False  ): ("hello-received", "DESCRIBE" , False ),
        ("hello-received" , "DESCRIPTION" , False  ): ("start-measure" , "MEASURE"  , False ),
        ("start-measure"  , "MEASUREMENT" , True   ): ("in-measure"    , None       , False ),
        ("in-measure"     , "MEASUREMENT" , True   ): ("in-measure"    , None       , False ),
        ("in-measure"     , "MEASUREMENT" , False  ): ("start-adjust"  , "ADJUST"   , True  ),
        ("start-adjust"   , "ADJUSTMENT"  , True   ): ("in-adjust"     , None       , False ),
        ("in-adjust"      , "ADJUSTMENT"  , True   ): ("in-adjust"     , None       , False ),
        ("in-adjust"      , "ADJUSTMENT"  , False  ): ("start-measure" , "MEASURE"  , False ),

        ("hello-received" , "GOODBYE"     , False  ): ("terminated"    , None       , False ),
        ("start-measure"  , "GOODBYE"     , False  ): ("terminated"    , None       , False ),
        ("in-measure"     , "GOODBYE"     , False  ): ("terminated"    , None       , False ),
        ("start-adjust"   , "GOODBYE"     , False  ): ("terminated"    , None       , False ),
        ("in-adjust"      , "GOODBYE"     , False  ): ("terminated"    , None       , False ),

        ("terminated"     , "HELLO"       , False  ): ("hello-received", "DESCRIBE" , False )
    }

    def __init__(self, name: str):
        self.name = name

    def _exc(self, msg: str):
        return SeqError(f"Sequence error for {self.name}: {msg}")

    def feed(self, ev: ServoEvent) -> Union[ServoNotifyResponse, ServoCommandResponse]:
        # handle next command query
        if ev.event == "WHATS_NEXT":
            if self.next_cmd == None:
                raise self._exc(f"Unexpected WHATS_NEXT in {self.state}")
            (cmd, self.next_cmd) = (self.next_cmd, None) # clear next_cmd
            self.curr_cmd = None
            self.curr_cmd_progress = None
            if cmd == "ADJUST":
                params = ADJUST_PAYLOAD
            else:
                params = {}
            return ServoCommandResponse(cmd = cmd, param = params)

        # lookup event and extract sequencer instruction
        try:
            progress = int(ev.param["progress"])
        except Exception:
            progress = None
        key = (self.state, ev.event, progress is not None)
        try:
            instr = self.fsmtab[key]
        except KeyError:
            pmsg = "" if progress is None else f"({progress}%)"
            raise self._exc(f"Unexpected event {ev}{pmsg} in state {self.state}")

        # check progress
        if progress is not None:
            if self.curr_cmd_progress is not None and progress < self.curr_cmd_progress:
                raise self._exc(f"Progress going backwards: old={self.curr_cmd_progress}, new={progress}")
            else:
                self.curr_cmd_progress = progress
        else:
            curr_cmd_progress = None # zero out progress on completed commands

        # "execute" instruction
        (self.state, self.next_cmd, bump_cycle) = instr 
        if bump_cycle:
            self.n_cycles += 1

        print(f"#seq# {self.name} processed {key} -> {instr}, {self.n_cycles} cycles {' [TERMINATED]' if self.state == 'terminated' else ''}")

        return ServoNotifyResponse(status = "ok")

# --- API Server

app = FastAPI()
state = {} # TODO determine if state needs to be guarded from concurrent access

@app.get("/")
async def root() -> VoraciousStatistics:
    cycles = [ x.n_cycles for x in state.values() ]
    n_apps = len(cycles)
    if n_apps == 0:
        return VoraciousStatistics(n = 0)
    min_ = max_ = sum_ = None
    for c in cycles:
        if min_ is None:
            min_ = max_ = sum_ = c
        else:
            if c < min_: min_ = c
            if c > max_: max_ = c
            sum_ += c
    return VoraciousStatistics(n = n_apps, average = float(sum_)/n_apps, min = min_, max = max_)


    return {"message": "Hello World... Oh, my!"}

@app.get("/accounts/{account}/application-overrides")
async def list(account: str):
    app = AppListItem(state = "active")
    return { app_name(n):app for n in range(N_APPS)}

@app.post("/accounts/{account}/applications/{app}/servo")
async def servo_get(account: str, app: str, ev: ServoEvent) -> Union[ServoNotifyResponse, ServoCommandResponse]:
    if app not in state:
        if ev.event == "HELLO":
            state[app] = App(name = app)
            print(f"Registered new application: {app}")
            # fall through to process event
        else:
            msg = f"Received event {ev.event} for unknown app {app}"
            print(msg)
            raise HTTPException(status_code=400, detail=msg)

    try:
        r = state[app].feed(ev) 
    except Exception as e:
        print(str(e))
        raise HTTPException(status_code=400, detail=str(e))

    return r