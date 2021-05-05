from typing import Dict, List, Optional
import functools
import servo

class ScriptsConfiguration(servo.BaseConfiguration):
    before: Optional[Dict[str, List[str]]]
    after: Optional[Dict[str, List[str]]]

class ScriptsConnector(servo.BaseConnector):
    config: ScriptsConfiguration

    @servo.on_event()
    async def startup(self) -> None:
        for preposition in (servo.Preposition.before, servo.Preposition.after):
            for event_name, scripts in getattr(self.config, str(preposition), {}).items():
                event = servo.events.EventContext.from_str(event_name)

                for script in scripts:
                    handler = functools.partial(_handler, script)
                    self.add_event_handler(event.event, preposition, handler)

async def _handler(script: str, *args, **kwargs) -> None:
    await servo.utilities.subprocess.run_subprocess_shell(script)
