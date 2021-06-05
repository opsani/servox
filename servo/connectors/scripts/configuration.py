from typing import Dict, List, Optional
import servo

class ScriptsConfiguration(servo.BaseConfiguration):
    before: Optional[Dict[str, List[str]]]
    on: Optional[Dict[str, List[str]]]
    after: Optional[Dict[str, List[str]]]
