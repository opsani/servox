import pydantic
import mechanism._dotenv
import servo

from typing import Optional


class Fixture(pydantic.BaseModel):
    dotenv: mechanism._dotenv.Dotenv
    optimizer: Optional[servo.Optimizer]
