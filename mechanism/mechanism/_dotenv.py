import os
import pathlib
import pydantic
import dotenv
import servo

from typing import Dict


# @pytest.fixture(scope='session')
# def path() -> pathlib.Path:
#     return DOTENV_PATH

# @pytest.fixture(scope='session')
# def values() -> Dict[str, str]:
#     return dotenv.dotenv_values(dotenv_path)

# requires_dotenv = pytest.mark.skipif(not dotenv_path.exists(), reason='Test dotenv file not found. Create tests/opsani.env')

# @pytest.fixture
# def optimizer(values: Dict[str, str]) -> servo.Optimizer:
#     return servo.Optimizer(
#         dotenv_values['OPSANI_OPTIMIZER'],
#         token=dotenv_values['OPSANI_TOKEN']
#     )

# def mark(str: environment):
#     ...


class TemporaryEnv:
    def __init__(self, env: Dict[str, str]) -> None:
        super().__init__()
        self._environ = os.environ.copy()
        os.environ.update(env)

    def restore(self) -> None:
        os.environ.clear()
        os.environ.update(self._environ)

    def __enter__(self) -> None:
        return TemporaryEnv(self.values)

    def __exit__(self, exc_type, exc_value, traceback):
        self.restore()


class Dotenv(pydantic.BaseModel):
    path: pathlib.Path

    def __init__(self, path: str) -> None:
        super().__init__(path=pathlib.Path(path).resolve())

    @property
    def values(self) -> Dict[str, str]:
        return dotenv.dotenv_values(self.path)

    def load(self) -> None:
        dotenv.load_dotenv(self.path, override=True)

    @property
    def optimizer(self) -> servo.Optimizer:
        return servo.Optimizer(
            self.values['OPSANI_OPTIMIZER'],
            token=self.values['OPSANI_TOKEN']
        )

    def __enter__(self) -> None:
        self._temp_env = TemporaryEnv(self.values)
        return self._temp_env

    def __exit__(self, exc_type, exc_value, traceback):
        self._temp_env.restore()
        del self._temp_env
