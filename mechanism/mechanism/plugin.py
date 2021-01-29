import pathlib
import pytest
import servo
import mechanism
import os
import dotenv

from typing import Optional

_optimizer: Optional[servo.Optimizer] = None

PATH_INI = (
    'path(path, exists=True, or_else=\'fail\'): '
    'require a file or directory at the specified path to match expectations. '
    'This marker enables conditional test execution based on filesystem state. '
    'Tests can be failed or skipped based on the presence or absence of files at '
    'a given path. The `exists` parameter indicates whether or not a file is '
    'expected to exist on the filesystem. The `or_else` parameter can be set to '
    '\'fail\' or \'skip\' to configure how tests should be handled when the path '
    'assertion fails to match the declared expectations.'
)

# TODO: Turn into generic dotenv marker...Maybe paired with an autouse fixture?
OPSANI_DOTENV_INI = (
    'opsani_dotenv(skippable=False): '
    'mark a test as requiring an Opsani dotenv file in order to execute. '
    'This marker checks for the existence of the Opsani dotenv file and '
    'conditionalizes test execution accordingly. By default, marked tests '
    'will fail if the dotenv file does not exist. The `skippable` keyword '
    'argument allows the tests to be skipped rather than failed.'
)

# TODO: support CLI arguments also
OPTIMIZER_INI = (
    'optimizer(id: Optional[str] = None, token: Optional[str] = None, *, env=False, dotenv=_from_config_option(), skippable=False) '
    'mark a test as requiring an Opsani Optimizer backend to execute. '
    'This marker allows you to configure an explicit optimizer backend '
    'programmatically, load the optimizer from the environment, or load '
    'the optimizer from a dotenv file. The `skippable` keyword '
    'argument allows marked tests to be skipped rather than failed if no optimizer can be resolved.'
)

def pytest_configure(config) -> None:
    """Register custom markers for use in the test suite."""
    config.addinivalue_line("markers", PATH_INI)
    config.addinivalue_line("markers", OPSANI_DOTENV_INI)
    config.addinivalue_line("markers", OPTIMIZER_INI)


def pytest_collection_modifyitems(config, items) -> None:
    dotenv_path = pathlib.Path(config.getoption('opsani_dotenv'))
    if not dotenv_path.is_absolute():
        dotenv_path = config.rootpath / dotenv_path

    for item in items:
        for mark in item.iter_markers(name='opsani_dotenv'):
            skippable = mark.kwargs.get('skippable', False)
            or_else = ('skip' if skippable else 'fail')
            item.add_marker(
                pytest.mark.path(dotenv_path, exists=True, or_else=or_else)
            )


def pytest_runtest_setup(item):
    """Run setup actions to prepare the test case.

    See Also:
        https://docs.pytest.org/en/latest/reference.html#_pytest.hookspec.pytest_runtest_setup
    """
    _setup_optimizer_marker(item)
    _setup_path_markers(item)


def _resolve_var_from_marker(var: str, item, marker) -> Optional[str]:
    if marker.kwargs.get('dotenv', True):
        dotenv_path = pathlib.Path(item.config.getoption('opsani_dotenv'))
        if dotenv_path.exists():
            values = dotenv.dotenv_values(dotenv_path)
            if value := values.get(var):
                return value

    if marker.kwargs.get('env', True):
        return os.environ.get(var)

    return None


def _setup_optimizer_marker(item) -> None:
    # Resolve the closest optimizer marker as authoritative
    marker = item.get_closest_marker("optimizer")
    if marker:
        # Explicit code overrides always wins
        debug("GOT MARKER: ", marker, marker.args)
        id, token = marker.args or (None, None)
        _optimizer = servo.Optimizer(
            id or _resolve_var_from_marker('OPSANI_OPTIMIZER', item, marker),
            token=token or _resolve_var_from_marker('OPSANI_TOKEN', item, marker)
        )
        debug("**** NOTE: Loaded up Optimizer: ", _optimizer)


def _setup_path_markers(item) -> None:
    for mark in item.iter_markers(name='path'):
        path_arg = mark.args[0]
        if callable(path_arg):
            path = pathlib.Path(path_arg()).resolve()
        else:
            path = pathlib.Path(mark.args[0]).resolve()
        exists = mark.kwargs.get('exists', True)
        or_else = mark.kwargs.get('or_else', 'fail')
        reason = mark.kwargs.get('reason', _reason_for_path_failure(path, exists))

        if or_else not in ('fail', 'skip'):
            raise ValueError(f"or_else argument must be 'fail' or 'skip'")

        if path.exists() != exists:
            if or_else == 'fail':
                # item.add_marker(pytest.mark.fail(reason=reason))
                pytest.fail(reason)
            elif or_else == 'skip':
                pytest.skip(reason)
                # item.add_marker(pytest.mark.skip(reason=reason))
            else:
                raise ValueError(f"unknown or_else value: {or_else}")


def pytest_addoption(parser):
    """Add options to pytest to configure mechanism.

    See Also:
        https://docs.pytest.org/en/latest/reference.html#_pytest.hookspec.pytest_addoption
    """

    group = parser.getgroup('mechanism', 'servox mechanism testing')
    group.addoption(
        '--opsani-dotenv',
        action='store',
        metavar='path',
        default='tests/opsani.env',
        help=(
            'path to a dotenv file containing opsani optimizer credentials. '
            'required for optimizer dependent integration/system tests'
        )
    )


def pytest_report_header(config) -> str:
    dotenv_file = config.getoption('opsani_dotenv')
    return f"servox mechanism dotenv: {dotenv_file}"


def _reason_for_path_failure(path: pathlib.Path, expected_to_exist: bool) -> str:
    descriptor = "to" if expected_to_exist else "not to"
    return f"expected path {descriptor} exist: {path}"


@pytest.fixture(name='mechanism')
def _mechanism(pytestconfig) -> mechanism.Fixture:
    dotenv_path = pathlib.Path(pytestconfig.getoption('opsani_dotenv'))
    if not dotenv_path.is_absolute():
        dotenv_path = pytestconfig.rootpath / dotenv_path

    fixture = mechanism.Fixture(
        dotenv=mechanism.Dotenv(dotenv_path),
        optimizer=_optimizer
    )
    try:
        value = yield fixture
        return value
    except Exception as e:
        debug("got: ", e)
        raise e



# TODO: Set default marks...
# Setup reusable marks
#pytestmark = [pytest.mark.asyncio, pytest.mark.integration, pytest.mark.path('tests/opsani.env', exists=True, or_else='skip')]
