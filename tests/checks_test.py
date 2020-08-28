import pytest
import re
from datetime import datetime
from inspect import Signature
from servo.configuration import BaseConfiguration
from servo.checks import Check, BaseChecks, check
from servo.configuration import BaseConfiguration
from servo.checks import check as check_decorator, CheckHandlerResult
from servo.utilities.inspect import get_instance_methods
from typing import List, Tuple, Union, Optional


pytestmark = pytest.mark.freeze_time('2020-08-24')


def test_created_at_set_automatically() -> None:
    check = Check(name="Test", success=True)
    assert check.created_at == datetime(2020, 8, 24, 0, 0)

def test_serialize_with_exception() -> None:
    exception = RuntimeError("Testing")
    check = Check(name="Test", success=False, exception=exception)
    assert check.json() == (
        '{"name": "Test", "id": "1bab7e8d", "description": null, "required": false, "tags": null, "success": false, "message": null, "exception": "RuntimeError(\'Testing\')", "created_at": "2020-08-24T00:00:00", "run_at": null, "runtime": null}'
    )

async def test_inline_check() -> None:
    check = await Check.run("Test Inline Runner", handler=lambda: True)
    assert check
    assert check.name == "Test Inline Runner"
    assert check.success
    assert check.created_at == datetime(2020, 8, 24, 0, 0)

async def test_inline_check_failure() -> None:
    def failing_check() -> None:
        raise RuntimeError("Testing Failure")

    check = await Check.run("Test Inline Failure", handler=failing_check)
    assert check
    assert check.name == "Test Inline Failure"
    assert not check.success
    assert check.created_at == datetime(2020, 8, 24, 0, 0)

##
## Checks class specific

async def test_raises_on_invalid_method_name() -> None:
    class MeasureChecks(BaseChecks):
        def invalid_check(self) -> Check:
            return Check(name="Test", success=True)
        
    with pytest.raises(ValueError) as e:
        config = BaseConfiguration()
        await MeasureChecks.check(config)
    
    assert e
    assert str(e.value) == "method names of Checks subtypes must start with \"_\" or \"check_\""

async def test_allows_underscored_method_names() -> None:
    class MeasureChecks(BaseChecks):
        def _allowed_helper(self) -> Check:
            return Check(name="Test", success=True)
    
    config = BaseConfiguration()
    assert await MeasureChecks.check(config)

async def test_raises_on_invalid_signature() -> None:
    class MeasureChecks(BaseChecks):
        def check_invalid(self) -> int:
            return 123
    
    with pytest.raises(TypeError) as e:
        config = BaseConfiguration()
        await MeasureChecks.check(config)

    assert e
    assert str(e.value) == 'invalid signature for method "check_invalid": expected <Signature () -> servo.checks.Check>, but found <Signature () -> int>'

async def test_valid_checks() -> None:
    class MeasureChecks(BaseChecks):
        def check_something(self) -> Check:
            return Check(name="Test", success=True)
    
    config = BaseConfiguration()
    checks = await MeasureChecks.check(config)
    assert checks == [Check(name='Test', success=True, created_at=datetime(2020, 8, 24, 0, 0))]

async def test_run_as_instance() -> None:
    class MeasureChecks(BaseChecks):
        def check_something(self) -> Check:
            return Check(name="Test", success=True)
    
    config = BaseConfiguration()
    checker = MeasureChecks(config)
    checks = await checker.run()
    assert checks == [Check(name='Test', success=True, created_at=datetime(2020, 8, 24, 0, 0))]

async def test_check_ordering() -> None:
    class MeasureChecks(BaseChecks):
        def check_one(self) -> Check:
            return Check(name="1", success=True)
        
        def check_two(self) -> Check:
            return Check(name="2", success=False)
        
        def check_three(self) -> Check:
            return Check(name="3", success=True)
    
    config = BaseConfiguration()
    checks = await MeasureChecks.check(config)
    values = list(map(lambda c: (c.name, c.success), checks))
    assert values == [("1", True), ("2", False), ("3", True)]

async def test_check_aborts_on_failed_requirement() -> None:
    class MeasureChecks(BaseChecks):
        def check_one(self) -> Check:
            return Check(name="1", success=True)
        
        def check_two(self) -> Check:
            return Check(name="2", success=False, required=True)
        
        def check_three(self) -> Check:
            return Check(name="3", success=True)
    
    config = BaseConfiguration()
    checks = await MeasureChecks.check(config)
    values = list(map(lambda c: (c.name, c.success), checks))
    assert values == [("1", True), ("2", False)]

class NamedChecks(BaseChecks):
    @check_decorator("Check connectivity")
    def check_connectivity(self) -> CheckHandlerResult:
        return True

    @check("Verify permissions")
    def check_permissions(self) -> None:
        ...

    @check("Ensure adequate resources")
    def check_resources(self) -> None:
        ...

@pytest.mark.parametrize(
    "return_value, success, message",
    [
        ("this is the message", True, "this is the message"),
        (True, True, None),
        ((False, "didn't work"), False, "didn't work"),
        (None, True, None),
    ]
)
def test_valid_check_decorator_return_values(return_value, success, message) -> None:    
    @check_decorator("Test decorator")
    def check_test() -> CheckHandlerResult:
        return return_value
    
    check = check_test()
    assert check
    assert isinstance(check, Check)
    assert check.success == success
    assert check.message == message
    assert check.exception is None

@pytest.mark.parametrize(
    "return_value, exception_type, message",
    [
        (123, ValueError, (
            "caught exception: ValueError('check method returned unexpected value of type \"int\"')"
        )),
        ((False, 187), ValueError, (
            "caught exception: ValidationError(model='Check', errors=[{'loc': ('message',), 'msg': 'str type expected'"
            ", 'type': 'type_error.str'}])"
        )),
        ((666, "fail"), ValueError, (
            "caught exception: ValidationError(model='Check', errors=[{'loc': ('success',), 'msg': 'value could not be"
            " parsed to a boolean', 'type': 'type_error.bool'}])"
        )),
    ]
)
def test_invalid_check_decorator_return_values(return_value, exception_type, message) -> None:
    @check_decorator("Test decorator")
    def check_test() -> CheckHandlerResult:
        return return_value
    
    check = check_test()
    assert check
    assert isinstance(check, Check)
    assert check.success == False    
    assert check.message == message
    assert check.exception is not None
    assert isinstance(check.exception, exception_type)

class ValidHandlerSignatures:
    def check_none(self) -> None:
        ...
    
    def check_str(self) -> str:
        ...
    
    def check_bool(self) -> bool:
        ...
    
    def check_tuple(self) -> Tuple[bool, str]:
        ...
    
    def check_union(self) -> Union[str, bool]:
        ...
    
    def check_optional(self) -> Optional[str]:
        ...
    
    def check_union_of_tuple(self) -> Union[str, Tuple[bool, str]]:
        ...
    
    def check_optional_tuple(self) -> Optional[Tuple[bool, str]]:
        ...

@pytest.mark.parametrize(
    "method",
    get_instance_methods(ValidHandlerSignatures()).values(),
    ids=get_instance_methods(ValidHandlerSignatures()).keys()
)
def test_valid_signatures(method) -> None:
    check_decorator(method.__name__)(method)

class InvalidHandlerSignatures:
    def check_int(self) -> int:
        ...
    
    def check_list(self) -> List[Check]:
        ...
    
    def check_invalid_tuple(self) -> Tuple[int, str]:
        ...
    
    def check_invalid_optional(self) -> Optional[float]:
        ...
    
    def check_invalid_union(self) -> Union[bool, str, float]:
        ...
    
    def check_invalid_union_with_tuple(self) -> Union[bool, Tuple[str, float]]:
        ...

@pytest.mark.parametrize(
    "method",
    get_instance_methods(InvalidHandlerSignatures()).values()
)
def test_invalid_signatures(method) -> None:
    with pytest.raises(TypeError) as e:
        check_decorator(method.__name__)(method)

    sig = Signature.from_callable(method)
    message = f'invalid check handler "{method.__name__}": incompatible return type annotation in signature {repr(sig)}, expected to match <Signature () -> Union[bool, str, Tuple[bool, str], NoneType]>'
    assert str(e.value) == message

def test_decorating_invalid_signatures() -> None:
    with pytest.raises(TypeError) as e:
        @check_decorator("Test decorator")
        def check_test() -> int:
            ...
    
    assert e
    assert str(e.value) == (
        'invalid check handler "check_test": incompatible return type annotation in signature <Signature () -> int>, e'
        'xpected to match <Signature () -> Union[bool, str, Tuple[bool, str], NoneType]>'
    )

@pytest.mark.freeze_time('2020-08-25', auto_tick_seconds=15)
async def test_check_timer() -> None:
    @check_decorator("Check timer")
    def check_test() -> None:
        ...
    
    check = check_test()
    assert check
    assert isinstance(check, Check)
    assert check.run_at == datetime(2020, 8, 25, 0, 0, 15)
    assert check.runtime == "15s"

@pytest.mark.freeze_time('2020-08-25', auto_tick_seconds=15)
async def test_decorate_async() -> None:
    @check_decorator("Check async")
    async def check_test() -> None:
        ...
    
    check = await check_test()
    assert check
    assert isinstance(check, Check)
    assert check.run_at == datetime(2020, 8, 25, 0, 0, 15)
    assert check.runtime == "15s"

async def test_run_check_by_name() -> None:
    nc = NamedChecks(BaseConfiguration())
    checks = await nc.run(name="Check connectivity")
    check = checks[0]
    assert check
    assert check.name == "Check connectivity"
    assert check.success

def test_generate_check_id() -> None:
    check = Check(name="Ensure adequate resources", success=True)
    assert check.id == "c272d5e0"

def test_decorator_sets_id_to_method_name() -> None:
    checks = NamedChecks(BaseConfiguration())
    assert checks.check_connectivity.__check__.id == "check_connectivity"

async def test_run_check_by_id() -> None:
    nc = NamedChecks(BaseConfiguration())
    checks = await nc.run(id="check_connectivity")
    assert len(checks) == 1
    check = checks[0]
    assert check
    assert check.name == "Check connectivity"
    assert check.success

class FilterableChecks(BaseChecks):
    @check("name-only")
    def check_one(self) -> None:
        ...
    
    @check("name-and-id", id="explicit-id")
    def check_two(self) -> None:
        ...
    
    @check("name-and-tags", tags=["one", "two"])
    def check_three(self) -> None:
        ...
    
    @check("name-and-identicial-tags", tags=["one", "two"])
    def check_four(self) -> None:
        ...
    
    @check("name-and-exclusive-tags", tags=["three", "four"])
    def check_five(self) -> None:
        ...
    
    @check("name-and-intersecting-tags", tags=["one", "four"])
    def check_six(self) -> None:
        ...

@pytest.mark.parametrize(
    "name, id, tags, expected_ids",
    [
        # name cases
        ("name-only", None, None, ["check_one"]),
        ("invalid", None, None, []),
        (("name-only", "name-and-id"), None, None, ["check_one", "explicit-id"]),
        (["name-only", "name-and-id"], None, None, ["check_one", "explicit-id"]),
        (re.compile('.*tags'), None, None, ['check_three', 'check_four', 'check_five', 'check_six']),
        
        # id cases
        (None, 'explicit-id', None, ['explicit-id']),
        (None, ('explicit-id', 'check_three'), None, ['explicit-id', 'check_three']),
        (None, re.compile('[i]+'), None, ['explicit-id', 'check_five', 'check_six']),

        # tag cases
        (None, None, ["one"], ['check_three','check_four','check_six']),
        (None, None, set(), []),
        (None, None, ("one", "four"), ["check_three", "check_four", "check_five", "check_six"]),
        (None, None, {"invalid"}, []),

        # compound cases
        (None, re.compile('[i]+'), {"four"}, ['check_five', 'check_six']),
        (re.compile('exclusive'), re.compile('[i]+'), None, ['check_five']),
    ]
)
async def test_filtering(name, id, tags, expected_ids) -> None:
    checks = await FilterableChecks.check(BaseConfiguration(), name=name, id=id, tags=tags)    
    ids = list(map(lambda c: c.id, checks))
    assert len(ids) == len(expected_ids)
    assert ids == expected_ids

class RequirementChecks(BaseChecks):
    @check("required-1", required=True)
    def check_one(self) -> None:
        ...
    
    @check("not-required-1")
    def check_two(self) -> None:
        ...
    
    @check("not-required-2")
    def check_three(self) -> None:
        raise RuntimeError("fail check")
    
    @check("required-2", required=True)
    def check_four(self) -> None:
        raise RuntimeError("fail check")
    
    @check("required-3", required=True)
    def check_five(self) -> None:
        ...
    
    @check("not-required-3")
    def check_six(self) -> None:
        ...

@pytest.mark.parametrize(
    "name, all, expected_results",
    [
        # no filter, halt at not-required-2
        (None, False, {
            'required-1': True,
            'not-required-1': True,
            'not-required-2': False,
            'required-2': False,
        }),
        # no filter, continue to end
        (None, True, {
            'required-1': True,
            'not-required-1': True,
            'not-required-2': False,
            'required-2': False,
            'required-3': True,
            'not-required-3': True,
        }),
        # run not-required-1, trigger 1 requirement, no failures
        ("not-required-1", False, {'not-required-1': True, 'required-1': True}),
        # run not-required-2, trigger 1 requirement, fail
        ("not-required-2", True, {'not-required-2': False, 'required-1': True}),
        # run required-3, trigger 2 requirements, halt at required-2
        ("not-required-3", False, {'required-1': True, 'required-2': False}),
        # run all required-3, trigger 2 requirements, required-2 fails
        ("not-required-3", True, {
            'required-1': True,
            'required-2': False,
            'required-3': True,
            'not-required-3': True,
        }),
        # run not-required-1 and not-required-3
        (("not-required-1", "not-required-3"), False, {
            'required-1': True,
            'not-required-1': True,
            'required-2': False,
        }),
        (("not-required-1", "not-required-3"), True, {
            'required-1': True,
            'not-required-1': True,
            'required-2': False,
            'required-3': True,
            'not-required-3': True,
        }),
    ]
)
async def test_running_requirements(name, all, expected_results) -> None:
    checks = await RequirementChecks.check(BaseConfiguration(), name=name, all=all)
    actual_results = dict(map(lambda c: (c.name, c.success), checks))
    assert actual_results == expected_results

class MixedChecks(BaseChecks):
    @check("one")
    def check_one(self) -> None:
        ...
    
    def check_two(self) -> Check:
        return Check(name="two", success=True)
    
    @check("three")
    def check_three(self) -> None:
        ...
    
    def check_four(self) -> Check:
        return Check(name="four", success=True)

@pytest.mark.parametrize(
    "name, expected_results",
    [
        ("one", ["one", "two", "four"]),
        ("three", ["two", "three", "four"]),
        ("unknown", ["two", "four"]),
    ]
)
async def test_mixed_checks(name, expected_results) -> None:
    checks = await MixedChecks.check(BaseConfiguration(), name=name)
    actual_results = list(map(lambda c: c.name, checks))
    assert actual_results == expected_results

from servo.checks import create_checks_from_iterable
async def test_generate_checks() -> None:
    handler = lambda c: f"so_check_it_{c}"
    items = ["one", "two", "three"]
    ItemChecks = create_checks_from_iterable(handler, items)
    checker = ItemChecks(BaseConfiguration())
    results = await checker.run()
    assert len(results) == 3
    messages = list(map(lambda c: c.message, results))
    assert messages == ["so_check_it_one", "so_check_it_two", "so_check_it_three"]

