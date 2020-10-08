import functools
import inspect
import sys
import types

import pytest
import servo.utilities.inspect

class OneClass:
    def one(self) -> None:
        ...
    
    def two(self) -> None:
        ...
    
    def three(self) -> None:
        ...
class TwoClass(OneClass):
    def four(self) -> None:
        ...
    
    def five(self) -> None:
        ...
class ThreeClass(TwoClass):
    def six(self) -> None:
        ...

@pytest.mark.parametrize(
    "cls, stop_at_parent, method_names",
    [
        (OneClass, None, ['one', 'two', 'three']),
        (TwoClass, None, ['four', 'five']),
        (TwoClass, OneClass, ['one', 'two', 'three', 'four', 'five']),
        (ThreeClass, OneClass, ['one', 'two', 'three', 'four', 'five', 'six']),
        (ThreeClass, TwoClass, ['four', 'five', 'six']),
    ]
)
def test_get_instance_methods(cls, stop_at_parent, method_names) -> None:
    methods = servo.utilities.inspect.get_instance_methods(cls, stop_at_parent=stop_at_parent)
    assert list(methods.keys()) == method_names

def test_get_instance_methods_invalid_parent() -> None:
    with pytest.raises(TypeError) as e:
        servo.utilities.inspect.get_instance_methods(OneClass, stop_at_parent=int)
    assert str(e.value) == 'invalid parent type "<class \'int\'>": not found in inheritance hierarchy'

def test_get_instance_methods_returns_bound_methods_if_possible() -> None:
    methods = servo.utilities.inspect.get_instance_methods(ThreeClass(), stop_at_parent=OneClass)
    assert list(methods.keys()) == ['one', 'two', 'three', 'four', 'five', 'six']
    assert functools.reduce(lambda bound, m: bound & inspect.ismethod(m), methods.values(), True)

def test_get_instance_methods_returns_finds_dynamic_instance_methods() -> None:
    def seven() -> None:
        ...
    instance = ThreeClass()
    instance.seven = types.MethodType(seven, instance)
    methods = servo.utilities.inspect.get_instance_methods(instance, stop_at_parent=OneClass)
    assert list(methods.keys()) == ['one', 'two', 'three', 'four', 'five', 'six', 'seven']
    assert functools.reduce(lambda bound, m: bound & inspect.ismethod(m), methods.values(), True)

def test_get_instance_methods_returns_ignores_attributes() -> None:
    class FourClass(ThreeClass):
        ignore_me: str = "ignore_me"

    instance = FourClass()
    methods = servo.utilities.inspect.get_instance_methods(instance, stop_at_parent=OneClass)
    assert list(methods.keys()) == ['one', 'two', 'three', 'four', 'five', 'six']
    assert functools.reduce(lambda bound, m: bound & inspect.ismethod(m), methods.values(), True)
    
def test_resolution_none() -> None:
    def test_type() -> None:
        ...
    
    def test_str() -> 'None':
        ...
    
    res_type, res_str = servo.utilities.inspect.resolve_type_annotations(
        inspect.Signature.from_callable(test_type).return_annotation, 
        inspect.Signature.from_callable(test_str).return_annotation
    )
    assert res_type == res_str
    
def test_resolution_none() -> None:
    def test_type() -> None:
        ...
    
    def test_str() -> 'None':
        ...
    
    res_type, res_str = servo.utilities.inspect.resolve_type_annotations(
        inspect.Signature.from_callable(test_type).return_annotation, 
        inspect.Signature.from_callable(test_str).return_annotation
    )
    assert res_type == res_str

def test_aliased_types() -> None:
    import servo
    import servo.types
    from servo import types
    from servo.types import Duration
    def test_type_path() -> servo.types.Duration:
        ...
    
    def test_type_abbr() -> types.Duration:
        ...
    
    def test_type() -> Duration:
        ...
    
    def test_str_path() -> 'servo.types.Duration':
        ...
    
    def test_str_abbr() -> 'types.Duration':
        ...
    
    def test_str() -> 'Duration':
        ...
    
    resolved = servo.utilities.inspect.resolve_type_annotations(
        inspect.Signature.from_callable(test_type_path).return_annotation, 
        inspect.Signature.from_callable(test_type_abbr).return_annotation,
        inspect.Signature.from_callable(test_type).return_annotation,
        inspect.Signature.from_callable(test_str_path).return_annotation, 
        inspect.Signature.from_callable(test_str_abbr).return_annotation,
        inspect.Signature.from_callable(test_str).return_annotation,
        globalns=globals(),
        localns=locals()
    )
    
    assert set(resolved) == {Duration}

# TODO: Compare compound return types, generic, skipping arguments...
# None, None.__class__, 'None'
# Optional[str], Dict[str, int], Dict[str, List[float]]
# omit argument, extra argument, argument with wrong type

# @pytest.mark.parametrize(
#     "reference_callable"
# )
import typing
from typing import Any, Dict, List

def test_equal_callable_descriptors() -> None:
    import servo
    import servo.types
    from servo import types
    from servo.types import Duration
    
    def test_one() -> dict:
        ...
    
    def test_two() -> typing.Dict[str, Any]:
        ...
    
    def test_three() -> typing.Dict[str, int]:
        ...
    
    def test_four() -> typing.Dict[float, str]:
        ...
    
    sig1 = inspect.Signature.from_callable(test_one)
    sig2 = inspect.Signature.from_callable(test_two)
    
    servo.utilities.inspect.assert_equal_callable_descriptors(
        servo.utilities.inspect.CallableDescriptor(signature=sig1, globalns=globals(), localns=locals()),
        servo.utilities.inspect.CallableDescriptor(signature=sig2, globalns=globals(), localns=locals())
    )
    
    servo.utilities.inspect.assert_equal_callable_descriptors(
        servo.utilities.inspect.CallableDescriptor(signature=inspect.Signature.from_callable(test_two), globalns=globals(), localns=locals()),
        servo.utilities.inspect.CallableDescriptor(signature=inspect.Signature.from_callable(test_three), globalns=globals(), localns=locals())
    )
    
    # before_handler_signature = inspect.Signature.from_callable(__before_handler)
    # servo.utilities.inspect.assert_equal_callable_descriptors(
    #     servo.utilities.inspect.CallableDescriptor(signature=before_handler_signature, module=event.module, globalns=event_globalns, localns=None),
    #     servo.utilities.inspect.CallableDescriptor(signature=handler_signature, module=handler_module, globalns=handler_globalns, localns=handler_localns),
    #     name=name,
    # )
    # servo.utilities.inspect.assert_equal_callable_descriptors()
    # ...