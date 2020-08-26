import pytest
import inspect
from functools import reduce
from servo.utilities.inspect import get_instance_methods, get_methods, get_defining_class

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
    methods = get_instance_methods(cls, stop_at_parent=stop_at_parent)
    assert list(methods.keys()) == method_names

def test_get_instance_methods_invalid_parent() -> None:
    with pytest.raises(TypeError) as e:
        get_instance_methods(OneClass, stop_at_parent=int)
    assert str(e.value) == 'invalid parent type "<class \'int\'>": not found in inheritance hierarchy'

def test_get_instance_methods_returns_bound_methods_if_possible() -> None:
    methods = get_instance_methods(ThreeClass(), stop_at_parent=OneClass)
    assert list(methods.keys()) == ['one', 'two', 'three', 'four', 'five', 'six']
    assert reduce(lambda bound, m: bound & inspect.ismethod(m), methods.values(), True)
