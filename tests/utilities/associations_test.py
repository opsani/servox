import pytest

from servo.utilities.associations import Associative, Mixin


class AssociativeObject(Mixin):
    pass


def test_association_init() -> None:
    obj = AssociativeObject()
    assert obj._associations == {}


def test_set_association() -> None:
    obj = AssociativeObject()
    obj._set_association("foo", 123)
    assert obj._associations == {"foo": 123}


def test_get_association() -> None:
    obj = AssociativeObject()
    obj._set_association("foo", 123)
    assert obj._get_association("foo") == 123


def test_get_association_unknown() -> None:
    obj = AssociativeObject()
    with pytest.raises(KeyError):
        obj._get_association("invalid")


def test_get_association_default() -> None:
    obj = AssociativeObject()
    assert obj._get_association("unknown", 1234) == 1234


def test_associations() -> None:
    obj = AssociativeObject()
    assert obj._associations is not None
    assert obj._associations == {}


def test_associative_protocol() -> None:
    obj = AssociativeObject()
    assert isinstance(obj, Associative)
