import abc
import pytest
import re

from servo.types.settings import *
from servo.types.settings import _is_step_aligned

class BasicSetting(Setting):
    name = "foo"
    type = "bar"

    def __opsani_repr__(self) -> dict:
        return {}


class TestSetting:
    def test_is_abstract_base_class(self) -> None:
        assert issubclass(Setting, abc.ABC)

    def test_requires_opsani_repr_implementation(self) -> None:
        assert "__opsani_repr__" in Setting.__abstractmethods__

    def test_validates_all(self) -> None:
        assert Setting.__config__.validate_all

    def test_validates_assignment(self) -> None:
        assert Setting.__config__.validate_assignment

    def test_human_readable_value(self) -> None:
        setting = BasicSetting(value="whatever")
        assert setting.human_readable_value == "whatever"

        class HumanReadableTestValue(str):
            def human_readable(self) -> str:
                return "another-value"

        setting = BasicSetting(value=HumanReadableTestValue("whatever"))
        assert setting.human_readable_value == "another-value"

    @pytest.mark.parametrize(
        ("pinned", "init_value", "new_value", "error_message"),
        [
            (True, None, None, None),
            (False, None, None, None),
            (True, None, 123, None),
            (False, None, 123, None),
            (True, 60, 60, None),
            (False, 60, 60, None),
            (True, 5.0, 5.0, None),
            (False, 5.0, 5.0, None),
            (True, "this", "this", None),
            (False, "this", "this", None),
            # Failures
            (
                True,
                1,
                None,
                "value of pinned settings cannot be changed: assigned value None is not equal to existing value 1",
            ),
            # FIXME: Type compare (True, 1, 1.0, 'value of pinned settings cannot be changed: assigned value 1.0 is not equal to existing value 1'),
            (
                True,
                5.0,
                2.5,
                "value of pinned settings cannot be changed: assigned value 2.5 is not equal to existing value 5.0",
            ),
            (
                True,
                5.0,
                3,
                "value of pinned settings cannot be changed: assigned value 3 is not equal to existing value 5.0",
            ),
            (
                True,
                1,
                "1",
                "value of pinned settings cannot be changed: assigned value '1' is not equal to existing value 1",
            ),
            (
                True,
                "something",
                "another",
                "value of pinned settings cannot be changed: assigned value 'another' is not equal to existing value 'something'",
            ),
            (
                True,
                "something",
                5.0,
                "value of pinned settings cannot be changed: assigned value 5.0 is not equal to existing value 'something'",
            ),
            # Coerced types
            (True, 5.0, 5, None),
            (True, 5, 5.0, None),
            (
                True,
                "5",
                5.0,
                "value of pinned settings cannot be changed: assigned value 5.0 is not equal to existing value '5'",
            ),
        ],
    )
    def test_validate_pinned_value_cannot_change(
        self,
        pinned: bool,
        init_value: Union[str, Numeric],
        new_value: Union[str, Numeric],
        error_message: str,
    ) -> None:
        setting = BasicSetting(value=init_value, pinned=pinned)
        if error_message is not None:
            assert pinned, "Cannot test validation for non-pinned settings"

            with pytest.raises(pydantic.ValidationError) as error:
                setting.value = new_value

            assert error
            assert "1 validation error for BasicSetting" in str(error.value)
            assert error.value.errors()[0]["loc"] == ("value",)
            assert error.value.errors()[0]["msg"] == error_message
        else:
            setting.value = new_value
            assert setting.value == new_value


class TestEnumSetting:
    @pytest.fixture
    def setting(self) -> EnumSetting:
        return EnumSetting(
            unit="unit", name="the-setting", values=["one", "two", "three"], value="two"
        )

    def test_type(self, setting: EnumSetting) -> None:
        assert setting.type == "enum"

    def test_opsani_repr(self, setting: EnumSetting) -> None:
        assert setting.__opsani_repr__() == {
            "the-setting": {
                "pinned": False,
                "type": "enum",
                "unit": "unit",
                "value": "two",
                "values": ["one", "two", "three"],
            },
        }

    def test_unit_value_are_excluded_from_opsani_repr_if_none(self) -> None:
        setting = EnumSetting(
            unit=None, name="the-setting", values=["one", "two", "three"], value=None
        )
        assert setting.__opsani_repr__() == {
            "the-setting": {
                "pinned": False,
                "type": "enum",
                "values": ["one", "two", "three"],
            },
        }

    def test_validate_type_must_be_enum(self, setting: EnumSetting) -> None:
        with pytest.raises(pydantic.ValidationError) as error:
            EnumSetting(type="foo", name="bar", values=["test"])

        assert error
        assert "1 validation error for EnumSetting" in str(error.value)
        assert error.value.errors()[0]["loc"] == ("type",)
        assert error.value.errors()[0]["type"] == "value_error.const"
        assert error.value.errors()[0]["msg"] == "unexpected value; permitted: 'enum'"

    def test_validate_values_list_is_not_empty(self) -> None:
        with pytest.raises(pydantic.ValidationError) as error:
            EnumSetting(name="bar", values=[])

        assert error
        assert "1 validation error for EnumSetting" in str(error.value)
        assert error.value.errors()[0]["loc"] == ("values",)
        assert error.value.errors()[0]["type"] == "value_error.list.min_items"
        assert (
            error.value.errors()[0]["msg"] == "ensure this value has at least 1 items"
        )

    def test_validate_value_is_included_in_values_list(self) -> None:
        with pytest.raises(pydantic.ValidationError) as error:
            EnumSetting(name="bar", values=["one", "two"], value="three")

        assert error
        assert "1 validation error for EnumSetting" in str(error.value)
        assert error.value.errors()[0]["loc"] == ("__root__",)
        assert error.value.errors()[0]["type"] == "value_error"
        assert (
            error.value.errors()[0]["msg"]
            == "invalid value: 'three' is not in the values list ['one', 'two']"
        )

class TestRangeSetting:
    def test_type(self) -> None:
        setting = RangeSetting(name="bar", min=0.0, max=1.0, step=0.1)
        assert setting.type == "range"

    def test_validate_type_must_be_enum(self) -> None:
        with pytest.raises(pydantic.ValidationError) as error:
            RangeSetting(type="foo", name="bar", min=0.0, max=1.0, step=0.1)

        assert error
        assert "1 validation error for RangeSetting" in str(error.value)
        assert error.value.errors()[0]["loc"] == ("type",)
        assert error.value.errors()[0]["type"] == "value_error.const"
        assert error.value.errors()[0]["msg"] == "unexpected value; permitted: 'range'"

    def test_validate_step_alignment_suggestion(self) -> None:
        with pytest.raises(pydantic.ValidationError, match=re.escape("RangeSetting('invalid' 3.0-11.0, 3.0) min/max difference is not step aligned: 8.0 is not a multiple of 3.0 (consider min 5.0 or 2.0, max 9.0 or 12.0).")):
            RangeSetting(name="invalid", min=3.0, max=11.0, step=3.0)

    @pytest.mark.parametrize(
        ("min", "max", "step", "error_message"),
        [
            (0, 5, 1, None),
            (2.0, 3.0, 1.0, None),
            (
                3.0,
                11.0,
                3.0,
                "RangeSetting('invalid' 3.0-11.0, 3.0) min/max difference is not step aligned: 8.0 is not a multiple of 3.0 (consider min 5.0 or 2.0, max 9.0 or 12.0).",
            ),
            (
                3.0,
                13.0,
                3.0,
                "RangeSetting('invalid' 3.0-13.0, 3.0) min/max difference is not step aligned: 10.0 is not a multiple of 3.0 (consider min 4.0 or 1.0, max 12.0 or 15.0).",
            ),
        ],
    )
    def test_validate_step_alignment(
        self, min: Numeric, max: Numeric, step: Numeric, error_message: str
    ) -> None:
        if error_message is not None:
            with pytest.raises(pydantic.ValidationError) as error:
                RangeSetting(name="invalid", min=min, max=max, step=step)

            assert error
            assert "1 validation error for RangeSetting" in str(error.value)
            assert error.value.errors()[0]["loc"] == ("__root__",)
            assert error.value.errors()[0]["type"] == "value_error"
            assert error.value.errors()[0]["msg"] == error_message
        else:
            RangeSetting(name="valid", min=min, max=max, step=step)

    @pytest.mark.parametrize(
        ("min", "max", "step", "error_message"),
        [
            (1, 5, 1, None),
            (0.0, 6.0, 2.0, None),
            (
                1.0,
                2,
                3,
                "invalid range: min, max, and step must all be of the same Numeric type (float: min. int: max, step.)",
            ),
            (
                1,
                2.0,
                3,
                "invalid range: min, max, and step must all be of the same Numeric type (int: min, step. float: max.)",
            ),
            (
                1,
                2,
                3.0,
                "invalid range: min, max, and step must all be of the same Numeric type (int: min, max. float: step.)",
            ),
        ],
    )
    def test_validate_all_elements_of_range_are_same_type(
        self, min: Numeric, max: Numeric, step: Numeric, error_message: str
    ) -> None:
        if error_message is not None:
            with pytest.raises(pydantic.ValidationError) as error:
                RangeSetting(name="invalid", min=min, max=max, step=step)

            assert error
            assert "1 validation error for RangeSetting" in str(error.value)
            assert error.value.errors()[0]["loc"] == ("__root__",)
            assert error.value.errors()[0]["type"] == "type_error"
            assert error.value.errors()[0]["msg"] == error_message
        else:
            RangeSetting(name="valid", min=min, max=max, step=step)

    @pytest.mark.parametrize(
        ("min", "max", "step", "value", "error_message"),
        [
            (1, 3, 1, 2, None),
            (0, 10, 1, 0, None),  # Check == min
            (0, 10, 1, 10, None),  # Check == max
            (1, 5, 1, 10, "invalid value: 10 is outside of the range 1-5"),
            (1, 10, 1, 0, "invalid value: 0 is outside of the range 1-10"),
            # Float values
            (1.0, 3.0, 1.0, 2.0, None),
            (0.0, 10.0, 1.0, 0.0, None),  # Check == min
            (0.0, 10.0, 1.0, 10.0, None),  # Check == max
            (
                1.0,
                5.0,
                1.0,
                10.0,
                "invalid value: 10.0 is outside of the range 1.0-5.0",
            ),
            (
                1.0,
                10.0,
                1.0,
                0.0,
                "invalid value: 0.0 is outside of the range 1.0-10.0",
            ),
        ],
    )
    def test_value_falls_in_range(
        self,
        min: Numeric,
        max: Numeric,
        step: Numeric,
        value: Numeric,
        error_message: str,
    ) -> None:
        if error_message is not None:
            with pytest.raises(pydantic.ValidationError) as error:
                RangeSetting(name="invalid", min=min, max=max, step=step, value=value)

            assert error
            assert "1 validation error for RangeSetting" in str(error.value)
            assert error.value.errors()[0]["loc"] == ("__root__",)
            assert error.value.errors()[0]["type"] == "value_error"
            assert error.value.errors()[0]["msg"] == error_message
        else:
            RangeSetting(name="valid", min=min, max=max, step=step, value=value)

    @pytest.mark.parametrize(
        ("min", "max", "step", "error_message"),
        [
            (1, 3, 1, None),
            (1, 1, 1, "step must be zero when min equals max: step 1 cannot step from 1 to 1"),
            (1, 0, 1, "min cannot be greater than max (1 > 0)"),
            (1.0, 3.0, 1.0, None),
            (1.0, 2.0, 3.0, "RangeSetting('invalid' 1.0-2.0, 3.0) min/max difference is not step aligned"),
            (1.0, 0.0, 1.0, "min cannot be greater than max (1.0 > 0.0)"),
        ],
    )
    def test_max_validation(
        self, min: Numeric, max: Numeric, step: Numeric, error_message: str
    ) -> None:
        if error_message is not None:
            with pytest.raises(pydantic.ValidationError) as error:
                RangeSetting(name="invalid", min=min, max=max, step=step, value=1)

            assert error
            assert "1 validation error for RangeSetting" in str(error.value)
            assert error.value.errors()[0]["type"] == "value_error"
            assert error.value.errors()[0]["msg"].startswith(error_message)
        else:
            RangeSetting(name="valid", min=min, max=max, step=step, value=1)

    def test_validation_on_value_mutation(
        self
    ) -> None:
        setting = RangeSetting(name="range", min=0, max=10, step=1)
        with pytest.raises(pydantic.ValidationError) as error:
            setting.value = 25

        assert error
        assert "1 validation error for RangeSetting" in str(error.value)
        assert error.value.errors()[0]["loc"] == ("__root__",)
        assert error.value.errors()[0]["type"] == "value_error"
        assert error.value.errors()[0]["msg"] == "invalid value: 25 is outside of the range 0-10"

    @pytest.mark.parametrize(
        ("min", "max", "step", "error_message"),
        [
            (1, 3, 0, "step cannot be zero"),
            (1.0, 3.0, 0.0, "step cannot be zero"),
        ],
    )
    def test_step_validation(
        self, min: Numeric, max: Numeric, step: Numeric, error_message: str
    ) -> None:
        if error_message is not None:
            with pytest.raises(pydantic.ValidationError) as error:
                RangeSetting(name="invalid", min=min, max=max, step=step, value=1)

            assert error
            assert "1 validation error for RangeSetting" in str(error.value)
            assert error.value.errors()[0]["loc"] == ("__root__",)
            assert error.value.errors()[0]["type"] == "value_error"
            assert error.value.errors()[0]["msg"] == error_message
        else:
            RangeSetting(name="valid", min=min, max=max, step=step, value=1)

    def test_step_cannot_be_zero(self) -> None:
        with pytest.raises(ValueError, match='step cannot be zero') as error:
            RangeSetting(name="range", min=0, max=10, step=0)

    def test_min_can_equal_max(self) -> None:
        RangeSetting(name="range", min=5, max=5, step=0)



class TestCPU:
    @pytest.fixture
    def setting(self) -> CPU:
        return CPU(min=1.0, max=10.0)

    def test_is_range_setting(self, setting: CPU) -> None:
        assert isinstance(setting, RangeSetting)

    def test_default_step(self) -> None:
        assert CPU.__fields__["step"].default == 0.125

    def test_name(self) -> None:
        assert CPU.__fields__["name"].default == "cpu"

    def test_validate_name_cannot_be_changed(self) -> None:
        with pytest.raises(pydantic.ValidationError) as error:
            CPU(name="other", min=1.0, max=10.0)

        assert error
        assert "1 validation error for CPU" in str(error.value)
        assert error.value.errors()[0]["loc"] == ("name",)
        assert error.value.errors()[0]["type"] == "value_error.const"
        assert error.value.errors()[0]["msg"] == "unexpected value; permitted: 'cpu'"

    def test_validate_min_cant_be_zero(self) -> None:
        with pytest.raises(pydantic.ValidationError) as error:
            CPU(min=0.0, max=10.0)

        assert error
        assert "1 validation error for CPU" in str(error.value)
        assert error.value.errors()[0]["loc"] == ("min",)
        assert error.value.errors()[0]["type"] == "value_error.number.not_gt"
        assert error.value.errors()[0]["msg"] == "ensure this value is greater than 0"


class TestMemory:
    def test_is_range_setting(self) -> None:
        assert issubclass(Memory, RangeSetting)

    def test_validate_name_cannot_be_changed(self) -> None:
        with pytest.raises(pydantic.ValidationError) as error:
            Memory(name="other", min=1.0, max=10.0, step=1.0)

        assert error
        assert "1 validation error for Memory" in str(error.value)
        assert error.value.errors()[0]["loc"] == ("name",)
        assert error.value.errors()[0]["type"] == "value_error.const"
        assert error.value.errors()[0]["msg"] == "unexpected value; permitted: 'mem'"

    def test_validate_min_cant_be_zero(self) -> None:
        with pytest.raises(pydantic.ValidationError) as error:
            Memory(min=0.0, max=10.0, step=1.0)

        assert error
        assert "1 validation error for Memory" in str(error.value)
        assert error.value.errors()[0]["loc"] == ("min",)
        assert error.value.errors()[0]["type"] == "value_error"
        assert error.value.errors()[0]["msg"] == "min must be greater than zero"


class TestReplicas:
    def test_is_range_setting(self) -> None:
        assert issubclass(Replicas, RangeSetting)

    @pytest.mark.parametrize(
        ("field_name", "required", "allow_none"),
        [
            ("min", True, False),
            ("max", True, False),
            ("step", False, False),
            ("value", False, True),
        ],
    )
    def test_range_fields_strict_integers(
        self, field_name: str, required: bool, allow_none: bool
    ) -> None:
        field = Replicas.__fields__[field_name]
        assert field.type_ == pydantic.StrictInt
        assert field.required == required
        assert field.allow_none == allow_none

    def test_validate_name_cannot_be_changed(self) -> None:
        with pytest.raises(pydantic.ValidationError) as error:
            Replicas(name="other", min=1, max=10)

        assert error
        assert "1 validation error for Replicas" in str(error.value)
        assert error.value.errors()[0]["loc"] == ("name",)
        assert error.value.errors()[0]["type"] == "value_error.const"
        assert (
            error.value.errors()[0]["msg"] == "unexpected value; permitted: 'replicas'"
        )

class TestInstanceType:
    def test_is_enum_setting(self) -> None:
        assert issubclass(InstanceType, EnumSetting)

    def test_validate_name_cannot_be_changed(self) -> None:
        with pytest.raises(pydantic.ValidationError) as error:
            InstanceType(
                name="other", values=["this", "that"], unit=InstanceTypeUnits.ec2
            )

        assert error
        assert "1 validation error for InstanceType" in str(error.value)
        assert error.value.errors()[0]["loc"] == ("name",)
        assert error.value.errors()[0]["type"] == "value_error.const"
        assert (
            error.value.errors()[0]["msg"] == "unexpected value; permitted: 'inst_type'"
        )

    def test_validate_unit(self) -> None:
        field = InstanceType.__fields__["unit"]
        assert field.type_ == InstanceTypeUnits
        assert field.default == InstanceTypeUnits.ec2
        assert field.required == False
        assert field.allow_none == False

@pytest.mark.parametrize(
    "value, step, aligned",
    [
        (1.0, 0.1, True),
        (1.0, 0.2, True),
        (1.0, 0.125, True),
        (1.0, 0.3, False),
        (0.6, 0.5, False),
        (10, 1, True),
        (3, 12, False),
        (12, 3, True),
        (6, 5, False),
        (5, 6, False),
        (0, 0, True),
        (1, 1, True),
        (0.1, 0.1, True),
    ]
)
def test_step_alignment(value, step, aligned) -> None:
    qualifier = "to" if aligned else "not to"
    assert _is_step_aligned(value, step) == aligned, f"Expected value {value} {qualifier} be aligned with step {step}"

@pytest.mark.parametrize(
    "input, expected_type",
    [
        ("int", int),
        ("float", float),
    ]
)
def test_numeric_type(input, expected_type):
    assert NumericType.validate(input) == expected_type

class TestEnvironmentSettings:
    @pytest.mark.parametrize(
        "expected_value, min, max, step, value, value_type",
        [
            (3.0, 0, 5, 1, "3", None),
            (3,   0, 5, 1, "3", "int"),
            (3.0, 0, 5, 1, "3", "float"),
        ]
    )
    def test_environment_range_setting(self, expected_value, min, max, step, value, value_type):
        test_value = EnvironmentRangeSetting(
            name="test", min=min, max=max, step=step, value=value, value_type=value_type
        ).value
        assert test_value == expected_value and type(test_value) == type(expected_value)

    @pytest.mark.parametrize(
        "name, literal, expected_name",
        [
            ("TEST1", None, "TEST1"),
            ("TEST2", "LITERAL", "LITERAL"),
        ]
    )
    def test_environment_enum_setting(self, name, literal, expected_name):
        assert EnvironmentEnumSetting(
            name=name, literal=literal, value="TEST", values=["TEST", "TSET"]
        ).variable_name == expected_name
