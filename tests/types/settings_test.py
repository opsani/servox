import abc
import loguru
import pytest
import re

from servo.types.settings import *
from servo.types.settings import _is_step_aligned


class BasicSetting(Setting):
    name: str = "foo"
    type: str = "bar"

    def __opsani_repr__(self) -> dict:
        return {}


class TestSetting:
    def test_is_abstract_base_class(self) -> None:
        assert issubclass(Setting, abc.ABC)

    def test_requires_opsani_repr_implementation(self) -> None:
        assert "__opsani_repr__" in Setting.__abstractmethods__

    def test_validates_default(self) -> None:
        assert Setting.model_config["validate_default"]

    def test_validates_assignment(self) -> None:
        assert Setting.model_config["validate_assignment"]

    def test_human_readable_value(self) -> None:
        setting = BasicSetting(value="whatever")
        assert setting.human_readable_value == "whatever"

        class HumanReadableTestValue(str):
            def human_readable(self) -> str:
                return "another-value"

            @classmethod
            def __get_pydantic_core_schema__(
                cls, _: Any, handler: pydantic.GetCoreSchemaHandler
            ) -> pydantic_core.CoreSchema:
                return pydantic_core.core_schema.no_info_after_validator_function(
                    cls, handler(str)
                )

        class HumanReadableSetting(BasicSetting):
            value: Optional[HumanReadableTestValue] = None

        setting2 = HumanReadableSetting(value=HumanReadableTestValue("whatever"))
        assert setting2.human_readable_value == "another-value"

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

            with pytest.raises(ValueError) as error:
                setting.value = new_value

            assert error
            assert str(error.value) == error_message
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
        assert error.value.errors()[0]["type"] == "literal_error"
        assert error.value.errors()[0]["msg"] == "Input should be 'enum'"

    def test_validate_values_list_is_not_empty(self) -> None:
        with pytest.raises(pydantic.ValidationError) as error:
            EnumSetting(name="bar", values=[])

        assert error
        assert "1 validation error for EnumSetting" in str(error.value)
        assert error.value.errors()[0]["loc"] == ("values",)
        assert error.value.errors()[0]["type"] == "too_short"
        assert (
            error.value.errors()[0]["msg"]
            == "List should have at least 1 item after validation, not 0"
        )

    def test_validate_value_is_included_in_values_list(self) -> None:
        with pytest.raises(pydantic.ValidationError) as error:
            EnumSetting(name="bar", values=["one", "two"], value="three")

        assert error
        assert "1 validation error for EnumSetting" in str(error.value)
        assert error.value.errors()[0]["loc"] == ()
        assert error.value.errors()[0]["type"] == "value_error"
        assert (
            error.value.errors()[0]["msg"]
            == "Value error, invalid value: 'three' is not in the values list ['one', 'two']"
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
        assert error.value.errors()[0]["type"] == "literal_error"
        assert error.value.errors()[0]["msg"] == "Input should be 'range'"

    def test_validate_step_alignment_suggestion(
        self, captured_logs: list["loguru.Message"]
    ) -> None:
        RangeSetting(name="invalid", min=3.0, max=11.0, step=3.0)
        assert any(
            c.record["message"]
            == "RangeSetting('invalid' 3.0-11.0, 3.0) min/max difference is not step aligned: 8.0 is not a multiple of 3.0 (consider min 5.0 or 2.0, max 9.0 or 12.0)."
            for c in captured_logs
        ), captured_logs

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
        self,
        min: Numeric,
        max: Numeric,
        step: Numeric,
        error_message: str,
        captured_logs: list["loguru.Message"],
    ) -> None:
        if error_message is not None:
            RangeSetting(name="invalid", min=min, max=max, step=step)
            any(c.record["message"] == error_message for c in captured_logs)

        else:
            RangeSetting(name="valid", min=min, max=max, step=step)
            assert len(captured_logs) < 1

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
        self,
        min: Numeric,
        max: Numeric,
        step: Numeric,
        error_message: str,
        captured_logs: list["loguru.Message"],
    ) -> None:
        if error_message is not None:
            RangeSetting(name="invalid", min=min, max=max, step=step)
            any(
                c.record["message"] == error_message for c in captured_logs
            ), captured_logs
        else:
            RangeSetting(name="valid", min=min, max=max, step=step)
            assert len(captured_logs) < 1

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
        captured_logs: list["loguru.Message"],
    ) -> None:
        if error_message is not None:
            RangeSetting(name="invalid", min=min, max=max, step=step, value=value)
            assert any(
                c.record["message"] == error_message for c in captured_logs
            ), captured_logs

        else:
            RangeSetting(name="valid", min=min, max=max, step=step, value=value)
            assert len(captured_logs) < 1

    @pytest.mark.parametrize(
        ("min", "max", "step", "error_message"),
        [
            (1, 3, 1, None),
            (
                1,
                1,
                1,
                "step must be zero when min equals max: step 1 cannot step from 1 to 1 (consider using the pinned attribute of settings if you have a value you don't want changed)",
            ),
            (1, 0, 1, "invalid value: 1 is outside of the range 1-0"),
            (1.0, 3.0, 1.0, None),
            (
                1.0,
                2.0,
                3.0,
                "RangeSetting('invalid' 1.0-2.0, 3.0) min/max difference is not step aligned: 1.0 is not a multiple of 3.0 (consider min -1.0 or -4.0, max 4.0 or 7.0).",
            ),
            (1.0, 0.0, 1.0, "invalid value: 1 is outside of the range 1.0-0.0"),
        ],
    )
    def test_max_validation(
        self,
        min: Numeric,
        max: Numeric,
        step: Numeric,
        error_message: str,
        captured_logs: list["loguru.Message"],
    ) -> None:
        if error_message is not None:
            RangeSetting(name="invalid", min=min, max=max, step=step, value=1)
            assert any(
                c.record["message"] == error_message for c in captured_logs
            ), captured_logs

        else:
            RangeSetting(name="valid", min=min, max=max, step=step, value=1)
            assert len(captured_logs) < 1

    def test_validation_on_value_mutation(
        self, captured_logs: list["loguru.Message"]
    ) -> None:
        setting = RangeSetting(name="range", min=0, max=10, step=1)
        setting.value = 25
        assert any(
            c.record["message"] == "invalid value: 25 is outside of the range 0-10"
            for c in captured_logs
        ), captured_logs

    @pytest.mark.parametrize(
        ("min", "max", "step", "error_message"),
        [
            (1, 3, 0, "step cannot be zero"),
            (1.0, 3.0, 0.0, "step cannot be zero"),
        ],
    )
    def test_step_validation(
        self,
        min: Numeric,
        max: Numeric,
        step: Numeric,
        error_message: str,
        captured_logs: list["loguru.Message"],
    ) -> None:
        if error_message is not None:
            RangeSetting(name="invalid", min=min, max=max, step=step, value=1)
            assert any(
                c.record["message"] == error_message for c in captured_logs
            ), captured_logs

        else:
            RangeSetting(name="valid", min=min, max=max, step=step, value=1)
            assert len(captured_logs) < 1

    def test_step_cannot_be_zero(self, captured_logs: list["loguru.Message"]) -> None:
        RangeSetting(name="range", min=0, max=10, step=0)
        assert any(
            c.record["message"] == "step cannot be zero"
            and c.record["level"].name == "WARNING"
            for c in captured_logs
        ), captured_logs

    def test_min_can_equal_max(self) -> None:
        RangeSetting(name="range", min=5, max=5, step=0)


class TestCPU:
    @pytest.fixture
    def setting(self) -> CPU:
        return CPU(min=1.0, max=10.0)

    def test_is_range_setting(self, setting: CPU) -> None:
        assert isinstance(setting, RangeSetting)

    def test_default_step(self) -> None:
        assert CPU.model_fields["step"].default == 0.125

    def test_name(self) -> None:
        assert CPU.model_fields["name"].default == "cpu"

    def test_validate_name_cannot_be_changed(self) -> None:
        with pytest.raises(pydantic.ValidationError) as error:
            CPU(name="other", min=1.0, max=10.0)

        assert error
        assert "1 validation error for CPU" in str(error.value)
        assert error.value.errors()[0]["loc"] == ("name",)
        assert error.value.errors()[0]["type"] == "literal_error"
        assert error.value.errors()[0]["msg"] == "Input should be 'cpu'"

    def test_validate_min_cant_be_zero(self) -> None:
        with pytest.raises(pydantic.ValidationError) as error:
            CPU(min=0.0, max=10.0)

        assert error
        assert "1 validation error for CPU" in str(error.value)
        assert error.value.errors()[0]["loc"] == ("min",)
        assert error.value.errors()[0]["type"] == "greater_than"
        assert error.value.errors()[0]["msg"] == "Input should be greater than 0"


class TestMemory:
    def test_is_range_setting(self) -> None:
        assert issubclass(Memory, RangeSetting)

    def test_validate_name_cannot_be_changed(self) -> None:
        with pytest.raises(pydantic.ValidationError) as error:
            Memory(name="other", min=1.0, max=10.0, step=1.0)

        assert error
        assert "1 validation error for Memory" in str(error.value)
        assert error.value.errors()[0]["loc"] == ("name",)
        assert error.value.errors()[0]["type"] == "literal_error"
        assert error.value.errors()[0]["msg"] == "Input should be 'mem'"

    def test_validate_min_cant_be_zero(
        self, captured_logs: list["loguru.Message"]
    ) -> None:
        Memory(min=0.0, max=10.0, step=1.0)
        any(
            c.record["message"] == "min must be greater than zero"
            for c in captured_logs
        )


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
        field = Replicas.model_fields[field_name]
        if not required and allow_none:
            assert (
                field.annotation
                == typing.Optional[typing.Annotated[int, pydantic.Strict(strict=True)]]
            )
        else:
            assert field.annotation == int
        assert field.is_required() == required

    def test_validate_name_cannot_be_changed(self) -> None:
        with pytest.raises(pydantic.ValidationError) as error:
            Replicas(name="other", min=1, max=10)

        assert error
        assert "1 validation error for Replicas" in str(error.value)
        assert error.value.errors()[0]["loc"] == ("name",)
        assert error.value.errors()[0]["type"] == "literal_error"
        assert error.value.errors()[0]["msg"] == "Input should be 'replicas'"


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
        assert error.value.errors()[0]["type"] == "literal_error"
        assert error.value.errors()[0]["msg"] == "Input should be 'inst_type'"

    def test_validate_unit(self) -> None:
        field = InstanceType.model_fields["unit"]
        assert field.annotation == InstanceTypeUnits
        assert field.default == InstanceTypeUnits.ec2
        assert field.is_required() == False


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
    ],
)
def test_step_alignment(value, step, aligned) -> None:
    qualifier = "to" if aligned else "not to"
    assert (
        _is_step_aligned(value, step) == aligned
    ), f"Expected value {value} {qualifier} be aligned with step {step}"


class TestEnvironmentSettings:
    @pytest.mark.parametrize(
        "expected_value, min, max, step, value, value_type",
        [
            (3.0, 0, 5, 1, "3", None),
            (3, 0, 5, 1, "3", "int"),
            (3.0, 0, 5, 1, "3", "float"),
        ],
    )
    def test_environment_range_setting(
        self, expected_value, min, max, step, value, value_type
    ):
        test_value = EnvironmentRangeSetting(
            name="test", min=min, max=max, step=step, value=value, value_type=value_type
        ).value
        assert test_value == expected_value and type(test_value) == type(expected_value)

    @pytest.mark.parametrize(
        "name, literal, expected_name",
        [
            ("TEST1", None, "TEST1"),
            ("TEST2", "LITERAL", "LITERAL"),
        ],
    )
    def test_environment_enum_setting(self, name, literal, expected_name):
        assert (
            EnvironmentEnumSetting(
                name=name, literal=literal, value="TEST", values=["TEST", "TSET"]
            ).variable_name
            == expected_name
        )
