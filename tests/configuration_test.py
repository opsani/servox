import pathlib
import servo
import tests

def test_config(servo_yaml: pathlib.Path) -> None:
    config_model = servo.assembly._create_config_model_from_routes(
        {
            "adjust": tests.helpers.AdjustConnector,
        }
    )
    config = config_model.generate()
    servo_yaml.write_text(config.yaml())

    optimizer = servo.configuration.Optimizer(
        id="dev.opsani.com/blake-ignite",
        token="bfcf94a6e302222eed3c73a5594badcfd53fef4b6d6a703ed32604",
    )
    assembly_ = servo.assembly.Assembly.assemble(
        config_file=servo_yaml, optimizer=optimizer
    )
