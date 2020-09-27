import pytest
from servo.connectors import formation

# TODO: CLI test with check and run --check

class TestFormationConfiguration:
    def test_generate_kubernetes_config(self) -> None:
        name, config = formation.FormationConfiguration.generate_kubernetes_config()
        debug(name, config)
    
    def test_generate_prometheus_config(self) -> None:
        name, config = formation.FormationConfiguration.generate_prometheus_config()
        debug(name, config)

        # TODO: Do we use a generator and yield them? Return a collection? Return a complete configuration obj?
    def test_generate(self) -> None:
        for name, config in formation.FormationConfiguration.generate():
            debug(name, config)
            
class TestFormationChecks:
    ...

class TestFormationConnector:
    ...

class TestFormationCLI:
    ...
