import yaml as _yaml


class PreservedScalarString(str):
    """
    PreservedScalarString is a utility class that will
    serialize into a multi-line YAML string in the '|' style
    """


def pss_representer(dumper, scalar_string: PreservedScalarString):
    return dumper.represent_scalar(u"tag:yaml.org,2002:str", scalar_string, style="|")


_yaml.add_representer(PreservedScalarString, pss_representer)
