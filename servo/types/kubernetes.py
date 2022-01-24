import enum


class ContainerLogOptions(str, enum.Enum):
    previous = "previous"
    current = "current"
    both = "both"
