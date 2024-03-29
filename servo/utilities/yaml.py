# Copyright 2022 Cisco Systems, Inc. and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import yaml as _yaml


class PreservedScalarString(str):
    """
    PreservedScalarString is a utility class that will
    serialize into a multi-line YAML string in the '|' style
    """


def pss_representer(dumper, scalar_string: PreservedScalarString):
    return dumper.represent_scalar("tag:yaml.org,2002:str", scalar_string, style="|")


_yaml.add_representer(PreservedScalarString, pss_representer)
