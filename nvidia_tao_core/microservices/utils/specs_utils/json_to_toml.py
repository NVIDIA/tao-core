# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Json to toml file conversion"""
import toml
from nvidia_tao_core.microservices.utils.core_utils import safe_load_file


def toml_format(data):
    """Converts the dictionary data into toml format string"""
    if type(data) is dict:
        data = data.copy()  # Don't modify original
        data.pop("version", None)
    return toml.dumps(data)


def convert(path):
    """Reads from json and dumps into toml format"""
    data = safe_load_file(path)
    return toml_format(data)
