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

"""Functional test conftest."""
import os
import pytest


@pytest.fixture
def ngc_key():
    """Configure NGC key. Skip test if not set."""
    key = os.environ.get("NGC_KEY")
    if not key:
        pytest.skip("NGC_KEY environment variable not set")
    return key


@pytest.fixture
def ngc_path():
    """Configure NGC path. Skip test if not set."""
    path = os.environ.get("NGC_PATH")
    if not path:
        pytest.skip("NGC_PATH environment variable not set")
    return path
