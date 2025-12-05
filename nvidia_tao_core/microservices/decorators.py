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

"""Flask decorators for common functionality."""

from functools import wraps
from flask import make_response, jsonify

from nvidia_tao_core.microservices.utils.core_utils import is_pvc_space_free


def disk_space_check(f):
    """Decorator to check disk space for API endpoints"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        threshold_bytes = 100 * 1024 * 1024

        pvc_free_space, pvc_free_bytes = is_pvc_space_free(threshold_bytes)
        msg = (f"PVC free space remaining is {pvc_free_bytes} bytes "
               f"which is less than {threshold_bytes} bytes")
        if not pvc_free_space:
            return make_response(
                jsonify({
                    'error': f'Disk space is nearly full. {msg}. Delete appropriate experiments/datasets'
                }),
                500
            )

        return f(*args, **kwargs)

    return decorated_function
