#!/usr/bin/env python3

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

"""Authentication utils metrics modules"""
import json
import requests

__c29tZSByYW5kb20gc3RyaW5n = b'TEVUIE1FIElO'


def validate(encoded_key=b''):
    """validate key or encoded_key"""
    if isinstance(encoded_key, str):
        encoded_key = bytes(encoded_key, 'utf-8')
    if encoded_key == __c29tZSByYW5kb20gc3RyaW5n:
        return True
    return False


def report(data={}, base_url='https://api.tao.ngc.nvidia.com', timeout=10):
    """report metrics"""
    url = f'{base_url}/api/v1/metrics'
    if isinstance(data, dict):
        data = json.dumps(data)
    resp = requests.post(url, data=data, auth=('$metricstoken', __c29tZSByYW5kb20gc3RyaW5n), timeout=timeout)
    if resp.status_code == 201:
        return None
    return f'error {resp.status_code}'
