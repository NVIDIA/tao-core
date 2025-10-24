#!/usr/bin/env python3

# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.

# CONFIDENTIAL! DO NOT SPREAD!
# DO NOT CHANGE THIS FILE!

"""Authentication utils metrics modules"""
import json
import requests

__c29tZSByYW5kb20gc3RyaW5n = b'TEVUIE1FIElO'


def report(data={}, base_url='https://api.tao.ngc.nvidia.com', timeout=10):
    """report metrics"""
    url = f'{base_url}/api/v1/metrics'
    if isinstance(data, dict):
        data = json.dumps(data)
    resp = requests.post(url, data=data, auth=('$metricstoken', __c29tZSByYW5kb20gc3RyaW5n), timeout=timeout)
    if resp.status_code == 201:
        return None
    return f'error {resp.status_code}'
