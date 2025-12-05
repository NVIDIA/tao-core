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

"""Health check endpoints

- Liveliness
- Readiness
"""
import os
import tempfile
from kubernetes import client, config
import logging

# Configure logging
TAO_LOG_LEVEL = os.getenv('TAO_LOG_LEVEL', 'INFO').upper()
tao_log_level = getattr(logging, TAO_LOG_LEVEL, logging.INFO)
logging.basicConfig(
    level=logging.WARNING,  # Root logger: suppress third-party DEBUG logs
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logging.getLogger('nvidia_tao_core').setLevel(tao_log_level)
logger = logging.getLogger(__name__)


def check_logging():
    """Checks if we are able to create and write into files"""
    try:
        file, path = tempfile.mkstemp()
        with os.fdopen(file, 'w') as tmp:
            tmp.write('Logging online!')
        os.remove(path)
        return True
    except Exception as e:
        logger.error("Exception thrown in check_logging is %s", str(e))
        return False


def check_k8s():
    """Checks if we are able to initialize kubernetes client"""
    try:
        if os.getenv("DEV_MODE", "False").lower() in ("true", "1"):
            os.getenv('NAMESPACE', default="default")
            config.load_kube_config()
        else:
            with open('/var/run/secrets/kubernetes.io/serviceaccount/namespace', 'r', encoding='utf-8') as f:
                current_name_space = f.read()
            os.getenv('NAMESPACE', default=current_name_space)
            config.load_incluster_config()
        client.BatchV1Api()
        return True
    except Exception as e:
        logger.error("Exception thrown in check_k8s is %s", str(e))
        return False
