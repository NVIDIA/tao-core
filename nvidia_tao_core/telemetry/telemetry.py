# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

"""Utilties to send data to the TAO Toolkit Telemetry Remote Service."""

import os
from typing import Any, Dict, List, Optional

import logging as _logging
_logging.basicConfig(
    format='[%(asctime)s - TAO Toolkit - %(name)s - %(levelname)s] %(message)s',
    level='INFO'
)
logging = _logging

try:
    from nvidia_tao_core.telemetry import metrics
    METRICS_MODULE_EXISTS = True
except Exception as e:
    logging.warning(f"Telemetry reporting script cannot import with [Error]: {e}")
    METRICS_MODULE_EXISTS = False

TAO_SERVER_URL = "https://api.tao.ngc.nvidia.com"
TELEMETRY_TIMEOUT = int(os.getenv("TELEMETRY_TIMEOUT", "30"))


def send_telemetry_data(
    network: str,
    action: str,
    gpu_data: List[Dict[str, Any]],
    num_gpus: int = 1,
    time_lapsed: Optional[int] = None,
    pass_status: bool = False,
    user_error: bool = False
) -> None:
    """Wrapper to send TAO telemetry data.

    Args:
        network (str): Name of the network being run.
        action (str): Subtask of the network called.
        gpu_data (list of dict): List of dictionaries containing data about the GPU's in the machine.
        num_gpus (int): Number of GPUs used in the job.
        time_lapsed (int): Time lapsed.
        pass_status (bool): Job passed or failed.
        user_error (bool): Whether the error is a user error.

    Environment variables:
        TELEMETRY_OPT_OUT (str): Whether to opt out of telemetry reporting, default: no.
        TAO_TELEMETRY_SERVER (str): Telemetry reporting url, default: https://api.tao.ngc.nvidia.com.
        TAO_TOOLKIT_VERSION (str): Verson of TAO Toolkit used, default: 5.3.0.
        TELEMETRY_TIMEOUT (int): Telemetry reporting request timeout limit, default: 30.

    Returns:
        No explicit returns.
    """
    logging.info("================> Start Reporting Telemetry <================")

    if os.getenv('TELEMETRY_OPT_OUT', "no").lower() in ["no", "false", "0"]:
        url = os.getenv("TAO_TELEMETRY_SERVER", TAO_SERVER_URL)
        data = {
            "version": os.getenv("TAO_TOOLKIT_VERSION", "5.3.0"),
            "action": action,
            "network": network,
            "gpu": [device["name"] for device in gpu_data[:num_gpus]],
            "success": pass_status,
            "user_error": user_error
        }
        if time_lapsed is not None:
            data["time_lapsed"] = time_lapsed
        if METRICS_MODULE_EXISTS:
            logging.info(f"Sending {data} to {url}.")
            response = metrics.report(data=data, base_url=url, timeout=TELEMETRY_TIMEOUT)
            if response:
                logging.info(f"Failed with reponse: {response}")
            else:
                logging.info("Telemetry sent successfully.")
    else:
        logging.info("Opted out of telemetry reporting. Skipped.")

    logging.info("================> End Reporting Telemetry <================")
