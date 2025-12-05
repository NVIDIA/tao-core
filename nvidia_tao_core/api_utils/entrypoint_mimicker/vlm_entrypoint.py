# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Original source taken from https://github.com/NVIDIA/NeMo
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

"""VLM entrypoint."""


import os
import re
import sys
import shlex
import shutil
import subprocess
from contextlib import contextmanager
from time import time
import logging

from nvidia_tao_core.telemetry.nvml import get_device_details
from nvidia_tao_core.telemetry.telemetry import send_telemetry_data

# Configure logging
TAO_LOG_LEVEL = os.getenv('TAO_LOG_LEVEL', 'INFO').upper()
tao_log_level = getattr(logging, TAO_LOG_LEVEL, logging.INFO)
logging.basicConfig(
    level=logging.WARNING,  # Root logger: suppress third-party DEBUG logs
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logging.getLogger('nvidia_tao_core').setLevel(tao_log_level)
logger = logging.getLogger(__name__)


def convert_dict_to_cli_args(data, parent_key=""):
    """Convert a dictionary to CLI arguments.

    Args:
    - data (dict): The dictionary to convert.
    - parent_key (str, optional): The parent key for nested dictionaries.

    Returns:
    - list: A list of CLI arguments.
    """
    cli_args = []
    for key, value in data.items():
        # Construct the current key path
        if isinstance(value, dict):
            # Recursively process nested dictionaries
            cli_args.extend(convert_dict_to_cli_args(value, key))
        else:
            # Append the CLI argument as --key value
            if str(value):
                cli_args.append(f"--{key}")
                # Handle multi-word strings by adding quotes if needed
                if isinstance(value, str) and (" " in value or "\t" in value):
                    cli_args.append(f'"{value}"')
                else:
                    cli_args.append(str(value))

    return cli_args


def handle_custom_script(specs, custom_script_key, target_script_path="scripts/custom_sft.py"):
    """Handle custom training script provided by user.

    Args:
    - specs (dict): The specifications dictionary that may contain custom_training_script.
    - custom_script_key (str): The key of the custom script in the specifications dictionary.
    - target_script_path (str): The target path where the custom script should be copied.

    Returns:
    - None. Modifies specs in place by removing the custom_training_script key if present.
    """
    if custom_script_key in specs:
        user_script_path = specs.pop(custom_script_key)

        if not user_script_path:
            logger.warning(f"{custom_script_key} is empty, skipping custom script copy")
            return

        if not os.path.exists(user_script_path):
            logger.error(f"Custom training script not found at: {user_script_path}")
            raise FileNotFoundError(f"Custom training script not found: {user_script_path}")

        # Create the target directory if it doesn't exist
        target_dir = os.path.dirname(target_script_path)
        if target_dir and not os.path.exists(target_dir):
            os.makedirs(target_dir, exist_ok=True)
            logger.info(f"Created target directory: {target_dir}")

        # Copy the user-provided script to overwrite the container's script
        try:
            shutil.copy2(user_script_path, target_script_path)
            logger.info(f"Successfully copied custom training script from {user_script_path} to {target_script_path}")
        except Exception as e:
            logger.error(f"Failed to copy custom training script: {e}")
            raise


@contextmanager
def dual_output(log_file=None):
    """Context manager to handle dual output redirection for subprocess.

    Args:
    - log_file (str, optional): Path to the log file. If provided, output will be
      redirected to both sys.stdout and the specified log file. If not provided,
      output will only go to sys.stdout.

    Yields:
    - stdout_target (file object): Target for stdout output (sys.stdout or log file).
    - log_target (file object or None): Target for log file output, or None if log_file
      is not provided.
    """
    if log_file:
        with open(log_file, "a", encoding="utf-8") as f:
            yield sys.stdout, f
    else:
        yield sys.stdout, None


def vlm_launch(neural_network_name, action, specs, job_id=""):
    """Launch a VLM model.

    Args:
    - neural_network_name (str): The name of the neural network.
    - action (str): The action to perform.
    - specs (dict): The specifications for the action.
    """
    command = []
    if 'lepton_specs' in specs:
        lepton_specs = specs.pop('lepton_specs')
        lepton_args = ['--lepton-mode']
        lepton_args += convert_dict_to_cli_args(lepton_specs)
        lepton_args = " ".join(lepton_args)
    else:
        lepton_args = ""
    if neural_network_name == "cosmos-rl" and action in ["train", "evaluate"]:
        # Handle custom training script if provided by user
        if action == "train":
            handle_custom_script(specs, "custom_script", target_script_path="scripts/custom_sft.py")

        train_args = ""
        if action == "train":
            train_args = f"{lepton_args} --port 8080 --rdzv-port 29345 scripts/custom_sft.py"
        launch_cmd = (
            f"{neural_network_name}-{action} --config /results/{job_id}/spec.toml {train_args}"
        )
        command = ["/bin/bash", "-c", launch_cmd]
    else:
        cli_args = convert_dict_to_cli_args(specs)
        cli_args = " ".join(cli_args)
        call = f"{neural_network_name}-{action} {cli_args}"
        command = shlex.split(call)
    process_passed = False
    try:
        # Run the script.
        log_file = ""
        if os.getenv("JOB_ID"):
            logs_dir = os.getenv('TAO_MICROSERVICES_TTY_LOG', '/results')
            log_file = f"{logs_dir}/{os.getenv('JOB_ID')}/microservices_log.txt"

        progress_bar_pattern = re.compile(r"Epoch \d+: \s*\d+%|\[.*\]")
        start = time()
        logger.info(f"command: {command}")
        with dual_output(log_file) as (stdout_target, log_target):
            proc = subprocess.Popen(  # pylint: disable=R1732
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=1,  # Line-buffered
                universal_newlines=True  # Text mode
            )
            last_progress_bar_line = None

            for line in proc.stdout:
                # Check if the line contains \r or matches the progress bar pattern
                if '\r' in line or progress_bar_pattern.search(line):
                    last_progress_bar_line = line.strip()
                    # Print the progress bar line to the terminal
                    stdout_target.write('\r' + last_progress_bar_line)
                    stdout_target.flush()
                else:
                    # Write the final progress bar line to the log file before a new log line
                    if last_progress_bar_line:
                        if log_target:
                            log_target.write(last_progress_bar_line + '\n')
                            log_target.flush()
                        last_progress_bar_line = None
                    stdout_target.write(line)
                    stdout_target.flush()
                    if log_target:
                        log_target.write(line)
                        log_target.flush()

            proc.wait()  # Wait for the process to complete
            # Write the final progress bar line after process completion
            if last_progress_bar_line and log_target:
                log_target.write(last_progress_bar_line + '\n')
                log_target.flush()
            if proc.returncode == 0:
                process_passed = True

    except (KeyboardInterrupt, SystemExit):
        logger.warning("Command was interrupted")
        process_passed = True
    except subprocess.CalledProcessError as e:
        if e.output is not None:
            logger.error(e.output)
        process_passed = False

    end = time()
    time_lapsed = int(end - start)

    try:
        gpu_data = []
        for device in get_device_details():
            gpu_data.append(device.get_config())
        logging.info("Sending telemetry data.")
        send_telemetry_data(
            neural_network_name,
            action,
            gpu_data,
            time_lapsed=time_lapsed,
            pass_status=process_passed
        )
    except Exception as e:
        logging.warning("Telemetry data couldn't be sent, but the command ran successfully.")
        logging.warning(f"[Error]: {e}")

    if not process_passed:
        logger.error("Execution status: FAIL")
        return False

    logger.info("Execution status: PASS")
    return True
