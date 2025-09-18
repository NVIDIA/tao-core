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

"""Helpers for managing medical image/labels/metrics"""

import json
import os
import re
import logging
import tarfile
import zipfile
import ruamel.yaml
from urllib.parse import urlparse

import requests
from filelock import FileLock
from nvidia_tao_core.microservices.handlers.stateless_handlers import get_handler_root, get_root
from nvidia_tao_core.microservices.utils import get_default_lock_file_path

CUSTOMIZED_BUNDLE_URL_FILE = "url.json"
CUSTOMIZED_BUNDLE_URL_KEY = "bundle_url"
MEDICAL_SERVICE_SCRIPTS = "/medical_service_scripts"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DynamicSorter:
    """Sort the metrics and the highest should stay in the first element (zero)."""

    def __init__(self):
        """Initialize the sorter"""
        self.pairs = []

    def add_pair(self, name, value):
        """Add a name-value pair to the sorter"""
        self.pairs.append((name, value))
        self.pairs.sort(key=lambda pair: pair[1], reverse=True)

    def get_pairs(self):
        """Get the sorted name-value pairs"""
        return self.pairs


class ImageLabelRecord:
    """Record of image-label pair for a dataset

    Args:
        record_file: path to the file to load the records.
    """

    def __init__(self, record_file):
        """Initialize the record

        Args:
            record_file: path to the file to load the notify records.
        """
        if not os.path.isfile(record_file):
            self.records = {
                "added": [],
                "updated": [],
                "removed": []
            }
        else:
            with open(record_file, "r", encoding="utf-8") as f:
                self.records = json.load(f)

    def _add_record(self, image_id, label_id):
        """Add a record to the list of added records"""
        entry = {
            "image": image_id,
            "label": label_id
        }
        self.records["added"].append(entry)

    def _update_record(self, image_id, label_id):
        """Add a record to the list of updated records"""
        entry = {
            "image": image_id,
            "label": label_id
        }
        # TODO: remove from added, if exists
        # self.records["added"] = [record for record in self.records["added"] if record["image"] != image_id]
        # Add to updated
        self.records["updated"].append(entry)

    def _remove_record(self, image_id, label_id):
        """Add a record to the list of removed records"""
        entry = {
            "image": image_id,
            "label": label_id
        }
        # TODO: remove from added or updated, if exists
        # self.records["added"] = [record for record in self.records["added"] if record["image"] != image_id]
        # self.records["updated"] = [record for record in self.records["updated"] if record["image"] != image_id]
        # Add to removed
        self.records["removed"].append(entry)

    def _validate_data(self, data):
        """Validate the data to be processed"""
        if not isinstance(data, dict):
            return False, f"{data} is not a dict"

        for section in data:
            if section not in ["added", "updated", "removed"]:
                return False, f"{section} is not a valid section"

        for section in ["added", "updated", "removed"]:
            # if data[section] is dict, it will be converted to a list
            status, msg = self._validate_section(data, section)
            if not status:
                return status, msg

        return True, "Validation successful"

    def _validate_section(self, data, section):
        """Validate a section of the data to be processed"""
        items = data.get(section, [])

        # Convert dict to list of dict for consistency
        if isinstance(items, dict):
            items = [items]
            data[section] = items

        for entry in items:
            for req_key in ["image", "label"]:
                if not entry.get(req_key):
                    return False, f"{entry} does not contain required key {req_key}"

        return True, f"{section} validation successful"

    def process_data(self, data):
        """Process the data to be added/updated/removed"""
        status, msg = self._validate_data(data)
        if not status:
            return status, msg

        # Handle additions
        for entry in data.get("added", []):
            self._add_record(entry["image"], entry["label"])

        # Handle updates
        for update_entry in data.get("updated", []):
            self._update_record(update_entry["image"], update_entry["label"])

        # Handle removals
        for remove_entry in data.get("removed", []):
            self._remove_record(remove_entry["image"], remove_entry["label"])

        return True, "Data processed successfully"

    def export(self, record_file):
        """Export the records to a file

        Args:
            record_file: path to the file to export the records to.
        """
        with open(record_file, "w", encoding="utf-8") as f:
            json.dump(self.records, f, indent=4)

    @staticmethod
    def select_unique_images(records, sections):
        """Select how many unique image items in the sections of the records.

        Args:
            records: a dict the notify records that contains (added, updated, removed) sections
            sections: a list of strings of the section to select from (added, updated, removed)
        """
        seen = set()
        unique_items_list = []

        if isinstance(sections, str):
            sections = [sections]

        for section in sections:
            if section not in ("added", "updated", "removed"):
                raise ValueError(f"{section} it not a valid section")
            items = records.get(section, [])
            for item in items:
                if item["image"] not in seen:
                    seen.add(item["image"])
                    unique_items_list.append(item)
        return unique_items_list

    @staticmethod
    def count_added_labels(current_records, latest_records):
        """Count how many labels are updated currently compared to the latest record."""
        current_added_unique = ImageLabelRecord.select_unique_images(current_records, ["added", "updated"])
        latest_added_unique = ImageLabelRecord.select_unique_images(latest_records, ["added", "updated"])

        added_count = len(current_added_unique) - len(latest_added_unique)

        return added_count


class CapGpuUsage:
    """Cap the maximum GPU usage for a user for real-time inference"""

    MAX_GPU_PER_USER_REALTIME_INFER = int(os.environ.get("MAX_GPU_PER_USER_REALTIME_INFER", 2))

    @staticmethod
    def schedule(org_name, num_gpus):
        """Schedule the GPU usage for a user.

        Args:
            org_name: name of the user/org
            num_gpus: the number of GPUs to be used
        """
        if CapGpuUsage.MAX_GPU_PER_USER_REALTIME_INFER <= 0:
            return True, ""

        user_config_file = get_root() + f"{org_name}/user_config.json"

        lock_file = get_default_lock_file_path(user_config_file)
        if not os.path.exists(lock_file):
            with open(lock_file, "w", encoding="utf-8"):
                pass

        with FileLock(lock_file):
            if os.path.exists(user_config_file):
                with open(user_config_file, "r", encoding="utf-8") as f:
                    user_config = json.load(f)
            else:
                user_config = {
                    "max_gpu_realtime_infer": CapGpuUsage.MAX_GPU_PER_USER_REALTIME_INFER,
                    "current_used": 0
                }
                with open(user_config_file, "w", encoding="utf-8") as fp:
                    json.dump(user_config, fp, indent=4)

            logger.info("Organization User %s config (pre-schedule) is %s", org_name, user_config)
            if user_config["current_used"] + num_gpus > user_config["max_gpu_realtime_infer"]:
                used = user_config["current_used"]
                max_allowed = user_config["max_gpu_realtime_infer"]
                msg = (
                    f"Organization User request {num_gpus} GPU(s), But with {num_gpus} + {used} "
                    f"(GPUs in used) will exceed the maximum number of GPUs allowed ({max_allowed}). "
                    "Please consider list th"
                )
                logger.warning(msg)
                return False, msg

            user_config["current_used"] += num_gpus
            with open(user_config_file, "w", encoding="utf-8") as fp:
                json.dump(user_config, fp, indent=4)
            logger.info("Organization User %s config (post-schedule) is %s", org_name, user_config)
            return True, ""

    @staticmethod
    def release_used(org_name, num_gpus):
        """Release the GPU usage for a user on record.

        Args:
            org_name: the name of the user/org
            num_gpus: the number of GPUs to be released
        """
        if CapGpuUsage.MAX_GPU_PER_USER_REALTIME_INFER <= 0:
            return True, ""

        user_config_file = get_root() + f"{org_name}/user_config.json"
        if not os.path.exists(user_config_file):
            logger.warning(
                "Organization User config file does not exist when release_used is call for %s", org_name
            )
            return False, "Internal Error"

        lock_file = get_default_lock_file_path(user_config_file)
        if not os.path.exists(lock_file):
            logger.warning("Lock file does not exist when release_used is call for %s", org_name)
            return False, "Internal Error"

        with FileLock(lock_file):
            with open(user_config_file, "r", encoding="utf-8") as f:
                user_config = json.load(f)
            logger.info("Organization User %s config (pre-released) is %s", org_name, user_config)
            user_config["current_used"] -= num_gpus
            if user_config["current_used"] < 0:
                logger.warning("Organization User %s current_used GPU is less than 0. Resetting to 0", org_name)
                user_config["current_used"] = 0
            with open(user_config_file, "w", encoding="utf-8") as fp:
                json.dump(user_config, fp, indent=4)
            logger.info("Organization User %s config (post-released) is %s", org_name, user_config)
            return True, ""


def download_from_url(url, handler_ptm):
    """Download a archived file as a ptm from a URL to the Persistent Storage (Non-NGC Workspace)"""
    # Send a GET request to the URL
    try:
        response = requests.get(url, timeout=120)
    except Exception as e:
        logger.error("Exception caught during download of MONAI file: %s", str(e))
        raise e
    ptm_root = get_handler_root(handler_id=handler_ptm)
    parsed_url = urlparse(url)
    filename = os.path.basename(parsed_url.path)

    # Join the folder path and filename
    ptm_file = os.path.join(ptm_root, filename)
    # Check if the request was successful
    if response.status_code == 200:
        # Write the content of the response to a file
        with open(ptm_file, 'wb') as file:
            file.write(response.content)
        return ptm_file
    return None


def find_config_file(config_name, path, ext):
    """Find the config file with the given name and extension in the given path."""
    for e in ext:
        name = f"{config_name}{e}"
        if os.path.exists(os.path.join(path, name)):
            return name
    return None


def check_necessary_arg(config_file, necessary_args_list):
    """Check if the necessary arguments are present in the given config file."""
    with open(config_file, "r", encoding="utf-8") as f:
        if config_file.endswith(("yaml", "yml")):
            yaml = ruamel.yaml.YAML(typ='safe')
            infer_config_content = yaml.load(f)
        else:
            infer_config_content = json.load(f)
        for arg in necessary_args_list:
            if arg not in infer_config_content:
                return False
            if arg == "dataset" and "data" not in infer_config_content[arg]:
                return False
    return True


def has_correct_monai_bundle_structure(folder, checks=[]):
    """Function to check if the folder is a valid MONAI bundle."""
    is_bundle_dir = os.path.exists(os.path.join(folder, "configs", "metadata.json"))
    if is_bundle_dir:
        if "infer" in checks:
            # 1. necessary files checks
            infer_config = find_config_file("inference", os.path.join(folder, "configs"), [".json", ".yaml", ".yml"])
            if infer_config is None:
                return False
            other_necessary_files_list = ["configs/logging.conf"]
            for file in other_necessary_files_list:
                if not os.path.exists(os.path.join(folder, file)):
                    return False
            # 2. required args checks
            # we must call self.workflow.evaluator.run(), and input data via override "dataset#data"
            necessary_args_list = ["evaluator", "dataset"]
            if not check_necessary_arg(os.path.join(folder, "configs", infer_config), necessary_args_list):
                return False
        return True
    return False


def check_and_extract_all(filepath, output_dir):
    """Extract file to the output directory. Expected file types are: `zip`, `tar.gz` and `tar`."""
    if not os.path.isdir(output_dir):
        return False
    if filepath.endswith("zip"):
        with zipfile.ZipFile(filepath, 'r') as zip_file:
            zip_file.extractall(output_dir)
        # remove the zip file to save space
        os.remove(filepath)
        return True
    if filepath.endswith("tar") or filepath.endswith("tar.gz"):
        with tarfile.open(filepath, 'r') as tar_file:
            tar_file.extractall(output_dir)
        # remove the zip file
        os.remove(filepath)
        return True
    return False


def remove_extension(filename):
    """Remove the extension from the filename"""
    for ext in ["zip", "tar", "tar.gz"]:
        if filename.endswith(ext):
            return filename.replace("." + ext, "")
    raise ValueError(f"Unknown file extension for {filename}")


def list_folders(directory):
    """Returns a set of all folder paths in the given directory."""
    paths = set()
    for root, dirs, _ in os.walk(directory):
        for name in dirs:
            paths.add(os.path.join(root, name))
    return paths


def validate_monai_bundle(handler_id, checks=[]):
    """Function to validate if the source folder is a valid MONAI bundle"""
    # Should contain one folder in the source folder
    source_folder = get_handler_root(handler_id=handler_id)
    for root, dirs, files in os.walk(source_folder):
        for dir_name in dirs:
            # Downloaded from NGC
            subfolder_path = os.path.join(root, dir_name)
            if has_correct_monai_bundle_structure(subfolder_path, checks=checks):
                return subfolder_path
        for file in files:
            # Downloaded as zip file
            subfolder_filepath = os.path.join(root, file)
            existing_folders = list_folders(root)
            if check_and_extract_all(subfolder_filepath, root):
                new_folders = list_folders(root)
                extracted_folders = new_folders - existing_folders
                for extracted_folder in list(extracted_folders):
                    if has_correct_monai_bundle_structure(extracted_folder, checks=checks):
                        return extracted_folder
    return None


def find_matching_bundle_dir(directory, patterns):
    """Find the first item in a directory that matches one of the patterns."""
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if not os.path.isdir(item_path):
            continue

        for pattern in patterns:
            match = re.match(pattern, item)
            if match:
                return item

        if has_correct_monai_bundle_structure(item_path):
            return item
    return None
