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

"""MONAI Dataset Handler module."""
import os
import pathlib
import random
import re
from threading import Thread
import requests
import validators
from filelock import FileLock
from uuid import uuid5, NAMESPACE_URL
import logging

from nvidia_tao_core.microservices.handlers import stateless_handlers
from nvidia_tao_core.microservices.handlers.encrypt import NVVaultEncryption
from nvidia_tao_core.microservices.handlers.monai.dataset.cache import LocalCache
from nvidia_tao_core.microservices.handlers.monai.dataset.dicom import DicomEndpoint
from nvidia_tao_core.microservices.handlers.monai.dataset.object_storage import ObjectStorageEndpoint
from nvidia_tao_core.microservices.handlers.utilities import Code
from nvidia_tao_core.microservices.handlers.stateless_handlers import resolve_existence, resolve_root, resolve_metadata
from nvidia_tao_core.microservices.handlers.monai.helpers import ImageLabelRecord
from nvidia_tao_core.microservices.utils import get_default_lock_file_path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

MONAI_DATASET_ACTIONS = [
    "nextimage",
    "cacheimage",
    "notify",
]

dataset_cache_handles: dict[str, LocalCache] = {}


def get_filename_from_cd(url, cd):
    """Get filename from content-disposition"""
    if not cd:
        if url.find('/'):
            return url.rsplit('/', 1)[1]
        return None

    fname = re.findall('filename=(.+)', cd)
    if len(fname) == 0:
        return None
    return fname[0]


class MonaiDatasetHandler:
    """MONAI Dataset Handler class."""

    @staticmethod
    def action_next_image(org_name, dataset_id, dataset_metadata, action_spec):
        """Get the next image from the dataset"""
        ep = MonaiDatasetHandler.endpoint(dataset_metadata)
        unlabeled = ep.get_unlabeled_images()
        if not unlabeled:
            return Code(404, msg="No Unlabeled Images Found")

        image = random.choices(list(unlabeled.keys()))[0]
        ret_info = {
            "image": image,
            "meta": unlabeled[image]
        }
        ret_info["meta"].update(ep.get_info(image))

        MonaiDatasetHandler.action_cache_image(org_name, dataset_id, dataset_metadata, {"image": image})
        return Code(201, ret_info, "Got the sample.")

    @staticmethod
    def action_cache_image(org_name, dataset_id, dataset_metadata, action_spec, background=True):
        """Cache the image to local cache"""
        if not resolve_existence('dataset', dataset_id):
            return Code(400, {}, f"Dataset Not Exists: {dataset_id}")

        # always cache images locally
        ds_root = os.path.join(stateless_handlers.get_root(), org_name, "datasets", dataset_id)
        cache_path = os.path.join(ds_root, "cache")
        cache_store = dataset_cache_handles.get(dataset_id)
        if not cache_store:
            cache_store = LocalCache(store_path=cache_path)
            dataset_cache_handles[org_name] = cache_store

        image = action_spec.get("image")
        if not image:
            return Code(400, {}, "Invalid Input. Spec is missing `image`")

        cache_id = image
        cache_info = cache_store.get_cache(cache_id)
        if cache_info is None:
            def download_image():
                logger.info("Downloading [%s] in Background (%s)", image, background)
                ep = MonaiDatasetHandler.endpoint(dataset_metadata)
                image_file = os.path.join(cache_path, cache_id, "image")
                save_file = ep.download(image, image_file)
                return cache_store.add_cache(cache_id, save_file, expiry=action_spec.get("ttl", 3600))[1]
            try:
                if background:
                    thread = Thread(target=download_image)
                    thread.start()
                else:
                    cache_info = download_image()
            except Exception as e:
                logger.error("Exception thrown in Monaidatasethandler action_cache_image is %s", str(e))
                return Code(400, {}, f"Cannot cache the image with id {image}. Please check the id and url.")

        return Code(201, cache_info.to_json() if cache_info else None, f"Caching Image: {image}")

    @staticmethod
    def action_notify(org_name, handler_id, spec):
        """Notify the dataset handler with the label info"""
        controller_path = os.path.join(resolve_root(org_name, "dataset", handler_id), "notify_record.json")
        controller_lock = get_default_lock_file_path(controller_path)
        with FileLock(controller_lock):
            image_recorder = ImageLabelRecord(controller_path)
            status, msg = image_recorder.process_data(spec)
            if not status:
                return Code(404, {}, msg)
            image_recorder.export(controller_path)

        return Code(201, {}, "Notification Received")

    @staticmethod
    def run_job(org_name, handler_id, handler_metadata, action, spec):
        """Run the job for the handler"""
        spec = spec if spec else {}
        if action == "nextimage":
            ret_code = MonaiDatasetHandler.action_next_image(org_name, handler_id, handler_metadata, spec)
        elif action == "cacheimage":
            ret_code = MonaiDatasetHandler.action_cache_image(org_name, handler_id, handler_metadata, spec)
            ret_code.data = {}
        elif action == "notify":
            ret_code = MonaiDatasetHandler.action_notify(org_name, handler_id, spec)
        else:
            return Code(404, {}, f"Cannot execute action {action}")

        return ret_code

    @staticmethod
    def cache_image_non_ds(org_name, image, ttl=3600):
        """Cache the image to local cache"""
        if not image:
            return Code(400, {}, "Invalid Input.  Spec is missing `image`")

        image_url = image.strip()
        if not validators.url(image_url):
            return Code(400, {}, "Invalid Image URL. Only URL is accepted")

        ds_root = os.path.join(stateless_handlers.get_root(), org_name)
        if not os.path.exists(ds_root):
            return Code(400, {}, f"User {org_name} Not Exists")

        handler_id = 'non_ds'
        cache_path = os.path.join(ds_root, "cache")
        cache_store = dataset_cache_handles.get(handler_id)
        if not cache_store:
            cache_store = LocalCache(store_path=cache_path)
            dataset_cache_handles[org_name] = cache_store

        cache_id = str(uuid5(NAMESPACE_URL, image_url))
        cache_info = cache_store.get_cache(cache_id)
        if cache_info is None:
            try:
                r = requests.get(image_url, allow_redirects=True, timeout=120)
            except Exception as e:
                logger.error("Exception caught during monai caching requests call: %s", e)
                raise e
            image_file = get_filename_from_cd(image_url, r.headers.get('content-disposition'))
            if not image_file:
                logger.error("Failed to cache %s; Can't determine filename", image_url)
                return Code(400, cache_info, f"Failed to determine Caching Image type: {image}")

            image_file = os.path.join(
                cache_path,
                cache_id,
                image_file
            )  # this will be /shared/orgs/<org_name>/cache/<cache_id>/<image_file>
            if not os.path.exists(os.path.dirname(image_file)):
                os.makedirs(os.path.dirname(image_file), exist_ok=True)  # exist_ok is True for simultaneous requests

            with open(image_file, 'wb') as fp:
                fp.write(r.content)

            file_ext = pathlib.Path(image_file).suffix
            uncompress = file_ext in [".zip", ".tar", ".gztar", ".bztar", ".xztar"]
            _, cache_info = cache_store.add_cache(cache_id, image_file, expiry=ttl, uncompress=uncompress)

        return Code(201, cache_info.to_json() if cache_info else None, f"Caching Image: {image}")

    @staticmethod
    def from_cache(org_name, dataset_id, image, ttl=3600):
        """Get the image from cache"""
        if dataset_id:
            dataset_metadata = resolve_metadata('dataset', dataset_id)
            return MonaiDatasetHandler.action_cache_image(
                org_name,
                dataset_id,
                dataset_metadata,
                {"image": image, "ttl": 3600},
                background=False,
            )
        return MonaiDatasetHandler.cache_image_non_ds(org_name, image, ttl=ttl)

    @staticmethod
    def clean_cache():
        """Clean the expired cache"""
        root_dir = stateless_handlers.get_root()
        if not os.path.exists(root_dir):
            return

        cache_paths = []
        for org_name in os.listdir(root_dir):
            if not os.path.isdir(os.path.join(root_dir, org_name)):
                continue

            path = os.path.join(root_dir, org_name, "cache")
            if os.path.exists(path) and os.path.isdir(path):
                cache_paths.append(path)

            ds_root = os.path.join(root_dir, org_name, "datasets")
            if not os.path.exists(ds_root) or not os.path.isdir(ds_root):
                continue

            for dataset_id in os.listdir(ds_root):
                path = os.path.join(ds_root, dataset_id, "cache")
            if os.path.exists(path) and os.path.isdir(path):
                cache_paths.append(path)

        for path in cache_paths:
            cache_store = LocalCache(store_path=path)
            cache_store.remove_expired()

    @staticmethod
    def get_schema(action):
        """Get the schema for the action"""
        if action == "nextimage":
            return {
                "limit": 1
            }
        if action == "cacheimage":
            return {
                "image": "",
                "ttl": 3600
            }
        if action == "notify":
            return {
                "added": [{
                    "image": "",
                    "label": "",
                }],
                "updated": [],
                "removed": [],
            }
        return {}

    @staticmethod
    def endpoint(metadata):
        """Get the DICOM endpoint by using the configs in the metadata"""
        # TODO Need to implement a function to tell the server automatically. No need for MVP.

        client_id = metadata.get("client_id", None)
        client_secret = metadata.get("client_secret", None)
        filters = metadata.get("filters", None)
        client_url = metadata.get("client_url", None)
        config_path = os.getenv("VAULT_SECRET_PATH", None)
        encryption = NVVaultEncryption(config_path)
        if encryption.check_config()[0]:
            client_secret = encryption.decrypt(client_secret)
        elif not os.getenv("DEV_MODE", "False").lower() in ("true", "1"):
            return None

        dicom_endpoint = DicomEndpoint(
            url=client_url,
            client_id=client_id,
            client_secret=client_secret,
            filters=filters,
        )

        obj_storage_endpoint = ObjectStorageEndpoint(
            url=client_url,
            client_id=client_id,
            client_secret=client_secret,
            filters=filters,
        )

        if obj_storage_endpoint.status_check()[0]:
            return obj_storage_endpoint

        if dicom_endpoint.status_check()[0]:
            return dicom_endpoint

        return None

    @staticmethod
    def status_check(metadata):
        """Check if a MONAI dataset is valid."""
        ep = MonaiDatasetHandler.endpoint(metadata)
        if ep is not None:
            return ep.status_check()

        return False, "Cannot resolve the dataset. Please make sure the dataset information is correct."
