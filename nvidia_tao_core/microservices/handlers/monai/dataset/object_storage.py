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

"""Object storage client"""
import json
import os
import tempfile
import time
from urllib.parse import urlparse
import logging

from libcloud.storage.providers import get_driver
from libcloud.storage.types import Provider

from .base import BaseEndpoint

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_manifest_name(url):
    """Get the manifest name from a url."""
    url_filename = os.path.basename(url)
    return url_filename if url_filename.endswith(".json") else "manifest.json"


class ObjectStorageClient:
    """A client to get information from the object storage."""

    def __init__(self, manifest_content, root_path_default=""):
        """Init the client with the content from manifest file.

        Args:
            manifest_content: the content of the manifest file.
            root_path_default: the default root path of the object storage.
        """
        self.manifest_content = manifest_content
        self.root_path_default = root_path_default

    def _retrieve_image(self, image_id):
        """Retrieve the image with given image_id."""
        for sample in self.manifest_content["data"]:
            if sample["image"].get("id") and image_id == sample["image"]["id"]:
                return sample["image"]

            if (
                sample.get("label") and
                isinstance(sample["label"], dict) and
                sample["label"].get("id") and
                image_id == sample["label"]["id"]
            ):
                return sample["label"]
        return None

    def get_sample_list(self):
        """Get all valid samples from the manifest content."""
        sample_list = self.manifest_content.get("data", [])
        sample_list = [x for x in sample_list if x.get("image") and x["image"].get("path") and x["image"].get("id")]
        return sample_list

    @staticmethod
    def _retrieve_path(sample, key):
        """Retrieve the path of given key from a sample.

        A sample is like:
        {
            "image": {
                "path": [
                    "imagesTr/spleen_19.nii.gz"
                ],
                "id": "722782b1-a9af-4c3c-8aa8-c88d3b34d934"
            },
            "label": {
                "path": [
                    "labelsTr/spleen_19.nii.gz"
                ],
                "id": "8fcd2cdf-c7f8-41e3-a5ba-57e2f8744298"
            }
        }
        """
        if not isinstance(sample, dict) or not sample.get(key, None):
            return ""

        key_dict = sample.get(key)
        if not isinstance(key_dict, dict) or not key_dict.get("path", None):
            return ""

        path_value = key_dict.get("path")
        if not isinstance(path_value, list) and not isinstance(path_value, str):
            return ""

        if path_value and isinstance(path_value, list):
            return path_value[0]

        if path_value and isinstance(path_value, str):
            return path_value

        return ""

    def get_sample_paths(self, image_key="image", label_key="label"):
        """Get all image and label paths in the manifest file."""
        sample_list = self.manifest_content.get("data", None)
        path_list = []
        for sample in sample_list:
            image_path = self._retrieve_path(sample, image_key)
            label_path = self._retrieve_path(sample, label_key)
            if image_path:
                path_list.append(image_path)

            if label_path:
                path_list.append(label_path)

        return path_list

    def get_root_path(self):
        """Get the url root path from manifest content."""
        return self.manifest_content.get("root_path", self.root_path_default)

    def get_image_path(self, image_id):
        """Get the path list of an image with given image_id."""
        sample_info = self._retrieve_image(image_id)
        return sample_info.get("path", None)


class ObjectStorageEndpoint(BaseEndpoint):
    """This class is a client of the cloud object storage like: s3, azure_blob or google drive.

    Object storage is a technology that stores and manages data in an unstructured format called objects.
    Modern organizations create and analyze large volumes of unstructured data such as photos, videos, email,
    web pages, sensor data, and audio files. Cloud object storage systems distribute this data across
    multiple physical devices but allow users to access the content efficiently from a single, virtual storage
    repository.

    For details about the object storage, please refer to links below:
    https://aws.amazon.com/what-is/object-storage/?nc1=h_ls
    https://cloud.google.com/learn/what-is-object-storage
    https://azure.microsoft.com/en-us/products/storage/blobs
    """

    PROVIDER_LIST = [Provider.S3, Provider.AZURE_BLOBS]

    def __init__(self, url=None, client_id=None, client_secret=None, filters=None):
        """Init function

        url: url to manifest file or root path of a manifest file.
        client_id: the client key or id to get access to the cloud object storage account.
        client_secret: the secret or access token to get access to the cloud object storage account.
        filters: rules to filter the samples.
        """
        # Url could be https://myaccount.blob.core.windows.net/containername/mydataset which is the root
        # of a manifest file. Or https://myaccount.blob.core.windows.net/containername/mydataset/manifest.json
        # which is the link to manifest file.
        super().__init__(url, client_id, client_secret, filters)
        self.driver = None
        # NOTE: prefix is the subfolder(s) in the container.
        (
            self.container_name,
            self.prefix
        ) = self._parse_container_name(self.url)
        self.download_retry_times = 3
        # The input url is one of these:
        # - https://myaccount.blob.core.windows.net/containername/mydataset/manifest.json
        # - https://myaccount.blob.core.windows.net/containername/mydataset/train_datalist.json
        # - https://myaccount.blob.core.windows.net/containername/mydataset/
        # the manifest JSON need to be removed to get the right prefix.
        if self.prefix is not None and self.prefix.endswith(".json"):
            self.prefix = os.path.dirname(self.prefix)
        self.root_path_default = self.url if not self.url.endswith(".json") else os.path.dirname(self.url)

    def _check_connection(self, provider):
        """Check if a connection can be set with given provider."""
        try:
            Driver = get_driver(provider)
            driver = Driver(key=self.client_id, secret=self.client_secret)
            # The AWS S3 bucket has a more specific roles and the secret may not be able to list all containers.
            container = driver.get_container(self.container_name)
            container.list_objects()
        except Exception:
            return False

        self.driver = driver
        return True

    @staticmethod
    def _parse_container_name(url):
        """Parse the container name and prefix from a url."""
        parser_result = urlparse(url)
        container_name = None
        prefix = None
        if parser_result.scheme.lower() == "s3":
            # Example: s3://containername/objectname
            container_name = parser_result.netloc
            prefix = parser_result.path.lstrip("/")
        if "s3.amazonaws.com" in parser_result.netloc:
            # Example: https://containername.s3.amazonaws.com/objectname/
            container_name = parser_result.netloc.split(".")[0]
            prefix = parser_result.path.lstrip("/")
        if "blob.core.windows.net" in parser_result.netloc:
            # Example: https://myaccount.blob.core.windows.net/containername
            path_split = parser_result.path.lstrip("/").split("/", 1)
            container_name = path_split[0]
            prefix = path_split[1] if len(path_split) == 2 else ""
        return container_name, prefix

    def _get_driver(self):
        """Get the appropriate storage driver.

        Returns:
            driver: The storage driver based on the configured storage type.
        """
        if self.driver is not None:
            return self.driver

        driver = None
        for provider in self.PROVIDER_LIST:
            # Try to connect to the server.
            connected = self._check_connection(provider)
            if connected:
                driver = self.driver
                break

        return driver

    def _get_client(self):
        """Get a client to process the object storage info."""
        # Download the manifest file if there is no manifest client.
        _manifest_name = get_manifest_name(self.url)
        manifest_url = (
            os.path.join(self.url, _manifest_name)
            if not self.url.endswith(".json")
            else self.url
        )
        with tempfile.TemporaryDirectory() as filepath:
            manifest_name = os.path.join(filepath, _manifest_name)
            self._download_by_url(manifest_url, manifest_name)
            with open(manifest_name, "r", encoding="utf8") as fp:
                manifest_content = json.load(fp)
            return ObjectStorageClient(manifest_content, root_path_default=self.root_path_default)

    def _get_all_objects(self):
        """Get all objects in the cloud storage."""
        object_set = {}
        driver = self._get_driver()
        if not driver:
            return object_set

        url = self.url if not self.url.endswith(".json") else os.path.dirname(self.url)
        container = driver.get_container(self.container_name)
        objs = container.list_objects(prefix=self.prefix)
        object_set = {os.path.join(url, x.name.removeprefix(self.prefix).lstrip("/")) for x in objs}
        return object_set

    def _check_url(self):
        """Check if the url attribute is valild."""
        try:
            self._get_client()
        except Exception as e:
            logger.error("Exception thrown in check_url is %s", str(e))
            return False
        return True

    def _check_manifest(self):
        """Check if the manifest content is correct."""
        manifest_client = self._get_client()
        manifest_url_root = manifest_client.get_root_path()
        sample_list = manifest_client.get_sample_list()
        if not sample_list:
            return False, "Must provide samples in the data dict of the manifest file."

        paths_list = [os.path.join(manifest_url_root, x) for x in manifest_client.get_sample_paths()]
        object_set = self._get_all_objects()
        paths_exist = [x in object_set for x in paths_list]
        if not all(paths_exist):
            return (
                False,
                "Some samples don't exist in the given object storage. "
                "Please check path correctness of the manifest file."
            )

        return True, None

    def _download_by_url(self, url, filename):
        """Download one object with given url."""
        driver = self._get_driver()
        if driver is None:
            raise ValueError(f"Cannot find a suitable driver for the url {url}.")
        # The prefix must be a path to a file.
        container_name, prefix = self._parse_container_name(url)
        container = driver.get_container(container_name)
        obj = container.list_objects(prefix=prefix)[0]
        savepath = os.path.dirname(filename)
        os.makedirs(savepath, exist_ok=True)
        driver.download_object(obj, filename, overwrite_existing=True)
        logger.info("Downloading %s to %s", url, filename)

    def download(self, uid, filepath):
        """Download an image to local path according to the `id` parameter."""
        client = self._get_client()
        # Could be a list of dcm files or one file.
        image_path = client.get_image_path(uid)
        image_path = image_path if isinstance(image_path, list) else [image_path]
        root_path = client.get_root_path()
        image_urls = [os.path.join(root_path, x) for x in image_path]
        filenames = [os.path.join(filepath, os.path.basename(x)) for x in image_path]
        for image_url, filename in zip(image_urls, filenames):
            self._download_by_url(image_url, filename)
        return filenames if len(filenames) > 1 else filenames[0]

    def list_all(self):
        """List all samples in the cloud storage."""
        client = self._get_client()
        sample_list = client.get_sample_list()
        images = {}
        for sample in sample_list:
            image_id = sample["image"].get("id", None)
            label = sample.get("label", None)
            label_id = sample["label"].get("id", None) if label is not None else None
            images[image_id] = {
                "labels": [label_id] if label_id is not None else []
            }
        return images

    def get_info(self, uid):
        """Get extra info of the sample with given id."""
        return {}

    def download_all_objects(self, filepath):
        """Download all objects in the given url to local path."""
        # TODO Need to add multiprocess or pool to make the download more efficient.
        driver = self._get_driver()
        if driver is None:
            raise ValueError(f"Cannot find a suitable driver for the url {self.url}.")

        container_name = self.container_name
        container = driver.get_container(container_name)
        objects = container.list_objects(prefix=self.prefix)
        start_time = time.time()
        for obj in objects:
            file_name = os.path.join(filepath, obj.name.removeprefix(self.prefix + "/"))
            file_path = os.path.dirname(file_name)
            os.makedirs(file_path, exist_ok=True)
            for i in range(self.download_retry_times):
                try:
                    driver.download_object(obj, file_name, overwrite_existing=True)
                except Exception as e:
                    logger.error(
                        f"#{i}: Cannot download file {file_name} from {container_name} in {self.url}.",
                        exc_info=True
                    )
                    if i == self.download_retry_times - 1:
                        raise TimeoutError(
                            f"Cannot download file {file_name} from {container_name} in {self.url}."
                        ) from e
                    continue
                break

        logger.info(
            "Time to download object storage container %s: %0.3f (sec)",
            self.url,
            time.time() - start_time
        )
        return filepath

    def status_check(self):
        """Check if this endpoint is valid."""
        driver = self._get_driver()
        if driver is None:
            return False, f"Failed to connect to URL {self.url}. Please check the given url, id and secret."

        url_valid = self._check_url()
        if not url_valid:
            return True, f"Please check if the url {self.url} is correct."

        manifest_valid, msg = self._check_manifest()
        if not manifest_valid:
            return True, msg

        return True, None
