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

"""Base class for a dataset endpoint."""
from abc import ABC, abstractmethod


class BaseEndpoint(ABC):
    """The is the basic class for an endpoint of a dataset.

    All methods contained in this class should be overrided or inherited to enable annotation and batch training
    workflows.
    After that, the new endpoint can be used by the MonaiDatasetHandler.
    """

    def __init__(self, url: str, client_id, client_secret, filters):
        """Set the url, client_id, client_secret and filters for the endpoint."""
        self.url = url.rstrip('/')
        self.client_id = client_id
        self.client_secret = client_secret
        self.filters = filters

    @abstractmethod
    def status_check(self):
        """Check if the network connection is OK. The subclass must implement this class."""
        pass

    @abstractmethod
    def list_all(self):
        """Get all samples."""
        pass

    def get_labeled_images(self):
        """Get all labeled samples."""
        return {k: v for k, v in self.list_all().items() if v.get("labels")}

    def get_unlabeled_images(self):
        """Get all unlabeled samples."""
        return {k: v for k, v in self.list_all().items() if not v.get("labels")}

    @abstractmethod
    def get_info(self, uid):
        """Get meta info of an image."""
        pass

    @abstractmethod
    def download(self, uid, filepath):
        """Download the image named by given id to the save_dir."""
        pass
