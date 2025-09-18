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

"""DICOM API module"""
import copy
import os
import time
from typing import Any, Dict, Optional, Tuple
from urllib.parse import parse_qsl

from cachetools import TTLCache, cached
from dicomweb_client import DICOMwebClient
from dicomweb_client.session_utils import create_session_from_user_pass
from pydicom import Dataset
from json import JSONDecodeError
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

DICOM_FIELDS = [
    "StudyDate",
    "StudyTime",
    "Modality",
    "RetrieveURL",
    "PatientID",
    "StudyInstanceUID",
]


class DicomEndpoint:
    """DICOM Endpoint class"""

    def __init__(self, url: str, client_id, client_secret, filters):
        """Initialize DicomEndpoint class

        Args:
            url: DICOMweb endpoint URL
            client_id: Client ID for DICOMweb endpoint
            client_secret: Client Secret for DICOMweb endpoint
            filters: Filters for DICOMweb endpoint
        """
        self.url = url.rstrip("/")
        self.client_id = client_id
        self.client_secret = client_secret
        self.filters = filters

        self.fuzzymatching = None
        self.limit = None
        self.offset = None
        self.search_filters = {}
        self.fields = [""]

        if self.filters:
            for k, v in parse_qsl(self.filters):
                if k == "includefield":
                    self.fields.append(v)
                elif k == "limit":
                    self.limit = int(v)
                elif k == "offset":
                    self.offset = int(v)
                elif k == "fuzzymatching":
                    self.fuzzymatching = v in {True, "yes", "y", "true", "t", "1", "on"}
                else:
                    self.search_filters[k] = v
        else:
            self.search_filters = {"Modality": "CT"}

    def _client(self):
        session = None
        # TODO:: Support other modes (Cert and BearerToken)
        # Refer: https://dicomweb-client.readthedocs.io/en/latest/usage.html#authentication-and-authorization
        if self.client_id and self.client_secret:
            session = create_session_from_user_pass(self.client_id, self.client_secret)
        return DICOMwebClient(url=self.url, session=session)

    def _studies(self, max_limit=0):
        return self._client().search_for_studies(
            fuzzymatching=self.fuzzymatching,
            limit=max(max_limit, self.limit) if self.limit else self.limit,
            offset=self.offset,
            fields=self.fields,
            search_filters=self.search_filters,
        )

    def status_check(self) -> Tuple[bool, Optional[str]]:
        """Check if the status of the DICOMweb endpoint is healthy"""
        try:
            studies = self._studies(max_limit=1)
            logger.info("Total Studies Found: %d", len(studies))
            return True, None
        except Exception as e:
            logger.error(str(e))
            return False, str(e)

    @cached(cache=TTLCache(maxsize=16, ttl=180))
    def list_images(self) -> Dict[str, Any]:
        """List all images from the DICOMweb endpoint. TTLCache is used to cache the results for 3 minutes"""
        client = self._client()

        datasets = client.search_for_series(
            fuzzymatching=self.fuzzymatching,
            limit=self.limit,
            offset=self.offset,
            fields=self.fields,
            search_filters=self.search_filters,
        )

        images = {}
        for ds in datasets:
            try:
                d = Dataset.from_json(ds)
                series = str(d["SeriesInstanceUID"].value)
                images[series] = self._meta_info(series, d)
            except JSONDecodeError as e:
                logger.error("JSON decode error for dataset: %s - %s - %s", datasets, ds, e)
                continue  # Skip this dataset and continue with the next

        logger.info("Total Images: %d", len(images))
        return images

    @cached(cache=TTLCache(maxsize=16, ttl=180))
    def list_all(self) -> Dict[str, Any]:
        """List all images and labels from the DICOMweb endpoint. TTLCache is used to cache the results for 3 minutes"""
        all_images = self.list_images()
        all_labels = self.list_labels()

        for k, v in all_labels.items():
            series = v["ReferencedSeriesUID"]
            if series in all_images:
                labels = all_images[series].get("labels", [])
                labels.append(k)
                all_images[series]["labels"] = labels

        logger.info(
            "All Images+Label: %d; All Labels: %d",
            len(all_images), len(all_labels)
        )
        return all_images

    def list_labels(self) -> Dict[str, Any]:
        """List all labels from the DICOMweb endpoint"""
        client = self._client()

        search_filters = copy.deepcopy(self.search_filters)
        search_filters["Modality"] = "SEG"
        datasets = client.search_for_series(
            fuzzymatching=self.fuzzymatching,
            limit=self.limit,
            offset=self.offset,
            fields=self.fields,
            search_filters=search_filters,
        )

        labels = {}
        for ds in datasets:
            d = Dataset.from_json(ds)
            series = str(d["SeriesInstanceUID"].value)

            meta = client.retrieve_series_metadata(
                str(d["StudyInstanceUID"].value), series
            )
            seg_meta = Dataset.from_json(meta[0])
            if seg_meta.get("ReferencedSeriesSequence"):
                referenced_series_instance_uid = str(
                    seg_meta["ReferencedSeriesSequence"]
                    .value[0]["SeriesInstanceUID"]
                    .value
                )
                labels[series] = {
                    **self._meta_info(series, d),
                    "ReferencedSeriesUID": referenced_series_instance_uid,
                }
            else:
                logger.warning(
                    "Label Ignored:: ReferencedSeriesSequence is NOT found: %s", series
                )

        logger.info("Total Labels: %d", len(labels))
        return labels

    @cached(cache=TTLCache(maxsize=16, ttl=180))
    def get_labeled_images(self):
        """Get all images with labels"""
        return {k: v for k, v in self.list_all().items() if v.get("labels")}

    def get_unlabeled_images(self) -> Dict[str, Any]:
        """Get all images without labels"""
        return {k: v for k, v in self.list_all().items() if not v.get("labels")}

    def get_info(self, uid):
        """Get meta info for a given image id"""
        client = self._client()
        ds = Dataset.from_json(
            client.search_for_series(search_filters={"SeriesInstanceUID": uid})[0]
        )
        return self._meta_info(uid, ds)

    def _meta_info(self, uid, ds):
        info = {"SeriesInstanceUID": uid}
        for f in DICOM_FIELDS:
            info[f] = str(ds[f].value) if ds.get(f) else "UNK"
        return info

    def download(self, uid, save_dir):
        """Download a given image id to a given directory"""
        start = time.time()
        info = self.get_info(uid)
        study_id = info["StudyInstanceUID"]

        os.makedirs(save_dir, exist_ok=True)

        client = self._client()
        instances = client.retrieve_series(study_id, uid)
        for instance in instances:
            instance_id = str(instance["SOPInstanceUID"].value)
            file_name = os.path.join(save_dir, f"{instance_id}.dcm")
            instance.save_as(file_name)

        logger.info(
            "Time to download %s: %.3f (sec)", uid, time.time() - start
        )
        return save_dir
