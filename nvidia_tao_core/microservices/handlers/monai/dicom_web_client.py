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

"""Medical communication toolbox"""

import requests
import json
import logging

from nvidia_tao_core.microservices.handlers.utilities import Code, TAOResponse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DicomWebClient:
    """Class to communicate with the dicom web."""

    successful_code = [200, 201]
    image_type = ["CT", "MR"]

    def __init__(self, location, user_id, access_key):
        """Initializes the DicomWebClient.

        Args:
            location (str): Base URL of the DICOM web server.
            user_id (str): User ID for authentication.
            access_key (str): Access key for authentication.
        """
        self.location = location
        self.user_id = user_id
        self.access_key = access_key

    def _get_content(self, content, dicom_filter=None):
        """Retrieves content from the DICOM web server.

        Args:
            content (str): URL path segment to append to the base location.
            dicom_filter (dict, optional): Filter criteria for retrieving specific samples.

        Returns:
            Code: Response object containing:
                - 201 if content is successfully retrieved.
                - Error code if the request fails.
        """
        content_url = f"{self.location}/{content}"
        if dicom_filter is None:
            try:
                response = requests.get(content_url, auth=(self.user_id, self.access_key), timeout=120)
                if response.status_code not in self.successful_code:
                    msg = f"Cannot get {content_url}"
                    return Code(response.status_code, {}, msg)
                return Code(201, response.json(), "Got the content.")
            except Exception as e:
                logger.error("Exception caught during getting Dicom web content: %s", e)
                return Code(500, {}, "Exception caught during getting Dicom web content")
        return Code(404, {}, "Issue in getting Dicom web content")

    def _get_content_data(self, content, dicom_filter=None):
        """Fetches content from the DICOM web server and returns JSON data.

        Args:
            content (str): URL path segment to append to the base location.
            dicom_filter (dict, optional): Filter criteria for retrieving specific samples.

        Returns:
            dict: JSON content if retrieval is successful.
            Code: Error code if retrieval fails.
        """
        content_code = self._get_content(content, dicom_filter)
        if content_code.code in self.successful_code:
            return content_code.data
        return content_code

    def generate_segmentation_sample(self, study_url, image_type, dicom_filter):
        """Generates a segmentation sample from a given DICOM study.

        Currently supports only CT and MR images.

        Args:
            study_url (str): URL of the study.
            image_type (str): Modality type to be treated as images (e.g., "CT", "MR").
            dicom_filter (dict): Filter criteria for retrieving samples.

        Returns:
            dict: Image-label pair if retrieval is successful.
            Code: Error code if retrieval fails.
        """
        sample = None
        dicom_web_study_dict = self._get_content_data(study_url, dicom_filter)
        # When met something wrong, here will be a TAOResponse to return the error code.
        if isinstance(dicom_web_study_dict, TAOResponse):
            return dicom_web_study_dict
        dicom_web_series = dicom_web_study_dict.get("Series", dicom_filter)
        # If the study is empty, just return nothing.
        if dicom_web_series is None:
            return None

        sample = {"study": study_url, "image": "", "label": "", "update_time": ""}
        # TODO Need to support multi series and update in backend.
        for series in dicom_web_series:
            series_url = f"series/{series}"
            series_dict = self._get_content_data(series_url, dicom_filter)
            if isinstance(series_dict, TAOResponse):
                return series_dict
            if series_dict["MainDicomTags"]["Modality"] in image_type:
                sample["image"] = series_url
            elif series_dict["MainDicomTags"]["Modality"] == "SEG":
                sample["label"] = series_url
        return sample

    def generate_dataset_manifest_dict(self, annotation_type="SEG", dicom_filter=None):
        """Generates a dataset manifest dictionary from the DICOM web server.

        The manifest contains information about studies, image-label pairs, and metadata.

        Args:
            annotation_type (str, optional): Type of annotation (default: "SEG").
            dicom_filter (dict, optional): Filter criteria for retrieving samples.

        Returns:
            Code: Response object containing:
                - 201 with the manifest dictionary if successful.
                - 400 if an unsupported annotation type is provided.
                - Error code if retrieval fails.
        """
        manifest_dict = {"location": self.location,
                         "user_id": self.user_id,
                         "access_key": self.access_key,
                         "data": [],
                         "force_fetch": False,
                         "labeled_list": [],
                         "unlabeled_list": []}
        current_url = "studies"
        dicom_web_studies = self._get_content_data(current_url, dicom_filter)
        # When met something wrong, here will be a TAOResponse to return the error code.
        if isinstance(dicom_web_studies, TAOResponse):
            return dicom_web_studies
        for dicom_web_study in dicom_web_studies:
            current_url = f"studies/{dicom_web_study}"
            if annotation_type == "SEG":
                sample = self.generate_segmentation_sample(current_url, self.image_type, dicom_filter)
                if isinstance(sample, TAOResponse):
                    return sample
                sample_index = len(manifest_dict["data"])
                index_info = {"index": sample_index, "fetch_time": 0.0, "al_score": 0.0}
                if sample["label"]:
                    manifest_dict["labeled_list"].append(index_info)
                else:
                    manifest_dict["unlabeled_list"].append(index_info)
                manifest_dict["data"].append(sample)
            else:
                return Code(400, {}, f"Doesn't support annotation type {annotation_type}.")
        return Code(201, manifest_dict, "Got the dict.")

    def create_dataset_manifest_file(self, manifest_path, annotation_type="SEG", dicom_filter=None):
        """Creates a dataset manifest file at the specified path.

        Args:
            manifest_path (str): File path to save the manifest.
            annotation_type (str, optional): Type of annotation (default: "SEG").
            dicom_filter (dict, optional): Filter criteria for retrieving samples.

        Returns:
            Code: Response object containing:
                - 201 if the manifest file is successfully saved.
                - 400 if an error occurs while writing the file.
        """
        manifest_dict_status = self.generate_dataset_manifest_dict(annotation_type, dicom_filter)
        if manifest_dict_status.code == 201:
            manifest_dict = manifest_dict_status.data
        else:
            return manifest_dict_status

        try:
            with open(manifest_path, "w", encoding='utf-8') as f:
                f.write(json.dumps(manifest_dict, indent=4))
        except Exception as e:
            logger.error("Exception thrown in create_dataset_manifest_file is %s", str(e))
            return Code(400, {}, f"Cannot write the {manifest_path}.")

        return Code(201, {}, "Saved the manifest file.")
