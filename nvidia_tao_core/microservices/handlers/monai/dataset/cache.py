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

"""MONAI Cache API module"""
import json
import logging
import os
import shutil
import tempfile
import time
from typing import Dict
from filelock import FileLock

logger = logging.getLogger(__name__)


class CacheInfo:
    """CacheInfo class"""

    def __init__(self, c=None):
        """Initialize CacheInfo class"""
        if c is None:
            c = {}
        self.name: str = c.get("name", "")
        self.path: str = c.get("path", "")
        self.image: str = c.get("image", "")
        self.meta: Dict = c.get("meta", {})
        self.create_ts: int = c.get("create_ts", 0)
        self.last_access_ts: int = c.get("last_access_ts", 0)
        self.expiry: int = c.get("expiry", 0)

    def to_str(self, indent=None):
        """Convert CacheInfo to string"""
        return json.dumps(self.__dict__, indent=indent)

    def to_json(self, indent=None):
        """Convert CacheInfo to json"""
        return json.loads(self.to_str(indent))


class LocalCache(dict):
    """LocalCache class"""

    def __init__(self, store_path: str, expiry: int = 3600):
        """Initialize LocalCache class

        Args:
            store_path: Path to store cache
            expiry: Expiry time for cache
        """
        dict.__init__(self)
        self.store_path = store_path
        self.expiry = expiry if expiry > 60 else 3600
        self._lock_file = os.path.join(store_path, ".lock")

    def remove_expired(self):
        """Remove expired cache"""
        count = 0
        current_ts = int(time.time())

        if not os.path.isdir(self.store_path):
            return count

        for item in os.listdir(self.store_path):
            if os.path.isdir(os.path.join(self.store_path, item)):
                cache_id = item
                cache_info = self.get_cache(cache_id, update_ts=False, fetch_cache=False)
                if cache_info is None:
                    # Handle dangling session-id
                    continue
                expiry_ts = cache_info.last_access_ts + cache_info.expiry

                if cache_info and cache_info.expiry > 0 and expiry_ts < current_ts:
                    logger.info(f"Removing expired; current ts: {current_ts}\n{cache_info.to_str()}")
                    self.remove_cache(cache_id)
                    count += 1
                elif cache_info:
                    logger.debug(
                        f"Skipped {cache_id}; current ts: {current_ts}; last ts: {cache_info.last_access_ts}; "
                        f"expiry: {cache_info.expiry}"
                    )
                else:
                    logger.info(f"Invalid session-id: {cache_id} (will be removed)")
                    self.remove_cache(cache_id)
        return count

    def get_cache(self, cache_id: str, update_ts: bool = True, fetch_cache: bool = True):
        """Get cached item

        Args:
            cache_id: Cache ID
            update_ts: Update timestamp
            fetch_cache: switch to decide if it should fetch cache
        """
        cache_info = self.get(cache_id) if fetch_cache else None
        if cache_info is None:
            meta_file = os.path.join(self.store_path, cache_id, "meta.info")
            if os.path.exists(meta_file):
                with FileLock(self._lock_file):
                    try:
                        with open(meta_file, encoding="utf-8") as meta:
                            content = meta.readline()
                            if content:
                                parsed_content = json.loads(content)
                                cache_info = CacheInfo(c=parsed_content)
                                self[cache_id] = cache_info
                            else:
                                # Handle empty file case
                                logger.error("The meta file is empty.")
                    except json.JSONDecodeError as e:
                        # Handle invalid JSON
                        logger.error("Error parsing JSON: %s", e)
        if cache_info:
            cache_info_image_path = cache_info.image[0] if isinstance(cache_info.image, list) else cache_info.image
            if not os.path.exists(cache_info_image_path):
                logger.info(f"Dangling session-id: {cache_id} (will be removed)")
                self.remove_cache(cache_id)
                cache_info = None

        if cache_info and update_ts:
            cache_info.last_access_ts = int(time.time())
            self._write_meta_info(cache_id, cache_info)
        return cache_info

    def remove_cache(self, cache_id: str):
        """Remove cached item"""
        cache_info = self.get(cache_id)
        if cache_info:
            self.pop(cache_id)
        path = os.path.join(self.store_path, cache_id)
        shutil.rmtree(path, ignore_errors=True)

    def add_cache(self, cache_id, data_file, expiry: int = 0, uncompress: bool = False):
        """Add cached item"""
        start = time.time()
        logger.debug(f"Load Data from: {data_file}")
        if isinstance(data_file, str):
            if os.path.isdir(data_file):
                image_path = data_file
                logger.debug(f"Input Dir (Multiple Input): {image_path}")
            else:
                image_path = data_file
                logger.debug(f"Input File (Single): {image_path}")

                if uncompress:
                    with tempfile.TemporaryDirectory() as tmp_folder:
                        logger.debug(f"UnArchive: {image_path} to {tmp_folder}")
                        shutil.unpack_archive(data_file, tmp_folder)
                        image_path = tmp_folder

        else:
            # example:
            # data_file = ['/image_path/image1.nii', '/image_path/image2.nii', ...]
            image_path = os.path.dirname(data_file[0])
            logger.debug(f"Input List: {data_file}")

        path = os.path.join(self.store_path, cache_id)
        expiry = expiry if expiry > 0 else self.expiry

        logger.debug(f"Using Path: {path} to save session")
        os.makedirs(path, exist_ok=True)

        meta: Dict = {}
        basename = os.path.basename(image_path)

        image_file = os.path.join(path, basename)
        if image_file != image_path and os.path.exists(image_path):
            shutil.move(image_path, image_file)

        cache_info = CacheInfo()
        cache_info.name = cache_id
        cache_info.path = path
        cache_info.image = image_file if isinstance(data_file, str) else data_file
        cache_info.meta = meta
        cache_info.create_ts = int(time.time())
        cache_info.last_access_ts = int(time.time())
        cache_info.expiry = min(expiry, self.expiry)

        self._write_meta_info(cache_id, cache_info)

        self[cache_id] = cache_info
        logger.info(f"++ Time consumed to add session {cache_id}: {time.time() - start}")
        return cache_id, cache_info

    def _write_meta_info(self, cache_id, cache_info):
        with FileLock(self._lock_file):
            path = os.path.join(self.store_path, cache_id)
            meta_file = os.path.join(path, "meta.info")
            with open(meta_file, "w", encoding="utf-8") as meta:
                meta.write(cache_info.to_str())
