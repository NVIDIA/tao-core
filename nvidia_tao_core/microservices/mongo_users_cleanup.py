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
"""Script to periodically cleanup expired session tokens"""
from datetime import datetime, timezone
import logging
import os

from nvidia_tao_core.microservices.utils.mongo_utils import MongoHandler

__SESSION_EXPIRY_SECONDS__ = 86400  # Equal to 24 hours

# Configure logging
TAO_LOG_LEVEL = os.getenv('TAO_LOG_LEVEL', 'INFO').upper()
tao_log_level = getattr(logging, TAO_LOG_LEVEL, logging.INFO)
logging.basicConfig(
    level=logging.WARNING,  # Root logger: suppress third-party DEBUG logs
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logging.getLogger('nvidia_tao_core').setLevel(tao_log_level)
logger = logging.getLogger(__name__)


def run():
    """Script to periodically cleanup expired session tokens"""
    mongo_users = MongoHandler("tao", "users")
    users = mongo_users.find(None)
    for user in users:
        new_tokens = []
        user_id = user.get('id')
        for token_info in user.get("token_info", []):
            if 'last_modified' in token_info:
                dt_delta = datetime.now(tz=timezone.utc) - token_info['last_modified']
                if dt_delta.total_seconds() < __SESSION_EXPIRY_SECONDS__:  # replace expired token
                    new_tokens.append(token_info)
        logger.info("Length of tokens before %d for user %s", len(user.get('token_info', [])), user_id)
        user["token_info"] = new_tokens
        logger.info("Length of tokens after %d for user %s", len(user.get('token_info', [])), user_id)
        mongo_users.upsert({'id': user_id}, user)


if __name__ == '__main__':
    run()
