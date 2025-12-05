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

# http://<server>:<port>/<namespace>/api/v1/orgs/<org_name>/experiments?<params>
# ['', '<namespace', 'api', 'v1', 'orgs', '<org_name>', 'experiments']

"""Authentication utils access control modeules"""
from nvidia_tao_core.microservices.utils.core_utils import get_admin_key
# import re
# from nvidia_tao_core.microservices.utils.mongo_utils import MongoHandler


class AccessControlError(Exception):
    """Access Control Error"""

    pass


def validate(user_id, org_name, url, token):
    """Validate org_name requested is accessible to the provided user"""
    user_id = str(user_id)
    err = None
    if url.endswith(":status_update") or url.endswith(":log_update"):
        admin_key = get_admin_key()
        if admin_key != token:
            err = AccessControlError("Invalid token")
            return err
    if not org_name:
        err = AccessControlError("Invalid Org requested")
    # if err is None:
    #     # TODO: Implement TAO NGC user role checking here
    #     mongo = MongoHandler("tao", "users")
    #     user_metadata = mongo.find_one({'id': user_id})
    #     member_of = user_metadata.get('member_of', [])
    #     pattern = fr"{org_name}/.*:(TAO_USER|MAXINE_USER)"
    #     if not any(re.match(pattern, member) for member in member_of):
    #         err = AccessControlError("No access granted for user in org " + org_name)
    return err
