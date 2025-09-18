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

"""Authentication utils validation modules"""
import os
import requests
import uuid
import logging

from nvidia_tao_core.microservices.auth_utils.credentials import decode_jwt_token
from nvidia_tao_core.microservices.auth_utils import session
from nvidia_tao_core.microservices.constants import AIRGAP_DEFAULT_USER

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

#
# class AuthenticationError(Exception):
#     """Authentication Error"""
#
#     pass
#


def _remove_prefix(text, prefix):
    """Removes prefix from given text and returns it"""
    if text.startswith(prefix):
        return text[len(prefix):]
    return text


def get_org_name(url):
    """Extract org name from URL"""
    tmp = _remove_prefix(url, 'http://')
    tmp = _remove_prefix(tmp, 'https://')
    tmp = _remove_prefix(tmp, tmp.split('/')[0])
    tmp = tmp.split('?')[0]
    parts = tmp.split('/')
    # check for user ID match in URL path, with or without domain name in path
    org_name = None
    if (len(parts) >= 5 and parts[3] == 'orgs'):
        org_name = parts[4]
    elif (len(parts) >= 6 and parts[4] == 'orgs'):
        org_name = parts[5]
    return org_name.split(":")[0]


def validate(url, token):
    """Validate Authentication"""
    ngc_api_base_url = 'https://api.ngc.nvidia.com/v2'

    err = None
    user_id = None
    org_name = None
    if not token:
        err = 'Authentication error: missing token'
        return user_id, org_name, err
    org_name = get_org_name(url)
    if not org_name:
        err = 'Authentication error: missing org_name in url'
        return user_id, org_name, err
    # Retrieve unexpired session, if any
    user = session.get_session(token, org_name)
    if user:
        user_id = user.get('id')
    if user_id and org_name:
        logger.info("Found session for user: %s in org %s", str(user_id), org_name)
        return str(user_id), org_name, err
    # Fall back on NGC to validate
    headers = {'Accept': 'application/json'}
    jwt_token = None
    if token:
        # Attempt to validate JWT Token
        creds, err = decode_jwt_token(token)
        if creds:
            user_id = creds.get('user_id')
            org_name = creds.get('org_name')
            jwt_token = token
            token = creds.get('user_key')
        headers['Authorization'] = 'Bearer ' + token
    try:
        headers['Accept-Encoding'] = 'identity'
        r = requests.get(f'{ngc_api_base_url}/users/me', headers=headers, timeout=120)
    except Exception as e:
        logger.error("Exception caught during getting NGC user info: %s", e)
        raise e
    if r.status_code != 200:
        if err:  # JWT Decode Error
            return user_id, org_name, str(err)
        err = 'Authentication error: ' + r.json().get("requestStatus", {}).get("statusDescription", "Unknown NGC user")
        return user_id, org_name, err
    ngc_user_id = r.json().get('user', {}).get('id')
    if not ngc_user_id:
        err = 'Authentication error: Unknown NGC user ID'
        return user_id, org_name, err
    user_id = str(uuid.uuid5(uuid.UUID(int=0), str(ngc_user_id)))
    logger.info("New session for user: %s", str(user_id))
    # Create a new or update an expired session
    member_of = []
    roles = r.json().get('user', {}).get('roles', [])
    for role in roles:
        org = role.get('org', {}).get('name', '')
        team = role.get('team', {}).get('name', '')
        for role_type in ("orgRoles", "teamRoles"):
            entitlements = role.get(role_type, [])
            if not entitlements:
                member_of.append(f"{org}/{team}")
            else:
                for entitlement in entitlements:
                    member_of.append(f"{org}/{team}:{entitlement}")
    extra_user_metadata = {'member_of': member_of}
    if jwt_token:
        session.set_session(user_id, org_name, jwt_token, extra_user_metadata)
    if token:
        session.set_session(user_id, org_name, token, extra_user_metadata)
    return user_id, org_name, None


def get_user_id(authorization: str, org_name: str) -> str:
    """Checks authorization header and returns user_id associated with token"""
    # special user id for air-gapped environments
    if os.getenv("AIRGAPPED_MODE", "false").lower() == "true":
        user_id = authorization.removeprefix("Bearer ").strip() if authorization else ""
        if not user_id:
            user_id = str(uuid.uuid5(uuid.UUID(int=0), AIRGAP_DEFAULT_USER))
        logger.info("Air-gapped mode auth → user_id '%s'", user_id)
        return user_id

    # Get user ID
    authorization_parts = authorization.split()
    token = None
    if len(authorization_parts) == 2 and authorization_parts[0].lower() == 'bearer':
        token = authorization_parts[1]

    user = session.get_session(token, org_name)
    user_id = None
    if user:
        user_id = user.get('id')

    return user_id
