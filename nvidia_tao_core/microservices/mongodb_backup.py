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

"""MongoDB backup script"""
import argparse
import logging

from nvidia_tao_core.microservices.utils import run_system_command
from nvidia_tao_core.microservices.handlers.cloud_storage import CloudStorage
from nvidia_tao_core.microservices.handlers.mongo_handler import mongo_connection_string

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def backup(access_key, secret_key, s3_bucket_name, endpoint_url=None, region=None):
    """Script to backup mongodump file to S3 bucket"""
    if not access_key or not secret_key or not s3_bucket_name:
        logger.error("Invalid arguments. Check script arguments and try again.")
        return

    try:
        cs_instance = CloudStorage(
            cloud_type="aws",
            bucket_name=s3_bucket_name,
            region=region,
            key=access_key,
            secret=secret_key,
            client_kwargs={"endpoint_url": endpoint_url}
        )

    except Exception as e:
        logger.error("Invalid cloud credentials. Check cloud credentials and try again: %s", str(e))
        return

    backup_file = "mongodb_backup.gz"
    backup_command = f'mongodump --uri="{mongo_connection_string}" --archive="{backup_file}" --gzip'
    run_system_command(backup_command)

    cs_instance.upload_file(backup_file, f"dump/archive/{backup_file}")

    logger.info("Successfully backed up MongoDB to S3")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Backup MongoDB data to S3")
    parser.add_argument("--access-key", help="AWS S3 access key to use for backup", default=None)
    parser.add_argument("--secret-key", help="AWS S3 secret key to use for backup", default=None)
    parser.add_argument("--region", help="AWS S3 region to use for backup", default=None)
    parser.add_argument("--s3-bucket-name", help="AWS S3 bucket to store backup data in", default=None)
    parser.add_argument("--endpoint-url", help="AWS S3 endpoint URL to use for backup", default=None)
    args = parser.parse_args()

    backup(args.access_key, args.secret_key, args.s3_bucket_name, args.endpoint_url, args.region)
