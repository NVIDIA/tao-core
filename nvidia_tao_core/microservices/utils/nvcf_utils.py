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

"""NVCF handlers modules"""
import os
import json
import time
import uuid
import requests
import traceback
import logging

from .ngc_utils import send_ngc_api_request, get_user_key
from .stateless_handler_utils import (
    update_job_details_with_microservices_response,
    get_job_specs,
    get_log_file_path,
    internal_job_status_update
)
from .handler_utils import get_cloud_metadata, get_num_nodes_from_spec
from .core_utils import retry_method, get_microservices_network_and_action


NUM_OF_RETRY = 3

# Configure logging
TAO_LOG_LEVEL = os.getenv('TAO_LOG_LEVEL', 'INFO').upper()
tao_log_level = getattr(logging, TAO_LOG_LEVEL, logging.INFO)
logging.basicConfig(
    level=logging.WARNING,  # Root logger: suppress third-party DEBUG logs
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logging.getLogger('nvidia_tao_core').setLevel(tao_log_level)
logger = logging.getLogger(__name__)


def get_available_nvcf_instances(user_id, org_name):
    """For the given org, format and return the NVCF cluster info"""
    ngc_key = get_user_key(user_id, org_name, admin_key_override=True)

    nvcf_info_endpoint = f"https://api.ngc.nvidia.com/v2/orgs/{org_name}/ngc/nvcf/deployments/instanceTypes"

    nvcf_info_response = send_ngc_api_request(
        endpoint=nvcf_info_endpoint,
        requests_method="GET",
        request_body={},
        ngc_key=ngc_key
    )
    available_nvcf_instances = {}
    if nvcf_info_response.ok:
        gpu_data = nvcf_info_response.json()
        for gpu_type, instances in gpu_data.items():
            for instance in instances:
                instance_name = instance['name']

                # Handle clusters - use clusterGroupName if clusters is empty
                clusters = instance.get('clusters', [])
                if not clusters and 'clusterGroupName' in instance:
                    clusters = [instance['clusterGroupName']]

                # If still no clusters, use 'default'
                if not clusters:
                    clusters = ['default']

                # Create separate entry for each cluster this instance is available in
                for cluster in clusters:
                    # Create a unique platform ID for this GPU type + cluster combination
                    platform_id = str(uuid.uuid5(uuid.NAMESPACE_X500, f"{cluster}_{gpu_type}_{instance_name}"))

                    # Extract availability information if present
                    max_instances = instance.get('maxInstances', 2)
                    current_instances = instance.get('currentInstances', 0)

                    # Calculate available instances if both values are present
                    if isinstance(max_instances, int) and isinstance(current_instances, int):
                        available = max_instances - current_instances
                    else:
                        available = 'N/A'

                    available_nvcf_instances[platform_id] = {
                        "cluster": cluster,
                        "gpu_type": gpu_type,
                        "instance_type": instance_name,
                        "gpu_count": instance['gpuCount'],
                        "cpu_cores": instance['cpuCores'],
                        "system_memory": instance['systemMemory'],
                        "gpu_memory": instance['gpuMemory'],
                        "regions": instance['regions'],
                        "max_limit": max_instances,
                        "current_used": current_instances,
                        "current_available": available,
                        "driver_version": instance.get('driverVersion', 'N/A'),
                        "storage": instance.get('storage', 'N/A')
                    }

    return available_nvcf_instances


# NVCF API wrapper functions


@retry_method(response=True)
def invoke_function(
    deployment_string,
    network="",
    action="",
    microservice_action="",
    cloud_metadata={},
    specs={},
    docker_env_vars={},
    kind="",
    handler_id="",
    job_id="",
    request_body={}
):
    """Invoke a NVCF function"""
    if not request_body:
        if not docker_env_vars.get("TAO_API_SERVER"):
            docker_env_vars["TAO_API_SERVER"] = "https://nvidia.com"
        if not docker_env_vars.get("TAO_LOGGING_SERVER_URL"):
            docker_env_vars["TAO_LOGGING_SERVER_URL"] = "https://nvidia.com"

    network, action = get_microservices_network_and_action(network, action)
    if action == "retrain":
        action = "train"

    request_metadata = {
        "api_endpoint": microservice_action,
        "request_body": {},
        "is_job": True,
        "kind": kind,
        "handler_id": handler_id,
        "job_id": job_id,
        "is_json_request": True
    }
    if not request_body and docker_env_vars:
        request_metadata["request_body"]["docker_env_vars"] = docker_env_vars
    if network:
        request_metadata["request_body"]["neural_network_name"] = network
    if action:
        request_metadata["request_body"]["action_name"] = action
    if specs:
        request_metadata["request_body"]["specs"] = specs
    if cloud_metadata:
        request_metadata["request_body"]["cloud_metadata"] = cloud_metadata
    if request_body:
        request_metadata["request_body"].update(request_body)

    if not request_body:
        if "docker_env_vars" not in request_metadata["request_body"]:
            request_metadata["request_body"]["docker_env_vars"] = {}
        request_metadata["request_body"]["docker_env_vars"]["CLOUD_BASED"] = "True"
        if os.getenv("HOST_PLATFORM", "local") == "NVCF":
            function_tao_api = os.getenv("FUNCTION_TAO_API", "")
            if not function_tao_api:
                raise ValueError("FUNCTION_TAO_API should be present for NVCF as host platform")

    num_nodes = get_num_nodes_from_spec(specs, action, network=network)
    if num_nodes > 1:
        request_metadata["request_body"]["statefulset_replicas"] = num_nodes

    function_id, version_id = deployment_string.split(":")

    url = f"https://api.nvcf.nvidia.com/v2/nvcf/pexec/functions/{function_id}/versions/{version_id}"
    headers = {
        'accept': 'application/json',
        'Content-Type': 'application/json',
        "Authorization": f"Bearer {docker_env_vars.get('TAO_USER_KEY')}",
    }

    try:
        response = requests.post(url, headers=headers, json=request_metadata, timeout=120)
    except Exception as e:
        logger.error("Exception caught during invoking NVCF function %s: %s", deployment_string, e)
        raise e

    if not response.ok:
        logger.error("Invocation failed.")
        logger.error("Response status code: %s", response.status_code)
        logger.error("Response content: %s", response.text)
    return response


@retry_method(response=True)
def get_status_of_invoked_function(request_id, ngc_key):
    """Fetch status of invoked function"""
    url = f"https://api.nvcf.nvidia.com/v2/nvcf/pexec/status/{request_id}"
    headers = {
        'accept': 'application/json',
        'Content-Type': 'application/json',
        "Authorization": f"Bearer {ngc_key}",
    }

    response = requests.get(url, headers=headers, timeout=120)

    if not response.ok:
        logger.error("Request failed.")
        logger.error("Response status code: %s", response.status_code)
        logger.error("Response content: %s", response.text)
    return response


def get_function(org_name, team_name, function_id, version_id, ngc_key):
    """Get function metadata"""
    team_string = f"teams/{team_name}/"
    if team_name in "no_team":
        team_string = ""
    endpoint = (
        f"https://api.ngc.nvidia.com/v2/orgs/{org_name}/"
        f"{team_string}nvcf/functions/{function_id}/versions/{version_id}"
    )
    requests_method = "GET"
    return send_ngc_api_request(endpoint, requests_method, request_body={}, json=False, ngc_key=ngc_key)


def create_function(org_name, team_name, job_id, container, ngc_key):
    """Create NVCF function"""
    helm_chart_service_name = f"tao-svc-{job_id}"
    nvcf_helm_chart = os.getenv(
        "NVCF_HELM_CHART",
        "https://helm.ngc.nvidia.com/ea-tlt/tao_ea/charts/tao-multi-node-6.0.7.tgz"
    )

    payload = {
        "name": job_id,
        "inferenceUrl": "/api/v1/orgs/ea-tlt/super_endpoint",
        "inferencePort": 8000,
        "apiBodyFormat": "CUSTOM",
        "helmChart": nvcf_helm_chart,
        "helmChartServiceName": helm_chart_service_name,
        "health": {
            "protocol": "HTTP",
            "uri": "/api/v1/health/readiness",
            "port": 8000,
            "timeout": "PT10S",
            "expectedStatusCode": 200
        },
    }

    team_string = f"teams/{team_name}/"
    if team_name in ["no_team"]:
        team_string = ""
    endpoint = (
        f"https://api.ngc.nvidia.com/v2/orgs/{org_name}/"
        f"{team_string}nvcf/functions"
    )
    requests_method = "POST"
    logger.debug("create endpoint %s %s", endpoint, payload)
    return send_ngc_api_request(endpoint, requests_method, request_body=json.dumps(payload), json=True, ngc_key=ngc_key)


def deploy_function(org_name, team_name, function_details, nvcf_backend_details, ngc_key, image=None, num_nodes=1):
    """Deploy NVCF function"""
    function_id = function_details["function"]["id"]
    version_id = function_details["function"]["versionId"]
    job_id = function_details["function"]["name"]
    helm_chart_service_name = f"tao-svc-{job_id}"
    statefulset_name = f"tao-sts-{job_id}"
    statefulset_service_name = f"tao-sts-svc-{job_id}"
    num_gpu_per_node = nvcf_backend_details.get("num_gpu_per_node", 1)
    instanceType = nvcf_backend_details['instance_type']
    nccl_ib_disable = os.getenv('NCCL_IB_DISABLE', default='0')
    nccl_ibext_disable = os.getenv('NCCL_IBEXT_DISABLE', default='0')
    payload = {
        "deploymentSpecifications": [
            {
                "gpu": nvcf_backend_details["gpu_type"],
                "backend": nvcf_backend_details["cluster"],
                "maxInstances": 1,
                "minInstances": 1,
                "instanceType": instanceType,
                "configuration": {
                    "image": image,
                    "numGpuPerNode": num_gpu_per_node,
                    "numNodes": num_nodes,
                    "jobId": job_id,
                    "helmChartServiceName": helm_chart_service_name,
                    "statefulSetName": statefulset_name,
                    "statefulSetServiceName": statefulset_service_name,
                    "ncclIbDisable": nccl_ib_disable,
                    "ncclIbExtDisable": nccl_ibext_disable,
                }
            }
        ]
    }

    team_string = f"teams/{team_name}/"
    if team_name in "no_team":
        team_string = ""
    endpoint = (
        f"https://api.ngc.nvidia.com/v2/orgs/{org_name}/"
        f"{team_string}nvcf/deployments/functions/{function_id}/versions/{version_id}"
    )
    requests_method = "POST"
    logger.debug("deploy endpoint %s %s", endpoint, payload)
    return send_ngc_api_request(endpoint, requests_method, request_body=json.dumps(payload), json=True, ngc_key=ngc_key)


def delete_function_version(org_name, team_name, function_id, version_id, ngc_key):
    """Un-deploy NVCF function"""
    team_string = f"teams/{team_name}/"
    if team_name in "no_team":
        team_string = ""
    endpoint = (
        f"https://api.ngc.nvidia.com/v2/orgs/{org_name}/"
        f"{team_string}nvcf/deployments/functions/{function_id}/versions/{version_id}"
    )
    requests_method = "DELETE"
    return send_ngc_api_request(endpoint, requests_method, request_body={}, json=False, ngc_key=ngc_key)


def add_authorized_party(org_name, team_name, function_id, version_id, authorized_party_nca_id, ngc_key):
    """Add an authorized party to a NVCF function

    Args:
        org_name (str): Organization name
        team_name (str): Team name
        function_id (str): Function ID
        version_id (str): Version ID
        authorized_party_nca_id (str): NCA ID of the authorized party to add
        ngc_key (str): NGC API key

    Returns:
        Response object from the API request
    """
    team_string = f"teams/{team_name}/"
    if team_name in "no_team":
        team_string = ""

    # First get current authorizations
    endpoint = (
        f"https://api.ngc.nvidia.com/v2/orgs/{org_name}/"
        f"{team_string}nvcf/authorizations/functions/{function_id}/versions/{version_id}"
    )
    get_response = send_ngc_api_request(endpoint, "GET", request_body={}, json=True, ngc_key=ngc_key)

    if not get_response.ok:
        return get_response

    # Add the new authorized party
    payload = {
        "authorizedParties": [{"ncaId": authorized_party_nca_id}]
    }

    return send_ngc_api_request(endpoint, "POST", request_body=json.dumps(payload), json=True, ngc_key=ngc_key)


def remove_authorized_party(org_name, team_name, function_id, version_id, authorized_party_nca_id, ngc_key):
    """Remove an authorized party from a NVCF function

    Args:
        org_name (str): Organization name
        team_name (str): Team name
        function_id (str): Function ID
        version_id (str): Version ID
        authorized_party_nca_id (str): NCA ID of the authorized party to remove
        ngc_key (str): NGC API key

    Returns:
        Response object from the API request
    """
    team_string = f"teams/{team_name}/"
    if team_name in "no_team":
        team_string = ""

    endpoint = (
        f"https://api.ngc.nvidia.com/v2/orgs/{org_name}/"
        f"{team_string}nvcf/authorizations/functions/{function_id}/versions/{version_id}/remove"
    )
    payload = {
        "authorizedParty": {"ncaId": authorized_party_nca_id}
    }

    return send_ngc_api_request(endpoint, "PATCH", request_body=json.dumps(payload), json=True, ngc_key=ngc_key)


# FTMS - NVCF interaction


def create_microservice_job_on_nvcf(job_metadata, docker_env_vars={}):
    """Create TAO microservice job on nvcf function"""
    nvcf_metadata = job_metadata.get("backend_details", {}).get("nvcf_metadata", {})
    network = job_metadata.get("network")
    action = job_metadata.get("action")
    tao_api_job_id = job_metadata.get("id")
    deployment_string = nvcf_metadata.get("deployment_string")
    ngc_key = docker_env_vars.get("TAO_USER_KEY")
    tao_api_status_callback_url = docker_env_vars.get("TAO_LOGGING_SERVER_URL", "")
    automl_experiment_number = docker_env_vars.get("AUTOML_EXPERIMENT_NUMBER", "0")

    job_message_job_id = tao_api_status_callback_url.split("/")[-1]

    cloud_metadata = {}
    get_cloud_metadata(nvcf_metadata.get("workspace_ids"), cloud_metadata)

    if job_message_job_id != tao_api_job_id:
        specs = get_job_specs(job_message_job_id, automl=True, automl_experiment_id=automl_experiment_number)
    else:
        specs = get_job_specs(tao_api_job_id)

    job_create_response = invoke_function(deployment_string,
                                          network,
                                          action,
                                          microservice_action="container_job_run",
                                          cloud_metadata=cloud_metadata,
                                          specs=specs,
                                          docker_env_vars=docker_env_vars,
                                          )

    if job_create_response.status_code not in [200, 202]:
        job_create_response_json = job_create_response.json()
        logger.error("Invocation error response code %s", job_create_response.status_code)
        logger.error("Invocation error response json %s", job_create_response_json)
        update_job_details_with_microservices_response(
            job_create_response_json.get('detail', ""),
            job_message_job_id,
            automl_expt_job_id=tao_api_job_id
        )
        logger.error(
            "Setting status of job %s to Error as microservices job couldn't be created",
            tao_api_job_id
        )
        return "Error", "Microservice job couldn't be created"

    job_create_response_json = job_create_response.json()
    logger.info("Microservice job successfully created for %s: %s", tao_api_job_id, job_create_response_json)
    job_id = job_create_response_json.get("job_id")

    if job_create_response.status_code == 202:
        req_id = job_create_response_json.get("reqId", "")
        while True:
            polling_response = get_status_of_invoked_function(req_id, ngc_key)
            if polling_response.status_code == 404:
                if polling_response.json().get("title") != "Not Found":
                    logger.error("Polling(job_create) response failed %s", polling_response.status_code)
                    logger.error(
                        "Setting status of job %s to Error as job create polling "
                        "failed with a non 200 response",
                        job_id
                    )
                    return "Error", "NVCF Polling failed"
            if polling_response.status_code != 202:
                break
            time.sleep(10)

        if polling_response.status_code != 200:
            logger.error(
                "Polling(job_create) response status code is not 200 %s",
                polling_response.status_code
            )
            logger.error(
                "Setting status of job %s to Error as job create polling "
                "failed with a non 200 response",
                job_id
            )
            return "Error", "NVCF Polling failed"
        job_id = polling_response.json().get("job_id")

    if not job_id:
        logger.error("Job ID couldn't be fetched")
        logger.error("Setting status of job %s to Error as job id can't be fetched from microservices", job_id)
        return "Error", "Job_id from microservices job created couldn't be fetched"

    return "Running", "Job submitted to NVCF"


def get_nvcf_microservices_job_status(job_metadata, status="", docker_env_vars={}):
    """Get and update NVCF custom resource status"""
    nvcf_metadata = job_metadata.get("backend_details", {}).get("nvcf_metadata", {})
    if not status:
        user_id = job_metadata.get("user_id")
        org_name = job_metadata.get("org_name")
        action = job_metadata.get("action")
        specs = job_metadata.get("specs", {})
        network = job_metadata.get("network")
        job_id = job_metadata.get("id")
        job_handler_id = job_metadata.get("handler_id")
        job_status = job_metadata.get("status")
        ngc_key = docker_env_vars.get("TAO_USER_KEY")
        automl_experiment_number = docker_env_vars.get("AUTOML_EXPERIMENT_NUMBER", "0")
        tao_api_status_callback_url = docker_env_vars.get("TAO_LOGGING_SERVER_URL", "")

        job_message_job_id = tao_api_status_callback_url.split("/")[-1]

        deployment_string = nvcf_metadata.get("deployment_string")
        if deployment_string.find(":") == -1:
            if job_status == "Error":
                return "Error"
            logger.debug(
                "Deployment not active yet for job %s %s (in get status function)",
                job_id, deployment_string
            )
            status = "Pending"
            return status

        if job_status in ("Done", "Error"):
            return job_status

        job_monitor_response = invoke_function(
            deployment_string,
            network,
            action,
            microservice_action="container_job_status",
            specs=specs,
            docker_env_vars=docker_env_vars
        )
        if job_monitor_response.status_code == 404:
            status = "Error"
            if job_monitor_response.json().get("title") == "Not Found":
                logger.info("NVCF function was deleted, setting status as done")
                status = "Done"

        if job_monitor_response.status_code == 202:
            req_id = job_monitor_response.json().get("reqId", "")
            while True:
                job_monitor_response = get_status_of_invoked_function(req_id, ngc_key)
                if job_monitor_response.status_code == 404:
                    if job_monitor_response.json().get("title") != "Not Found":
                        logger.error("Polling(job_monitor) response failed %s", job_monitor_response.status_code)
                        status = "Error"
                if job_monitor_response.status_code != 202:
                    break
                time.sleep(10)

            if job_monitor_response.status_code != 200:
                logger.error(
                    "Polling(job_monitor) response status code is not 200 %s",
                    job_monitor_response.status_code
                )
                status = "Error"

        if not status:
            try:
                job_monitor_response_json = job_monitor_response.json()
                error_message = job_monitor_response_json.get("detail")
                status = job_monitor_response_json.get("status")
                if status:
                    if status not in ("Pending", "Done", "Running"):
                        logfile = get_log_file_path(
                            user_id,
                            org_name,
                            job_handler_id,
                            job_message_job_id,
                            job_id,
                            automl_experiment_number
                        )
                        internal_job_status_update(
                            job_message_job_id,
                            automl=False,
                            automl_experiment_number=automl_experiment_number,
                            message="Container microservices reported an error, more logs to be found on NVCF UI",
                            logfile=logfile
                        )
                        status = "Error"
                else:
                    status = "Pending"
                    if "Job ID Not Present" in error_message:
                        logger.error("Job ID Not Present in %s for job %s", deployment_string, job_id)
                        status = "Error"
            except Exception as e:
                logger.error("Exception thrown in get_nvcf_microservices_job_status: %s", str(e))
                logger.error(traceback.format_exc())
                logger.error(
                    "Exception while calling job fetch microservices in %s for job %s, %s",
                    deployment_string, job_id, job_monitor_response.text
                )
                status = "Error"

    if not status:
        logger.error("Status couldn't be inferred")
        status = "Pending"
    return status
