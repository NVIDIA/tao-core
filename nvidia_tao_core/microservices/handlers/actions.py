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

"""Pipeline construction for all experiment actions"""
import copy
import datetime
import json
import os
import threading
import time
import traceback
import uuid
import logging

from nvidia_tao_core.microservices.automl.utils import delete_lingering_checkpoints, wait_for_job_completion
from nvidia_tao_core.microservices.constants import (
    _DATA_GENERATE_ACTIONS,
    _DATA_SERVICES_ACTIONS,
    MONAI_NETWORKS,
    MEDICAL_AUTOML_ARCHITECT,
    MEDICAL_NETWORK_ARCHITECT,
    MEDICAL_CUSTOM_ARCHITECT,
    NETWORK_CONTAINER_MAPPING,
    COPY_MODEL_PARAMS_FROM_TRAIN_NETWORKS
)
from nvidia_tao_core.microservices.handlers.cloud_storage import create_cs_instance
from nvidia_tao_core.microservices.handlers.ngc_handler import get_user_key
from nvidia_tao_core.microservices.handlers.nvcf_handler import get_available_nvcf_instances
from nvidia_tao_core.microservices.handlers.docker_images import DOCKER_IMAGE_MAPPER, DOCKER_IMAGE_VERSION
from nvidia_tao_core.microservices.handlers.infer_data_sources import apply_data_source_config
from nvidia_tao_core.microservices.handlers.infer_params import CLI_CONFIG_TO_FUNCTIONS
from nvidia_tao_core.microservices.handlers.encrypt import NVVaultEncryption
from nvidia_tao_core.microservices.handlers.monai.helpers import (
    CUSTOMIZED_BUNDLE_URL_FILE,
    CUSTOMIZED_BUNDLE_URL_KEY,
    MEDICAL_SERVICE_SCRIPTS
)
from nvidia_tao_core.microservices.handlers.stateless_handlers import (
    BACKEND,
    base_exp_uuid,
    get_base_experiment_metadata,
    get_handler_job_metadata,
    get_handler_log_root,
    get_handler_metadata,
    get_jobs_root,
    get_handler_root,
    get_toolkit_status,
    resolve_metadata,
    get_job_specs,
    save_job_specs,
    get_automl_brain_info,
    get_automl_best_rec_info,
    get_automl_controller_info,
    save_automl_controller_info,
    get_dnn_status,
    update_job_metadata,
    update_job_status,
    write_handler_metadata,
    update_job_details_with_microservices_response,
    get_user_telemetry_opt_out
)
from nvidia_tao_core.microservices.handlers.utilities import (
    StatusParser,
    build_cli_command,
    generate_cl_script,
    get_num_nodes_from_spec,
    get_total_epochs,
    read_nested_dict,
    search_for_base_experiment,
    get_num_gpus_from_spec,
    validate_monai_bundle_params,
    write_nested_dict,
    get_cloud_metadata
)
from nvidia_tao_core.microservices.utils import (
    remove_key_by_flattened_string,
    read_network_config,
    get_admin_key,
    safe_load_file,
    find_differences,
    merge_nested_dicts,
    get_monitoring_metric,
    get_microservices_network_and_action
)
from nvidia_tao_core.microservices.job_utils import executor as jobDriver
from nvidia_tao_core.microservices.network_utils.network_constants import ptm_mapper
from nvidia_tao_core.microservices.specs_utils import json_to_kitti, json_to_yaml, json_to_toml

SPEC_BACKEND_TO_FUNCTIONS = {
    "protobuf": json_to_kitti.kitti,
    "yaml": json_to_yaml.yml,
    "toml": json_to_toml.toml_format
}
HOST_PLATFORM = os.getenv("HOST_PLATFORM", "local-k8s")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ActionPipeline:
    """ActionPipeline - Train, Evaluate, Retrain, Prune, Export, Gen_trt_engine (Model),

    To spawn a job by handling all dependencies, monitor and close a job end-to-end
    - Inputs:
        - JobContext: To communicate with the Model / Dataset handler
        - Requires Handler & AppHandler to run()
    - Processes spec requirements
        - Prepares specs (generate_specs step)
            - dataset config (defined for each network's train, evaluate, retrain)
            - base_experiment config (for train, evaluate)
            - parent model and load graph (for retrain)
            - CLI paramters (for all actions)
            - Classwise configs (for all applicable train, evaluate, retrain) => currently for OD networks only
            - Converts json to spec backend

        - Prepare command (generate_run_command)
            - Generate run command
        - Infers image from config.json and platform information (if applicable) and sends it to K8s
        - Interacts with status callbacks from DNN container (ETA: TBD) and communicated to Handlers through JobContext
        - Supports delete job
        - Supports resume for train
    - Exposed functions:
        - run():
        - delete():
        - resume(): Same as run()
    - Internal functions:
        - parse_status():
        - generate_config(): Assumes <action>.json exists
        - generate_run_command():
    - Helper functions():
        - __init__():
        - _read_api_params()
    """

    def __init__(self, job_context):
        """Initialize the ActionPipeline class"""
        # Job Context - bridge between Action and Handler
        self.job_context = job_context
        # Get some handler related data
        self.network = self.job_context.network
        self.action = self.job_context.action
        self.network_config = read_network_config(self.network)
        self.api_params = self._read_api_params()
        self.handler_kind = self.job_context.kind
        self.handler_metadata = get_handler_metadata(self.job_context.handler_id, self.handler_kind)
        self.workspace_id = self.handler_metadata.get("workspace")
        self.workspace_metadata = get_handler_metadata(self.workspace_id, "workspaces")
        self.handler_root = get_handler_root(self.job_context.org_name, None, self.job_context.handler_id, None)
        self.jobs_root = get_jobs_root(self.job_context.user_id, self.job_context.org_name)
        self.handler_log_root = get_handler_log_root(
            self.job_context.user_id,
            self.job_context.org_name,
            self.job_context.handler_id
        )
        self.handler_id = self.job_context.handler_id
        self.tao_deploy_actions = False
        self.parent_job_action = get_handler_job_metadata(self.job_context.parent_id).get("action")
        if (self.job_context.action in ("gen_trt_engine", "trtexec") or
           self.parent_job_action in ("gen_trt_engine", "trtexec")):
            self.tao_deploy_actions = True
        self.image = DOCKER_IMAGE_MAPPER[self.api_params.get("image", "")]
        if self.job_context.action in _DATA_SERVICES_ACTIONS and not self.network.startswith("monai"):
            self.image = DOCKER_IMAGE_MAPPER["TAO_DS"]
        # If current or parent action is gen_trt_engine or trtexec, then it'a a tao-deploy container action
        # Override version of image specific for networks
        if self.tao_deploy_actions:
            team = "TAO"
            if "maxine" in self.network:
                team = "MAXINE"
            self.image = DOCKER_IMAGE_MAPPER[f"{team}_DEPLOY"]
        using_previous_version = False
        if self.network in DOCKER_IMAGE_VERSION.keys():
            self.tao_framework_version, self.tao_model_override_version = DOCKER_IMAGE_VERSION[self.network]
            if self.tao_model_override_version not in self.image:
                image = self.image.replace(self.tao_framework_version, self.tao_model_override_version)
                if image != self.image:
                    self.image = image
                    using_previous_version = True
        if self.action in self.network_config.get("api_params", {}).get("image_override_per_action", {}):
            image_override_per_action = self.api_params.get("image_override_per_action", {})
            override_key = image_override_per_action.get(self.action)
            if override_key:
                self.image = DOCKER_IMAGE_MAPPER[override_key]
        # This will be run inside a thread
        self.thread = None
        # if self.network == "maxine_eye_contact":
        # if self.action == "auto_labeling":

        # Parameters to launch a job and monitor status
        self.job_name = str(self.job_context.id)

        self.spec = {}
        self.config = {}
        self.job_env_variables = {
            "ORCHESTRATION_API_NETWORK": self.network,
            "ORCHESTRATION_API_ACTION": self.action
        }
        if using_previous_version:
            self.job_env_variables = {}
        self.platform_id = self.job_context.platform_id
        if not self.platform_id:
            if BACKEND == "NVCF":
                self.platform_id = "052fc221-ffaa-5c15-8d22-b663e7339349"

        self.run_command = ""
        self.logfile = os.path.join(self.handler_log_root, str(self.job_context.id) + ".txt")
        self.cloud_metadata = {}
        self.cs_instance = None  # initialized in run()
        self.ngc_runner = False
        if BACKEND == "NVCF":
            self.ngc_runner = True
        self.local_cluster = False
        self.monai_env_variable = {}
        self.num_gpu = (
            self.job_context.specs.get("num_gpu", self.job_context.num_gpu)
            if self.job_context.specs else self.job_context.num_gpu
        )
        self.num_nodes = get_num_nodes_from_spec(self.job_context.specs, self.action, network=self.network)
        self.recursive_dataset_file_download = self.api_params.get("recursive_dataset_file_download", False)
        # add an entry on the docker image mapper for trt engine generation MAXINE DEPLOY
        # if action is trt engine generation and network is a maxine network, override image from docker image mapper
        # TODO: robbie add image mpping fix for trt engine gen

    def _read_api_params(self):
        """Read network config json file and return api_params key"""
        return self.network_config.get("api_params", {})

    def generate_config(self):
        """Generate config for this action; Actions may override"""
        return {}, {}

    def generate_run_command(self):
        """Generate run command for this action; Actions may override"""
        return "", None

    def post_run(self):
        """Run & modify internal variables after toolkit job is done; Actions may override"""
        return

    def decrypt_docker_env_vars(self, docker_env_vars):
        """Decrypt NvVault encrypted values"""
        config_path = os.getenv("VAULT_SECRET_PATH", None)
        if config_path:
            encryption = NVVaultEncryption(config_path)
            for docker_env_var_key, docker_env_var_value in docker_env_vars.items():
                if encryption.check_config()[0]:
                    docker_env_vars[docker_env_var_key] = encryption.decrypt(docker_env_var_value)

    def generate_env_variables(self, automl_brain_job_id=None, experiment_number=None, automl_exp_job_id=None):
        """Generate env variables required for a job"""
        host_base_url = os.getenv("HOSTBASEURL", "no_url")
        if HOST_PLATFORM == "NVCF":
            function_version_string = os.getenv("FUNCTION_TAO_API")
            if not function_version_string:
                raise ValueError(
                    "For HOST Platform NVCF, FUNCTION_TAO_API should be present in chart values "
                    "in the form of function_id:version_id"
                )
            if BACKEND == "local-k8s":
                raise ValueError("For HOST Platform NVCF, Backend should also be NVCF")
            self.job_env_variables["NVCF_HELM"] = function_version_string
            host_base_url = "http://10.123.4.56:32080"  # Will not be used by DNN containers, just to match a URL format
        log_callback_job_id = self.job_context.id
        if automl_exp_job_id:
            log_callback_job_id = automl_exp_job_id

        org_name = self.job_context.org_name
        if BACKEND == "local-k8s":
            cluster_ip, cluster_port = jobDriver.get_cluster_ip()
            if cluster_ip and cluster_port:
                host_base_url = f"http://{cluster_ip}:{cluster_port}"

        handler_kind = "experiments"
        if (not self.handler_metadata.get("train_datasets", [])) and self.job_context.network not in (
            ["auto_label", "image"] + MEDICAL_AUTOML_ARCHITECT + MEDICAL_NETWORK_ARCHITECT
        ):
            handler_kind = "datasets"

        status_url = (
            f"{host_base_url}/api/v1/orgs/{org_name}/{handler_kind}/"
            f"{self.handler_id}/jobs/{self.job_context.id}"
        )
        if automl_brain_job_id:
            status_url = (
                f"{host_base_url}/api/v1/orgs/{org_name}/{handler_kind}/"
                f"{self.handler_id}/jobs/{automl_brain_job_id}"
            )
            if experiment_number:
                self.job_env_variables["AUTOML_EXPERIMENT_NUMBER"] = experiment_number

        self.job_env_variables["TELEMETRY_OPT_OUT"] = get_user_telemetry_opt_out(
            self.job_context.user_id,
            self.job_context.org_name
        )
        self.job_env_variables["CLOUD_BASED"] = "True"
        user_key = get_user_key(
            self.job_context.user_id,
            self.job_context.org_name,
            admin_key_override=True
        )
        self.job_env_variables["TAO_USER_KEY"] = user_key
        self.job_env_variables["RECURSIVE_DATASET_FILE_DOWNLOAD"] = str(self.recursive_dataset_file_download)
        self.job_env_variables["TAO_ADMIN_KEY"] = get_admin_key()
        self.job_env_variables["TAO_API_KEY"] = get_admin_key(legacy_key=True)
        self.job_env_variables["TAO_API_SERVER"] = host_base_url
        self.job_env_variables["TAO_API_JOB_ID"] = log_callback_job_id
        self.job_env_variables["TAO_LOGGING_SERVER_URL"] = status_url

    def generate_nv_job_metadata(self, nv_job_metadata):
        """Convert run command generated into format that"""
        nv_job_metadata["teamName"] = os.getenv("NVCF_DEPLOYMENT_TEAM_NAME", "no_team")
        nv_job_metadata["dockerImageName"] = self.image
        if BACKEND == "NVCF":
            nv_job_metadata["workspace_ids"] = list(self.workspace_ids)
            nv_job_metadata["deployment_string"] = os.getenv(f'FUNCTION_{NETWORK_CONTAINER_MAPPING[self.network]}')

            available_nvcf_instances = get_available_nvcf_instances(self.job_context.user_id, self.job_context.org_name)
            # if not available_nvcf_instances:
            available_nvcf_instances["052fc221-ffaa-5c15-8d22-b663e7339349"] = {
                "cluster": "GFN",
                "gpu_type": "L40S",
                "instance_type": "gl40s_1x2.br25_4xlarge"
            }
            instance_type = available_nvcf_instances[self.platform_id]["instance_type"]
            nv_job_metadata["nvcf_backend_details"] = {
                "cluster": available_nvcf_instances[self.platform_id]["cluster"],
                "gpu_type": available_nvcf_instances[self.platform_id]["gpu_type"],
                "instance_type": instance_type,
                "current_available": available_nvcf_instances[self.platform_id]["current_available"]
            }
            for gpu_postfix in ["2x", "4x", "8x"]:
                if gpu_postfix in instance_type:
                    dividing_factor = 1
                    if available_nvcf_instances[self.platform_id]["cluster"] == "GFN":
                        dividing_factor = 2
                    nv_job_metadata["nvcf_backend_details"]["num_gpu_per_node"] = int(
                        int(gpu_postfix[:-1]) / dividing_factor
                    )
                    break

            if self.tao_deploy_actions:
                team = "TAO"
                if "maxine" in self.network:
                    team = "MAXINE"
                nv_job_metadata["deployment_string"] = os.getenv(f'FUNCTION_{team}_DEPLOY')
            nv_job_metadata["network"] = self.network
            for key, value in self.job_env_variables.items():
                nv_job_metadata[key] = value

    def get_handler_cloud_details(self):
        """Gather cloud details from various handlers associated for the job"""
        self.workspace_ids = []
        workspace_cache = {}

        def process_metadata(data_type, dataset_id=None, metadata=None, workspace_cache={}):
            """Process metadata for datasets, workspaces, etc."""
            if not metadata:
                metadata = resolve_metadata(data_type, dataset_id)
            else:
                metadata = copy.deepcopy(metadata)
            workspace_id = metadata.get("workspace", "")
            if not workspace_id:
                return
            self.workspace_ids.append(workspace_id)

        if self.handler_metadata.get("train_datasets", []):
            for train_ds in self.handler_metadata.get("train_datasets", []):
                process_metadata("dataset", dataset_id=train_ds, workspace_cache={})
        elif (self.job_context.network not in ["auto_label", "image"] +
              MEDICAL_AUTOML_ARCHITECT + MEDICAL_NETWORK_ARCHITECT):
            process_metadata("dataset", metadata=self.handler_metadata, workspace_cache=workspace_cache)

        eval_ds = self.handler_metadata.get("eval_dataset", None)
        if eval_ds:
            process_metadata("dataset", dataset_id=eval_ds, workspace_cache=workspace_cache)

        infer_ds = self.handler_metadata.get("inference_dataset", None)
        if infer_ds:
            process_metadata("dataset", dataset_id=infer_ds, workspace_cache=workspace_cache)

        experiment_metadata = copy.deepcopy(self.handler_metadata)
        exp_workspace_id = experiment_metadata.get("workspace")
        self.workspace_ids.append(exp_workspace_id)
        if self.network in MONAI_NETWORKS or BACKEND in ("local-k8s", "local-docker"):
            get_cloud_metadata(self.workspace_ids, self.cloud_metadata)

    def handle_ptm_anomalies(self):
        """Remove one of end-end or backbone related PTM field based on the Handler metadata info"""
        for base_experiment_id in self.handler_metadata.get("base_experiment", []):
            base_experiment_metadata = get_base_experiment_metadata(base_experiment_id)
            if base_experiment_metadata.get("base_experiment_metadata", {}).get("is_backbone"):
                # if ptm is a backbone remove end_to_end field from config and spec
                parameter_to_remove = ptm_mapper.get("end_to_end", {}).get(base_experiment_metadata.get("network_arch"))
            else:
                # if ptm is not a backbone remove it field from config and spec
                parameter_to_remove = ptm_mapper.get("backbone", {}).get(base_experiment_metadata.get("network_arch"))
            if parameter_to_remove:
                remove_key_by_flattened_string(self.spec, parameter_to_remove)
                remove_key_by_flattened_string(self.config, parameter_to_remove)

        if not self.handler_metadata.get("base_experiment", []):
            parameters_to_remove = [
                ptm_mapper.get("end_to_end", {}).get(self.network),
                ptm_mapper.get("backbone", {}).get(self.network),
                ptm_mapper.get("default", {}).get(self.network),
            ]
            for parameter_to_remove in parameters_to_remove:
                if parameter_to_remove:
                    remove_key_by_flattened_string(self.spec, parameter_to_remove)
                    remove_key_by_flattened_string(self.config, parameter_to_remove)

    def detailed_print(self, *args, **kwargs):
        """Print with job context"""
        # Remove 'file' from kwargs since logger.info() doesn't accept it
        if 'file' in kwargs:
            del kwargs['file']
        # Join all args and kwargs into a single string message
        message = ' '.join(str(arg) for arg in args)
        kwargs_str = ' '.join(f'{k}={v}' for k, v in kwargs.items())
        if kwargs_str:
            message = f'{message} {kwargs_str}'
        logger.info(message)

    def create_microservice_action_job(self, job_id):
        """Call executor function to create microservice pod and then invoke it"""
        logger.info("Creating microservices job_action ms pod")
        response = jobDriver.create_microservice_and_send_request(api_endpoint="post_action",
                                                                  network=self.network,
                                                                  action=self.action,
                                                                  cloud_metadata=self.cloud_metadata,
                                                                  specs=self.spec,
                                                                  microservice_pod_id=self.job_name,
                                                                  num_gpu=self.num_gpu,
                                                                  microservice_container=self.image,
                                                                  org_name=self.job_context.org_name,
                                                                  handler_id=self.handler_id,
                                                                  handler_kind=self.handler_kind,
                                                                  accelerator=self.platform_id,
                                                                  docker_env_vars=self.job_env_variables,
                                                                  num_nodes=self.num_nodes
                                                                  )
        if response and not response.ok:
            update_job_details_with_microservices_response(response.json().get("error", ""), job_id, self.job_name)

    def monitor_job(self):
        """Monitors the job status and updates job metadata"""
        if self.network in MONAI_NETWORKS:
            _, _, outdir = CLI_CONFIG_TO_FUNCTIONS["monai_output_dir"](self.job_context, self.handler_metadata)
            outdir = outdir.rstrip(os.path.sep)
        else:
            _, outdir = self.generate_run_command()
        if not outdir:
            outdir = CLI_CONFIG_TO_FUNCTIONS["output_dir"](self.job_context, self.handler_metadata)

        status_parser = StatusParser(self.job_context.network, outdir)

        total_epochs = 1
        if self.job_context.action in ['train', 'distill', 'retrain', 'quantize']:
            total_epochs = get_total_epochs(self.job_context, self.job_context.specs)

        metric = self.handler_metadata.get("metric", "")
        if not metric:
            metric = get_monitoring_metric(self.network)

        k8s_status = jobDriver.status(
            self.job_context.org_name,
            self.handler_id,
            self.job_name,
            self.handler_kind,
            use_ngc=self.ngc_runner,
            network=self.network,
            action=self.action,
            automl_exp_job=False,
            docker_env_vars=self.job_env_variables
        )

        # Delete job if is canceled/paused during pod creation
        metadata_status = get_handler_job_metadata(self.job_name).get("status", "Error")
        if metadata_status in ("Canceling", "Canceled", "Pausing", "Paused"):
            self.detailed_print(f"Terminating job {self.job_name}")
            jobDriver.delete(self.job_name, use_ngc=self.ngc_runner)

        # Monitor job status
        while k8s_status in ["Done", "Error", "Running", "Pending"]:
            # If Done, try running self.post_run()
            # Poll every 30 seconds
            time.sleep(30)

            metadata_status = get_handler_job_metadata(self.job_name).get("status", "Error")
            if metadata_status in ("Canceled", "Paused") and k8s_status == "Running":
                self.detailed_print(f"Terminating job {self.job_name}")
                jobDriver.delete(self.job_name, use_ngc=self.ngc_runner)
            if k8s_status == "Done":
                update_job_status(self.handler_id, self.job_name, status="Running", kind=self.handler_kind)
                # Retrieve status one last time!
                new_results = status_parser.update_results(total_epochs=total_epochs, job_id=self.job_name)
                update_job_metadata(
                    self.handler_id,
                    self.job_name,
                    metadata_key="job_details",
                    data=new_results,
                    kind=self.handler_kind
                )
                try:
                    self.detailed_print("Post running")
                    # If post run is done, make it done
                    self.post_run()
                    if self.job_context.action in ['train', 'distill', 'retrain', 'quantize']:
                        _, best_checkpoint_epoch_number, latest_checkpoint_epoch_number = status_parser.read_metric(
                            results=new_results[self.job_name],
                            metric=metric,
                            brain_epoch_number=total_epochs
                        )
                        self.handler_metadata["checkpoint_epoch_number"][
                            f"best_model_{self.job_name}"
                        ] = best_checkpoint_epoch_number
                        self.handler_metadata["checkpoint_epoch_number"][
                            f"latest_model_{self.job_name}"
                        ] = latest_checkpoint_epoch_number
                        write_handler_metadata(self.handler_id, self.handler_metadata, self.handler_kind)
                    if not os.path.exists(f"{self.jobs_root}/{self.job_name}"):
                        os.makedirs(f"{self.jobs_root}/{self.job_name}")
                    update_job_status(self.handler_id, self.job_name, status="Done", kind=self.handler_kind)
                    break
                except Exception as e:
                    # If post run fails, call it Error
                    logger.error("Exception thrown in post run after done status %s", str(e))
                    self.detailed_print(traceback.format_exc())
                    update_job_status(self.handler_id, self.job_name, status="Error", kind=self.handler_kind)
                    break
            # If running in K8s, update results to job_context
            elif k8s_status == "Running":
                update_job_status(self.handler_id, self.job_name, status="Running", kind=self.handler_kind)
                # Update results
                new_results = status_parser.update_results(total_epochs=total_epochs, job_id=self.job_name)
                update_job_metadata(
                    self.handler_id,
                    self.job_name,
                    metadata_key="job_details",
                    data=new_results,
                    kind=self.handler_kind
                )

            # Pending is if we have queueing systems down the road
            elif k8s_status == "Pending":
                k8s_status = jobDriver.status(
                    self.job_context.org_name,
                    self.handler_id,
                    self.job_name,
                    self.handler_kind,
                    use_ngc=self.ngc_runner,
                    network=self.network,
                    action=self.action,
                    automl_exp_job=False,
                    docker_env_vars=self.job_env_variables
                )
                continue

            # If the job never submitted or errored out!
            else:
                new_results = status_parser.update_results(total_epochs=total_epochs, job_id=self.job_name)
                update_job_metadata(
                    self.handler_id,
                    self.job_name,
                    metadata_key="job_details",
                    data=new_results,
                    kind=self.handler_kind
                )
                update_job_status(self.handler_id, self.job_name, status="Error", kind=self.handler_kind)
                break
            k8s_status = jobDriver.status(
                self.job_context.org_name,
                self.handler_id,
                self.job_name,
                self.handler_kind,
                use_ngc=self.ngc_runner,
                network=self.network,
                action=self.action,
                automl_exp_job=False,
                docker_env_vars=self.job_env_variables
            )

        metadata_status = get_handler_job_metadata(self.job_name).get("status", "Error")

        toolkit_status = get_toolkit_status(self.job_name)
        self.detailed_print(f"Toolkit status for {self.job_name} is {toolkit_status}")
        if (metadata_status not in ("Canceled", "Canceling", "Paused", "Pausing") and
                toolkit_status != "SUCCESS" and
                self.job_context.action != "trtexec"):
            update_job_status(self.handler_id, self.job_name, status="Error", kind=self.handler_kind)
            metadata_status = "Error"

        self.detailed_print(f"Job Done: {self.job_name} Final status: {metadata_status}")
        if self.ngc_runner or (self.network not in MONAI_NETWORKS and BACKEND in ("local-k8s", "local-docker")):
            if metadata_status not in ("Canceled", "Canceling", "Paused", "Pausing"):
                jobDriver.delete(self.job_name)

    def run(self):
        """Calls necessary setup functions and calls job creation"""
        # Set up
        self.thread = threading.current_thread()
        try:
            # Generate config
            self.spec, self.config = self.generate_config()
            self.cs_instance, _ = create_cs_instance(self.workspace_metadata)
            self.handle_ptm_anomalies()
            # Populate the cloud metadata for the job
            self.get_handler_cloud_details()
            # Generate run command
            self.run_command, outdir = self.generate_run_command()
            if self.network not in MONAI_NETWORKS and self.spec:
                self.num_gpu = get_num_gpus_from_spec(
                    self.spec, self.job_context.action, network=self.network, default=self.num_gpu
                )
                self.num_nodes = get_num_nodes_from_spec(
                    self.spec,
                    self.job_context.action,
                    network=self.network,
                    default=self.num_nodes)
                self.detailed_print(f"Job {self.job_name} running with {self.num_gpu} GPUs and {self.num_nodes} nodes")
            if not outdir:
                outdir = f"/results/{self.job_name}"
            # Pipe stdout and stderr to logfile
            self.run_command += f" 2>&1 | tee /{self.job_name}.txt"
            # After command runs, make sure subdirs permission allows anyone to enter and delete
            self.run_command += f"; find {outdir} -type d | xargs chmod 777"
            # After command runs, make sure artifact files permission allows anyone to delete
            if self.local_cluster:
                outdir = os.path.normpath(outdir) + os.sep  # remove double trailing slashes for filepath
            self.run_command += f"; find {outdir} -type f | xargs chmod 666"
            # Optionally, pipe self.run_command into a log file
            self.detailed_print(self.run_command)
            self.detailed_print(self.image)

            nv_job_metadata = {}
            # Convert self.spec to a backend and post it into a <self.handler_spec_root><job_id>.txt file
            if self.spec:
                file_type = self.api_params["spec_backend"]
                if file_type == "json":
                    kitti_out = self.spec
                    kitti_out = json.dumps(kitti_out)
                else:
                    kitti_out = SPEC_BACKEND_TO_FUNCTIONS[file_type](self.spec)
                # Save specs to DB
                save_job_specs(self.job_name, kitti_out)

            # Submit to K8s
            # Platform is None, but might be updated in self.generate_config() or self.generate_run_command()
            # If platform is indeed None, jobDriver.create would take care of it.
            docker_env_vars = self.handler_metadata.get("docker_env_vars", {})
            self.decrypt_docker_env_vars(docker_env_vars)
            self.job_env_variables.update(copy.deepcopy(docker_env_vars))
            # Add environment variables from monai.
            self.generate_env_variables()
            if self.monai_env_variable:
                self.job_env_variables.update(self.monai_env_variable)

            # The monai local jobs like training for cl jobs are designed to run on cluster local GPUs.
            if self.ngc_runner:
                self.generate_nv_job_metadata(nv_job_metadata)
            else:
                nv_job_metadata = None

            if self.network not in MONAI_NETWORKS and BACKEND in ("local-k8s", "local-docker"):
                self.create_microservice_action_job(self.job_name)
            else:
                jobDriver.create(
                    self.job_context.org_name,
                    self.job_name,
                    self.image,
                    self.run_command,
                    num_gpu=self.num_gpu,
                    num_nodes=self.num_nodes,
                    accelerator=self.platform_id,
                    docker_env_vars=self.job_env_variables,
                    nv_job_metadata=nv_job_metadata,
                    local_cluster=self.local_cluster,
                    automl_exp_job=False,
                )
            self.detailed_print("Job created", self.job_name)
            self.monitor_job()
            return

        except Exception as e:
            # Something went wrong inside...
            self.detailed_print(traceback.format_exc())
            self.detailed_print(f"Job {self.job_name} did not start")
            update_job_status(self.handler_id, self.job_name, status="Error", kind=self.handler_kind)
            result_dict = {
                self.job_name: {
                    "detailed_status": {
                        "message": "Error due to unmet dependencies",
                        "status": "FAILURE"
                    }
                }
            }
            if isinstance(e, ValueError):
                result_dict = {self.job_name: {"detailed_status": {"message": str(e), "status": "FAILURE"}}}
            if isinstance(e, TimeoutError):
                result_dict = {
                    self.job_name: {
                        "detailed_status": {
                            "message": "Data downloading from cloud storage failed.",
                            "status": "FAILURE"
                        }
                    }
                }
            update_job_metadata(
                self.handler_id,
                self.job_name,
                metadata_key="job_details",
                data=result_dict,
                kind=self.handler_kind
            )
            return


class CLIPipeline(ActionPipeline):
    """CLIPipeline for actions involve only cli params"""

    def __init__(self, job_context):
        """Initialize the CLIPipeline class"""
        super().__init__(job_context)

        self.network = job_context.network
        self.action = job_context.action

        # Handle anomalies in network action names
        if self.action == "retrain":
            self.action = "train"

        # Use the centralized function to map network and action
        self.network, self.action = get_microservices_network_and_action(self.network, self.action)

    def generate_config(self):
        """Generate config dictionary"""
        # Get some variables
        action = self.job_context.action
        # User stored CLI param in a json file
        config = get_job_specs(self.job_context.id)
        network = self.job_context.network
        # Get CLI params from config json
        network_config = read_network_config(network)
        if action in network_config.get("cli_params", {}).keys():
            for field_name, inference_fn in network_config["cli_params"][action].items():
                if inference_fn in CLI_CONFIG_TO_FUNCTIONS:
                    field_value = CLI_CONFIG_TO_FUNCTIONS[inference_fn](self.job_context, self.handler_metadata)
                    if field_value:
                        config[field_name] = field_value
        return {}, config

    def generate_run_command(self):
        """Generate run command"""
        if self.action == "dataset_convert":
            if self.network not in ("ocrnet", "pointpillars"):
                self.config["results_dir"] = CLI_CONFIG_TO_FUNCTIONS["output_dir"](
                    self.job_context,
                    self.handler_metadata
                )

        params_to_cli = build_cli_command(self.config)
        run_command = f"{self.network} {self.action} {params_to_cli}"
        if self.action == "trtexec":
            run_command = f"{self.action} {params_to_cli}"

        return run_command, None


# Specs are modified as well => Train, Evaluate, Retrain Actions
class TrainVal(CLIPipeline):
    """Class for experiment actions which involves both spec file as well as cli params"""

    def generate_config(self):
        """Generates spec and cli params

        Returns:
        spec: contains the network's spec file parameters
        config: contains cli params
        """
        network = self.job_context.network
        action = self.job_context.action
        # Infer CLI params
        config = {}
        network_config = read_network_config(network)
        if action in network_config.get("cli_params", {}).keys():
            for field_name, inference_fn in network_config["cli_params"][action].items():
                if inference_fn in CLI_CONFIG_TO_FUNCTIONS:
                    field_value = CLI_CONFIG_TO_FUNCTIONS[inference_fn](self.job_context, self.handler_metadata)
                    if field_value:
                        config[field_name] = field_value

        # Read spec from <action>.json for train, resume train, evaluate, retrain. If not there, use train.json
        spec = get_job_specs(self.job_context.id)
        if network in COPY_MODEL_PARAMS_FROM_TRAIN_NETWORKS:
            cur_job_id = self.job_context.id
            while True:
                cur_job_meta = get_handler_job_metadata(cur_job_id)
                if not cur_job_meta:
                    break
                parent_job_id = cur_job_meta.get("parent_id", "")
                if not parent_job_id:
                    break
                parent_job_metadata = get_handler_job_metadata(parent_job_id)
                parent_action = parent_job_metadata.get("action", "")
                if not parent_action:
                    break
                if parent_action in ("train", "distill", "quantize"):
                    from nvidia_tao_core.microservices.handlers.app_handler import AppHandler  # pylint: disable=C0415
                    default_spec_schema_response = AppHandler.get_spec_schema(
                        self.job_context.user_id,
                        self.job_context.org_name,
                        self.job_context.handler_id,
                        action,
                        self.handler_kind
                    )
                    user_modified_values = {}
                    if default_spec_schema_response.code == 200:
                        spec_schema = default_spec_schema_response.data
                        default_spec = spec_schema["default"]
                        user_modified_values = find_differences(spec, default_spec)
                    automl = False
                    best_rec_id, best_rec_job_id = get_automl_best_rec_info(parent_job_id)
                    logger.info(f"Best rec id: {best_rec_id}, Best rec job id: {best_rec_job_id}")
                    if best_rec_id != "-1":
                        automl = True
                        parent_spec = get_job_specs(best_rec_job_id, automl=automl, automl_experiment_id=best_rec_id)
                    else:
                        parent_spec = get_job_specs(parent_job_id)
                    train_specs_passed_in_req_body = get_job_specs(parent_job_id)
                    modified_values = find_differences(parent_spec, train_specs_passed_in_req_body)
                    spec = merge_nested_dicts(spec, modified_values)
                    spec = merge_nested_dicts(spec, user_modified_values)
                    break
                cur_job_id = parent_job_id
            save_job_specs(self.job_context.id, spec)

        # Take .json file, read in spec params, infer spec params
        if action in network_config["spec_params"].keys():
            for field_name, inference_fn in network_config["spec_params"][action].items():
                field_value = (
                    CLI_CONFIG_TO_FUNCTIONS[inference_fn](self.job_context, self.handler_metadata)
                    if inference_fn in CLI_CONFIG_TO_FUNCTIONS
                    else inference_fn
                )
                if field_value:
                    write_nested_dict(spec, field_name, field_value)

        # Move CLI params from spec to config
        spec_keys_all = copy.deepcopy(list(spec.keys()))
        if "cli_params" in network_config and action in network_config["cli_params"]:
            for field_name in spec_keys_all:
                cnd1 = field_name in network_config["cli_params"][action].keys()
                cnd2 = network_config["cli_params"][action].get(field_name, None) == "from_csv"
                cnd3 = type(spec[field_name]) in [str, float, int, bool]
                if cnd1 and cnd2 and cnd3:
                    config[field_name] = spec.pop(field_name)
        self.detailed_print("Loaded specs")

        # Infer dataset config
        spec = apply_data_source_config(spec, self.job_context, self.handler_metadata)
        self.detailed_print("Loaded dataset")

        return spec, config

    def post_run(self):
        """Carry's out functions after the job is executed"""
        # copy pruned model so that evaluate can access via parent relation
        action = self.job_context.action
        if self.network in ("ocdnet", "ocrnet") and action == "retrain":
            inference_fn = "parent_model"
            pruned_model_path = CLI_CONFIG_TO_FUNCTIONS[inference_fn](self.job_context, self.handler_metadata)
            bucket_name = pruned_model_path.split("//")[1].split("/")[0]
            pruned_model_path = pruned_model_path[pruned_model_path.find(bucket_name) + len(bucket_name):]
            _, file_extension = os.path.splitext(pruned_model_path)
            self.detailed_print(
                f"Copying pruned model {pruned_model_path} after retrain to "
                f"/results/{self.job_name}/pruned_model{file_extension}\n"
            )
            self.cs_instance.copy_file(pruned_model_path, f"/results/{self.job_name}/pruned_model{file_extension}")
        if self.job_context.action == "annotation_format_convert":
            handler_metadata = get_handler_metadata(self.handler_id, self.handler_kind)
            if self.spec["data"]["input_format"] == "KITTI":
                handler_metadata["format"] = "coco"
            elif self.spec["data"]["input_format"] == "COCO":
                handler_metadata["format"] = "kitti"
            write_handler_metadata(self.handler_id, handler_metadata, self.handler_kind)
        # Create dataset for data service actions that generate new dataset
        # These actions create a new dataset as part of their actions
        if action in _DATA_GENERATE_ACTIONS:
            from nvidia_tao_core.microservices.handlers.app_handler import AppHandler  # pylint: disable=C0415
            handler_metadata = get_handler_metadata(self.handler_id, self.handler_kind)
            request_dict = AppHandler.create_dataset_dict_from_experiment_metadata(
                self.job_context.id,
                self.action,
                handler_metadata
            )
            if action == "dataset_convert_gaze":
                request_dict["format"] = "maxine_gaze"
            response = AppHandler.create_dataset(
                self.job_context.user_id,
                self.job_context.org_name,
                request_dict,
                dataset_id=self.job_context.id
            )
            if response.code != 200:
                self.detailed_print(
                    f"Failed to create dataset from {self.action} job. "
                    f"Response code: {response.code}"
                )
                update_job_status(self.handler_id, self.job_context.id, status="Error", kind=self.handler_kind)


class AutoMLPipeline(ActionPipeline):
    """Class for handling AutoML pipeline operations.

    This class contains methods for managing and executing AutoML pipeline tasks.
    """

    def __init__(self, job_context):
        """Initialize the AutoMLPipeline class"""
        super().__init__(job_context)
        self.network, self.action = get_microservices_network_and_action(self.network, self.action)
        self.automl_brain_job_id = self.job_context.id
        self.job_root = os.path.join(
            get_jobs_root(self.job_context.user_id, self.job_context.org_name),
            self.automl_brain_job_id
        )
        self.rec_number = self.get_recommendation_number()
        self.expt_root = f"{self.job_root}/experiment_{self.rec_number}"
        self.recs_dict = get_automl_controller_info(self.automl_brain_job_id)
        self.brain_dict = get_automl_brain_info(self.automl_brain_job_id)
        # Assign a new job id if not assigned already
        self.job_name = self.recs_dict[self.rec_number].get("job_id", None)
        if not self.job_name:
            self.job_name = str(uuid.uuid4())
            self.detailed_print("New job id being assigned to recommendation", self.job_name)
            self.recs_dict[self.rec_number]["job_id"] = self.job_name
            save_automl_controller_info(self.automl_brain_job_id, self.recs_dict)

        if not os.path.exists(self.expt_root):
            os.makedirs(self.expt_root)

    def add_ptm_dependency(self, recommended_values):
        """Add PTM as a dependency if backbone or num_layers is part of hyperparameter sweep"""
        # See if a ptm is needed (if not searching num_layers / backbone, no PTM), just take default
        ptm_id = None
        if "backbone" in recommended_values.keys() or "num_layers" in recommended_values.keys():
            for dep in self.job_context.dependencies:
                if dep.type == "automl_ptm":
                    ptm_id = dep.name
                    break
        if ptm_id:
            recommended_values["base_experiment"] = search_for_base_experiment(
                get_handler_root(base_exp_uuid, "experiments", base_exp_uuid, ptm_id)
            )

    def generate_config(self, recommended_values):
        """Generate config for AutoML experiment"""
        spec = get_job_specs(self.automl_brain_job_id)

        epoch_multiplier = self.brain_dict.get("epoch_multiplier", None)
        if epoch_multiplier is not None:
            current_ri = int(
                self.brain_dict.get("ri", {"0": [float('-inf')]})[
                    str(self.brain_dict.get("bracket", 0))
                ][0]
            )

        for field_name, inference_fn in self.network_config["automl_spec_params"].items():
            if "automl_" in inference_fn:
                field_value = CLI_CONFIG_TO_FUNCTIONS[inference_fn](
                    self.job_context,
                    self.handler_metadata,
                    self.job_root,
                    self.rec_number,
                    self.job_name
                )
            elif "assign_const_value" in inference_fn:
                if epoch_multiplier is not None:
                    field_value = int(epoch_multiplier * current_ri)
                else:
                    field_value = int(read_nested_dict(spec, field_name))
                    if "assign_const_value," in inference_fn:
                        dependent_parameter_names = inference_fn.split(",")
                        dependent_field_value = int(read_nested_dict(spec, dependent_parameter_names[1]))
                        if len(dependent_parameter_names) == 2:
                            field_value = min(field_value, dependent_field_value)
                        elif len(dependent_parameter_names) == 3:
                            field_value = int(read_nested_dict(spec, dependent_parameter_names[2]))

            else:
                field_value = (
                    CLI_CONFIG_TO_FUNCTIONS[inference_fn](self.job_context, self.handler_metadata)
                    if inference_fn in CLI_CONFIG_TO_FUNCTIONS
                    else inference_fn
                )
            if field_value:
                write_nested_dict(spec, field_name, field_value)

        spec = apply_data_source_config(spec, self.job_context, self.handler_metadata)
        self.detailed_print("Loaded AutoML specs")

        for param_name, param_value in recommended_values.items():
            write_nested_dict(spec, param_name, param_value)
        self.num_gpu = get_num_gpus_from_spec(spec, "train", network=self.network, default=self.num_gpu)
        self.num_nodes = get_num_nodes_from_spec(spec, "train", network=self.network, default=self.num_nodes)

        return spec

    def save_recommendation_specs(self):
        """Save recommendation specs to AutoML brain DB"""
        updated_spec = SPEC_BACKEND_TO_FUNCTIONS[self.api_params["spec_backend"]](self.spec)
        save_job_specs(job_id=self.job_name, specs=updated_spec, automl=True, automl_experiment_id=str(self.rec_number))

    def generate_run_command(self):
        """Generate the command to be run inside docker for AutoML experiment"""
        params_to_cli = build_cli_command(self.config)
        run_command = f"{self.network} train {params_to_cli}"
        logfile = f'/{self.job_name}.txt'
        run_command += f"  2>&1 | tee {logfile}"
        return run_command

    def get_recommendation_number(self):
        """Return the current recommendation number"""
        rec_number = None
        for dep in self.job_context.dependencies:
            if dep.type == "automl":
                rec_number = int(dep.name)
                break
        return rec_number

    def monitor_job(self, nv_job_metadata=None):
        """Monitors the job status and updates job metadata"""
        if not self.spec:
            recommended_values = self.recs_dict[self.rec_number].get("specs", {})
            self.spec = self.generate_config(recommended_values)
            self.handle_ptm_anomalies()
            self.get_handler_cloud_details()
            self.save_recommendation_specs()

        if not self.job_env_variables:
            docker_env_vars = self.handler_metadata.get("docker_env_vars", {})
            self.decrypt_docker_env_vars(docker_env_vars)
            self.job_env_variables = copy.deepcopy(docker_env_vars)
            self.generate_env_variables(
                automl_brain_job_id=self.automl_brain_job_id,
                experiment_number=str(self.rec_number)
            )

        run_command = self.generate_run_command()
        if not nv_job_metadata:
            nv_job_metadata = {}
            if self.ngc_runner:
                self.generate_nv_job_metadata(nv_job_metadata)

        k8s_status = jobDriver.status(
            self.job_context.org_name,
            self.handler_id,
            self.job_name,
            self.handler_kind,
            use_ngc=self.ngc_runner,
            network=self.network,
            action=self.action,
            automl_exp_job=True,
            docker_env_vars=self.job_env_variables,
            automl_experiment_id=str(self.rec_number),
        )
        while k8s_status in ["Done", "Error", "Running", "Pending", "Creating"]:
            time.sleep(5)
            if get_dnn_status(
                self.automl_brain_job_id,
                automl=True,
                experiment_number=str(self.rec_number)
            ) or (BACKEND == "NVCF" and k8s_status == "Running"):
                break
            job_metadata = get_handler_job_metadata(self.automl_brain_job_id)
            detailed_message = (
                job_metadata.get("job_details", {})
                .get(self.job_name, {})
                .get("detailed_status", {})
                .get("message", "")
            )
            if "Invalid schema" in detailed_message:
                break
            if k8s_status == "Error":
                self.detailed_print(f"Relaunching job {self.job_name}")
                wait_for_job_completion(self.job_name)
                if self.network not in MONAI_NETWORKS and BACKEND in ("local-k8s", "local-docker"):
                    self.create_microservice_action_job(self.automl_brain_job_id)
                else:
                    jobDriver.create(
                        self.job_context.org_name,
                        self.job_name,
                        self.image,
                        run_command,
                        num_gpu=self.num_gpu,
                        num_nodes=self.num_nodes,
                        docker_env_vars=self.job_env_variables,
                        nv_job_metadata=nv_job_metadata,
                        automl_exp_job=True
                    )
            k8s_status = jobDriver.status(
                self.job_context.org_name,
                self.handler_id,
                self.job_name,
                self.handler_kind,
                use_ngc=self.ngc_runner,
                network=self.network,
                action=self.action,
                automl_exp_job=True,
                docker_env_vars=self.job_env_variables,
                automl_experiment_id=str(self.rec_number),
            )
        if k8s_status == "Error":
            self.recs_dict[self.rec_number]["status"] = "failure"
            save_automl_controller_info(self.automl_brain_job_id, self.recs_dict)

    def run(self):
        """Calls necessary setup functions and calls job creation"""
        try:
            recommended_values = self.recs_dict[self.rec_number].get("specs", {})
            self.cs_instance, _ = create_cs_instance(self.workspace_metadata)
            self.add_ptm_dependency(recommended_values)

            self.spec = self.generate_config(recommended_values)
            self.handle_ptm_anomalies()
            self.get_handler_cloud_details()
            self.save_recommendation_specs()
            run_command = self.generate_run_command()

            self.detailed_print(run_command)

            # Wait for existing AutoML jobs to complete
            wait_for_job_completion(self.job_name)

            delete_lingering_checkpoints(self.recs_dict[self.rec_number].get("best_epoch_number", ""), self.expt_root)
            docker_env_vars = self.handler_metadata.get("docker_env_vars", {})
            self.decrypt_docker_env_vars(docker_env_vars)
            self.job_env_variables = copy.deepcopy(docker_env_vars)
            self.generate_env_variables(
                automl_brain_job_id=self.automl_brain_job_id,
                experiment_number=str(self.rec_number),
                automl_exp_job_id=self.job_name
            )

            nv_job_metadata = {}
            if self.ngc_runner:
                self.generate_nv_job_metadata(nv_job_metadata)

            if self.network not in MONAI_NETWORKS and BACKEND in ("local-k8s", "local-docker"):
                self.create_microservice_action_job(self.automl_brain_job_id)
            else:
                jobDriver.create(
                    self.job_context.org_name,
                    self.job_name,
                    self.image,
                    run_command,
                    num_gpu=self.num_gpu,
                    num_nodes=self.num_nodes,
                    docker_env_vars=self.job_env_variables,
                    nv_job_metadata=nv_job_metadata,
                    automl_exp_job=False
                )
            self.detailed_print(
                f"AutoML recommendation with experiment id {self.rec_number} "
                f"and job id {self.job_name} submitted"
            )
            self.monitor_job(nv_job_metadata)

            return True

        except Exception as e:
            self.detailed_print(
                f"AutoMLpipeline for network {self.network} failed due to "
                f"exception {traceback.format_exc()}"
            )
            result_dict = {
                self.job_name: {
                    "detailed_status": {
                        "message": "Error due to unmet dependencies",
                        "status": "FAILURE"
                    }
                }
            }
            if isinstance(e, ValueError):
                result_dict = {self.job_name: {"detailed_status": {"message": str(e), "status": "FAILURE"}}}
            update_job_metadata(
                self.handler_id,
                self.automl_brain_job_id,
                metadata_key="job_details",
                data=result_dict,
                kind=self.handler_kind
            )
            self.detailed_print(self.job_name)

            self.recs_dict[self.rec_number]["status"] = "failure"
            save_automl_controller_info(self.automl_brain_job_id, self.recs_dict)

            update_job_status(self.handler_id, self.job_context.id, status="Error", kind=self.handler_kind)
            jobDriver.delete(self.job_context.id, use_ngc=False)
            return False


class ContinualLearning(ActionPipeline):
    """Class for continual learning specific changes required during annotation action."""

    def __init__(self, job_context):
        """Initialize the ContinualLearning class"""
        super().__init__(job_context)
        self.ngc_runner = False
        # override the ActionPipeline's handler_root
        self.train_ds = self.handler_metadata["train_datasets"]
        self.image = DOCKER_IMAGE_MAPPER["API"]
        self.job_root = os.path.join(self.jobs_root, self.job_context.id)
        self.logs_from_toolkit = f"{self.jobs_root}/{self.job_name}/logs_from_toolkit.txt"

    def generate_convert_script(self, notify_record):
        """Generate a script to perform continual learning"""
        cl_script = generate_cl_script(
            notify_record,
            self.job_context,
            self.handler_root,
            self.logfile,
            self.logs_from_toolkit
        )
        cl_script_path = os.path.join(self.job_root, "continual_learning.py")
        with open(cl_script_path, "w", encoding="utf-8") as f:
            f.write(cl_script)

        return cl_script_path

    def monitor_job(self):
        """Monitors the job status and updates job metadata"""
        k8s_status = jobDriver.status(
            self.job_context.org_name,
            self.handler_id,
            self.job_name,
            self.handler_kind,
            use_ngc=False,
            automl_exp_job=False,
            docker_env_vars=self.job_env_variables
        )
        while k8s_status in ["Running", "Pending"]:
            # Poll every 30 seconds
            time.sleep(30)
            k8s_status = jobDriver.status(
                self.job_context.org_name,
                self.handler_id,
                self.job_name,
                self.handler_kind,
                use_ngc=False,
                automl_exp_job=False,
                docker_env_vars=self.job_env_variables
            )
        self.detailed_print(f"Job status: {k8s_status}")
        if k8s_status == "Error":
            update_job_status(self.handler_id, self.job_context.id, status="Error")
        self.detailed_print("Continual Learning finished.")

    def run(self):
        """Run the continual learning pipeline"""
        # Set up
        self.thread = threading.current_thread()
        try:
            # Find the train dataset path
            if len(self.train_ds) != 1:
                raise ValueError("Continual Learning only supports one train dataset.")
            train_dataset = self.train_ds[0]
            dataset_path = get_handler_root(self.job_context.org_name, "datasets", train_dataset, None)

            # Track the current training manifest file
            notify_record = os.path.join(dataset_path, "notify_record.json")
            cl_script = self.generate_convert_script(notify_record)

            outdir = self.job_root
            self.detailed_print("Continual Learning started")
            self.run_command = f"python {cl_script} 2>&1 | tee {self.logfile}"
            # After command runs, make sure subdirs permission allows anyone to enter and delete
            self.run_command += f"; find {outdir} -type d | xargs chmod 777"
            # After command runs, make sure artifact files permission allows anyone to delete
            self.run_command += f"; find {outdir} -type f | xargs chmod 666"
            # Optionally, pipe self.run_command into a log file
            self.detailed_print(self.run_command)
            jobDriver.create(
                self.job_context.org_name,
                self.job_name,
                self.image,
                self.run_command,
                num_gpu=0,
                cl_medical=True,
                automl_exp_job=False
            )
            self.detailed_print("Job created", self.job_name)
            self.monitor_job()
        except Exception:
            self.detailed_print(
                f"ContinualLearning for {self.network} failed because "
                f"{traceback.format_exc()}"
            )
            update_job_status(self.handler_id, self.job_context.id, status="Error")


class BundleTrain(ActionPipeline):
    """Class for MONAI bundle specific changes required during training"""

    CLOUD_STORAGE_LIB = "cloud_storage"
    UTILS_LIB = "utils"

    def __init__(self, job_context):
        """Initialize the BundleTrain class"""
        super().__init__(job_context)
        spec = self.get_spec()
        # override the ActionPipeline's handler_root
        if "cluster" in spec and spec["cluster"] == "local":
            self.ngc_runner = False  # this will make the executor uses local gpu even if BACKEND = True
            # Though self.local_cluster is kind of duplicate of self.ngc_runner, a wide scope of property,
            # the diff is local_cluster only used to determine the mounting
            # strategy in executor create for medical bundles training.
            self.local_cluster = True
            self.handler_root = get_handler_root(self.job_context.org_name, "experiments", None)
        self.network = job_context.network
        self.action = job_context.action
        self.job_root = os.path.join(self.jobs_root, self.job_context.id)
        self.model_script_path = os.path.join(self.job_root, "train.py")
        self.config_file = "configs/train.json"
        self.bundle_dir = ""

    def get_spec(self):
        """Get spec from spec file or job context"""
        if self.job_context.specs is not None:
            spec = get_job_specs(self.job_context.id)
        else:
            raise RuntimeError("Spec needs to be passed at job run")

        for k, v in spec.items():
            if isinstance(v, str) and v == "$default_monai_label_mapping":
                labels = self.handler_metadata.get("model_params", {}).get("labels", None)
                spec[k] = {"default": [[int(i), int(i)] for i in labels]}
                self.detailed_print(f"Using default {k}: {spec[k]}")
        self.detailed_print("Specs are loaded.")
        return spec

    def generate_config(self):
        """Generate config for monai bundle train

        Returns:
        spec: contains the params for building monai bundle train command
        Notes:
        Similar to TrainVal.generate_config, but we have the following differences:
        - Drop using network config files
        - Drop using the csv spec file
        But we cannot delete the network config and the csv spec file,
        because they are used by the app_handler, e.g. save_spec and job_run methods.
        """
        spec = self.get_spec()
        success, msg = validate_monai_bundle_params(spec)
        if not success:
            self.detailed_print(msg)
            raise RuntimeError(msg)
        # prepare dataset and update spec
        self.detailed_print("Loaded specs")
        spec = apply_data_source_config(spec, self.job_context, self.handler_metadata)
        self.detailed_print("Loaded dataset")
        return spec, {}

    @staticmethod
    def add_prefix(config, prefix):
        """Add prefix to config string or config list."""
        if isinstance(config, list):
            config_file = [os.path.join(prefix, x) for x in config]  # If it's a list, join the items
        elif isinstance(config, str):
            config_file = os.path.join(prefix, config)
        else:
            raise RuntimeError(f"Do not support config file with type {type(config)}")
        return config_file

    def update_dataset_env_variables(self, datasets_info):
        """Update the environment variables relating to datasets."""
        for dataset_usage in datasets_info:
            secret = datasets_info[dataset_usage].pop("client_secret", "")
            secret_env = datasets_info[dataset_usage]["secret_env"]
            self.monai_env_variable.update({secret_env: secret})

    def get_dicom_convert(self, datasets_info):
        """Check if need to convert the dicom datasets."""
        is_convert_list = []
        for dataset_usage in datasets_info:
            dataset_info = datasets_info[dataset_usage]
            cur_is_dicom = dataset_info["is_dicom"]
            is_convert_list.append(cur_is_dicom)
        is_convert_set = set(is_convert_list)
        if len(is_convert_set) > 1:
            raise RuntimeError("Job does not accept multi type datasets.")

        return is_convert_set.pop()

    def get_ngc_path(self):
        """Get the ngc path of the pretrained model."""
        base_experiment_ids = self.handler_metadata.get("base_experiment", [])
        if len(base_experiment_ids) > 1:
            raise RuntimeError("Bundle train only supports 1 base experiment.")

        meta_data = get_base_experiment_metadata(base_experiment_ids[0])
        ngc_path = meta_data.get("ngc_path", "")
        return ngc_path

    def _get_bundle_url(self, root):
        """Get the url of the customized bundle."""
        bundle_url = ""
        bundle_file = os.path.join(root, CUSTOMIZED_BUNDLE_URL_FILE)
        if os.path.exists(bundle_file):
            bundle_content = safe_load_file(bundle_file)
            bundle_url = bundle_content.get(CUSTOMIZED_BUNDLE_URL_KEY, "")
        return bundle_url

    def get_experiment_cloud_meta(self):
        """Get experiment cloud storage metadata."""
        cloud_type = self.cloud_metadata["experiment"][0]["workspace"]["cloud_type"]
        cloud_specific_details = self.cloud_metadata["experiment"][0]["workspace"]["cloud_specific_details"]
        account_str = "account_name" if cloud_type == "azure" else "access_key"
        account_id = cloud_specific_details[account_str]
        secret_str = "access_key" if cloud_type == "azure" else "secret_key"
        secret = cloud_specific_details[secret_str]

        cloud_meta = {
            "cloud_type": cloud_type,
            "bucket_name": cloud_specific_details["cloud_bucket_name"],
            "region": cloud_specific_details.get("cloud_region", None),
            "access_key": account_id,
            "secret_key": secret,
        }
        return cloud_meta

    def get_job_bundle_info(self, bundle_name):
        """Get the bundle information of the job."""
        bundle_root = os.path.join(self.job_root, bundle_name)
        path_prefix = bundle_root + "/"
        override = self.spec if self.spec else {}
        config_file = self.add_prefix(override.pop("config_file", self.config_file), path_prefix)
        logging_file = self.add_prefix(override.pop("logging_file", "configs/logging.conf"), path_prefix)
        meta_file = self.add_prefix(override.pop("meta_file", "configs/metadata.json"), path_prefix)

        bundle_info = {
            "bundle_root": bundle_root,
            "config_file": config_file,
            "logging_file": logging_file,
            "meta_file": meta_file,
        }
        bundle_info.update(override)
        if self.num_gpu > 1:
            bundle_info["num_gpu"] = self.num_gpu
        return bundle_info

    def generate_datasets_meta(self, datasets_info):
        """Generate the datasets metadata dict."""
        labels = self.handler_metadata.get("model_params", {}).get("labels", None)
        return {"datasets_info": datasets_info, "labels": labels}

    def generate_experiment_meta(self, bundle_name):
        """Generate the experiment metadata dict."""
        bundle_info = self.get_job_bundle_info(bundle_name)
        experiment_cloud_meta = self.get_experiment_cloud_meta()

        experiment_meta = {
            "bundle_info": bundle_info,
            "experiment_cloud_meta": experiment_cloud_meta,
        }
        return experiment_meta

    def generate_job_meta(self, ptm_root, is_customized, src_dir, dst_dir):
        """Generate job metadata dict."""
        # ptm_meta = {"ptm_root": "ptm_root", "bundle_url": "bundle_url", "ngc_path":"ngc_path", "is_customized":"Bool"}
        bundle_url = self._get_bundle_url(ptm_root)
        ngc_path = self.get_ngc_path()
        ptm_meta = {"ptm_root": ptm_root,
                    "bundle_url": bundle_url,
                    "ngc_path": ngc_path,
                    "is_customized": is_customized,
                    }
        copy_meta = {"src_dir": src_dir, "dst_dir": dst_dir}
        return {"ptm_meta": ptm_meta, "copy_meta": copy_meta}

    def generate_run_command(self):
        """Generate batch inference run command"""
        ptm_root, bundle_name, overriden_output_dir = (
            CLI_CONFIG_TO_FUNCTIONS["monai_output_dir"](
                self.job_context,
                self.handler_metadata
            )
        )
        copy_needed = bool(ptm_root)
        overriden_output_dir = overriden_output_dir.rstrip(os.path.sep)
        datasets_info = self.spec.pop("datasets_info", {})
        self.update_dataset_env_variables(datasets_info)
        self.get_dicom_convert(datasets_info)
        src_dir = os.path.join(ptm_root, bundle_name) if copy_needed else ""
        dst_dir = os.path.join(overriden_output_dir, bundle_name) if copy_needed else ""
        self.bundle_dir = dst_dir
        experiment_json_meta = json.dumps(self.generate_experiment_meta(bundle_name))
        datasets_json_meta = json.dumps(self.generate_datasets_meta(datasets_info))
        is_customized = self.network in MEDICAL_CUSTOM_ARCHITECT
        job_json_meta = json.dumps(self.generate_job_meta(ptm_root, is_customized, src_dir, dst_dir))
        run_command = f"mkdir -p {overriden_output_dir} && mkdir -p {ptm_root} && cd {overriden_output_dir}" + " && "
        run_command += f"cp -r {MEDICAL_SERVICE_SCRIPTS}/* {overriden_output_dir} && "
        run_command += (
            f"python {self.action}.py "
            f"--experiment_json_meta '{experiment_json_meta}' "
            f"--datasets_json_meta '{datasets_json_meta}' "
            f"--job_json_meta '{job_json_meta}'"
        )
        if self.network in MEDICAL_CUSTOM_ARCHITECT:
            bundle_url = self._get_bundle_url(ptm_root)
            if not bundle_url:
                raise RuntimeError("Cannot find the pretrained model url for the customized bundle.")
        elif self.network != "monai_automl_generated":  # monai_automl_generated does not need ngc path nor ptm_root
            ngc_path = self.get_ngc_path()
            if not ngc_path:
                raise RuntimeError("Cannot find the ngc path for the pretrained model.")
        run_command = f"({run_command}) 2>&1"
        return run_command, overriden_output_dir


class BatchInfer(BundleTrain):
    """Class for MONAI bundle specific changes required during batch inference."""

    def __init__(self, job_context):
        """Initialize the BatchInfer class"""
        super().__init__(job_context)
        self.config_file = "configs/inference.json"

    def get_train_job_meta(self, train_job_id, bundle_name):
        """Get the train job meta data."""
        results_root = get_jobs_root(user_id=self.job_context.user_id, org_name=self.job_context.org_name)
        train_job_path = os.path.join(results_root, train_job_id, bundle_name)
        train_meta = {"train_meta": {"job_path": train_job_path}}
        return train_meta, train_job_path

    def generate_run_command(self):
        """Generate batch inference run command"""
        ptm_root, bundle_name, overriden_output_dir = (
            CLI_CONFIG_TO_FUNCTIONS["monai_output_dir"](
                self.job_context,
                self.handler_metadata
            )
        )
        datasets_info = self.spec.pop("datasets_info", {})
        train_job_id = self.spec.pop("train_job_id", "")
        if datasets_info:
            self.update_dataset_env_variables(datasets_info)
            self.get_dicom_convert(datasets_info)
        src_dir = os.path.join(ptm_root, bundle_name)
        dst_dir = os.path.join(overriden_output_dir, bundle_name)
        experiment_json_meta = json.dumps(self.generate_experiment_meta(bundle_name))
        datasets_json_meta = json.dumps(self.generate_datasets_meta(datasets_info))
        is_customized = self.network in MEDICAL_CUSTOM_ARCHITECT
        train_meta = {}
        if train_job_id:
            train_meta, train_job_path = self.get_train_job_meta(train_job_id, bundle_name)
            src_dir = train_job_path

        job_meta = self.generate_job_meta(ptm_root, is_customized, src_dir, dst_dir)
        job_meta.update(train_meta)
        job_json_meta = json.dumps(job_meta)
        run_command = f"mkdir -p {overriden_output_dir} && mkdir -p {ptm_root} && cd {overriden_output_dir}" + " && "
        run_command += f"cp -r {MEDICAL_SERVICE_SCRIPTS}/* {overriden_output_dir} && "
        run_command += (
            "python batchinfer.py "
            f"--experiment_json_meta '{experiment_json_meta}' "
            f"--datasets_json_meta '{datasets_json_meta}' "
            f"--job_json_meta '{job_json_meta}'"
        )
        if self.network in MEDICAL_CUSTOM_ARCHITECT:
            bundle_url = self._get_bundle_url(ptm_root)
            if not bundle_url:
                raise RuntimeError("Cannot find the pretrained model url for the customized bundle.")
        else:
            ngc_path = self.get_ngc_path()
            if not ngc_path:
                raise RuntimeError("Cannot find the ngc path for the pretrained model.")
        run_command = f"({run_command}) 2>&1"
        return run_command, overriden_output_dir


class Auto3DSegTrain(BundleTrain):
    """MONAI Auto3DSeg training action pipeline"""

    def generate_experiment_meta(self, bundle_name):
        """Generate the experiment metadata dict."""
        bundle_root = os.path.join(self.job_root, bundle_name)
        experiment_cloud_meta = self.get_experiment_cloud_meta()
        experiment_meta = {
            "input_info": {
                "work_dir": bundle_root,
                "templates_path_or_url": os.path.join(bundle_root, "algorithm_templates"),
                "multi_gpu": self.num_gpu > 1,
                **self.spec,
            },
            "experiment_cloud_meta": experiment_cloud_meta,
        }
        return experiment_meta

    def generate_job_meta(self, ptm_root, is_customized, src_dir, dst_dir):
        """Generate job metadata dict."""
        ngc_path = self.get_ngc_path()
        ptm_meta = {"ptm_root": ptm_root, "ngc_path": ngc_path}
        copy_meta = {"src_dir": src_dir, "dst_dir": dst_dir}
        return {"ptm_meta": ptm_meta, "copy_meta": copy_meta}

    def post_run(self):
        """Create a new model after the job is executed"""
        # set the url of the best model
        best_model_dir = os.path.join(self.bundle_dir, "best_model")
        cloud_folder = best_model_dir.lstrip(os.path.sep)
        # create a new experiment
        from nvidia_tao_core.microservices.handlers.app_handler import AppHandler  # pylint: disable=C0415
        request_dict = {
            "name": self.spec.get("output_experiment_name", "auto3dseg_automl_experiment"),
            "description": self.spec.get(
                "output_experiment_description",
                "AutoML Generated Segmentation Model based on MONAI Auto3DSeg"
            ),
            "type": "medical",
            "network_arch": "monai_automl_generated",
            "inference_dataset": self.spec.get("inference_dataset"),
            "realtime_infer": False,
            "bundle_url": cloud_folder,
            "workspace": self.handler_metadata.get("workspace")
        }
        ret_code = AppHandler.create_experiment(self.job_context.user_id, self.job_context.org_name, request_dict)
        new_model_id = ret_code.data["id"]
        self.detailed_print(f"New model is generated with id: {new_model_id}")

        # update the status file to include newly generated model id
        _ = {
            "date": datetime.date.today().isoformat(),
            "time": datetime.datetime.now().isoformat(),
            "status": "SUCCESS",
            "message": f"A segmentation experiment (id={new_model_id}) is successfully created by MONAI AuoML.",
            "model_id": new_model_id,
        }

        self.cs_instance.download_folder(cloud_folder, best_model_dir, maintain_src_folder_structure=True)


class Auto3DSegInfer(BundleTrain):
    """MONAI inference action pipeline"""

    def generate_config(self):
        """Generate spec for monai inference and creates "datalist.json"

        Returns:
            spec: contains the params for building monai bundle train command
        """
        # load spec
        spec = self.get_spec()

        # create datalist.json
        inference_dataset_id = spec.pop("inference_dataset", None)
        if inference_dataset_id:
            self.handler_metadata["inference_dataset"] = inference_dataset_id
        spec = apply_data_source_config(spec, self.job_context, self.handler_metadata)
        self.detailed_print("Datasets are loaded.")

        return spec, {}

    def generate_experiment_meta(self, bundle_name):
        """Generate the experiment metadata dict."""
        experiment_cloud_meta = self.get_experiment_cloud_meta()

        experiment_meta = {
            "input_info": {"work_dir": self.job_root, **self.spec},
            "experiment_cloud_meta": experiment_cloud_meta,
        }
        return experiment_meta

    def generate_job_meta(self, *args):
        """Generate job metadata dict."""
        handler_root = get_handler_root(self.job_context.org_name, "experiments", self.job_context.handler_id, None)
        auto3dseg_url = self._get_bundle_url(handler_root)
        if not auto3dseg_url:
            raise RuntimeError("Cannot find the pretrained Auto3DSeg model.")
        ptm_meta = {"bundle_url": auto3dseg_url, "is_auto3dseg_inference": True}
        return {"ptm_meta": ptm_meta}


# Each Element can be called with a job_context and returns an ActionPipeline (or its derivative) object
ACTIONS_TO_FUNCTIONS = {"train": TrainVal,
                        "evaluate": TrainVal,
                        "prune": CLIPipeline,
                        "prune_with_spec": TrainVal,
                        "retrain": TrainVal,
                        "export": CLIPipeline,
                        "export_with_spec": TrainVal,
                        "inference": TrainVal,
                        "gen_trt_engine": TrainVal,
                        "trtexec": TrainVal,
                        "purpose_built_models_ds_convert": TrainVal,
                        "odconvert": TrainVal,
                        "pyt_odconvert": TrainVal,
                        "data_services": TrainVal,
                        "monai_annotation": ContinualLearning,
                        "medical_automl_auto3dseg": Auto3DSegTrain,
                        "monai_train": BundleTrain,
                        "medical_automl_batchinfer": Auto3DSegInfer,
                        "monai_batchinfer": BatchInfer,
                        "monai_generate": BatchInfer,
                        }
