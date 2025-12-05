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

"""AutoML controller modules"""
import os
import re
import glob
import time
import uuid
import traceback
import logging
from copy import deepcopy
from datetime import timedelta

from nvidia_tao_core.microservices.utils.automl_utils import Recommendation, ResumeRecommendation, JobStates
from nvidia_tao_core.microservices.utils.core_utils import get_monitoring_metric
from nvidia_tao_core.microservices.constants import (
    _ITER_MODELS,
    NO_VAL_METRICS_DURING_TRAINING_NETWORKS,
    MISSING_EPOCH_FORMAT_NETWORKS
)
if os.getenv("BACKEND") == "NVCF":
    from nvidia_tao_core.microservices.dgx_controller import overwrite_job_logs_from_bcp
from nvidia_tao_core.microservices.utils.cloud_utils import create_cs_instance_with_decrypted_metadata
from nvidia_tao_core.microservices.utils.handler_utils import (
    StatusParser,
    get_total_epochs,
    get_file_list_from_cloud_storage,
    filter_files,
    format_epoch,
    get_network_config
)
from nvidia_tao_core.microservices.utils.stateless_handler_utils import (
    update_job_status,
    get_handler_metadata,
    write_handler_metadata,
    get_handler_job_metadata,
    update_job_metadata,
    update_job_message,
    write_job_metadata,
    get_job_specs,
    save_job_specs,
    get_automl_controller_info,
    save_automl_controller_info,
    get_automl_current_rec,
    save_automl_current_rec,
    save_automl_best_rec_info,
    get_automl_brain_info,
    delete_dnn_status,
    update_automl_stats,
    report_health_beat,
    delete_health_beat
)
from nvidia_tao_core.microservices.utils.automl_job_utils import (
    on_new_automl_job,
    on_delete_automl_job,
    on_cancel_automl_job
)

# Configure logging
TAO_LOG_LEVEL = os.getenv('TAO_LOG_LEVEL', 'INFO').upper()
tao_log_level = getattr(logging, TAO_LOG_LEVEL, logging.INFO)
logging.basicConfig(
    level=logging.WARNING,  # Root logger: suppress third-party DEBUG logs
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logging.getLogger('nvidia_tao_core').setLevel(tao_log_level)
logger = logging.getLogger(__name__)

time_per_epoch = 0
time_per_epoch_counter = 0


class Controller:
    """Abstractly, just a collection of threads and a switch to start and stop them

    - start(): Start all threads needed to run AutoML
    - stop(): Stop all threads started by start()
    - generate_recommendations(): Runs the automl algorithm to generate and analyze recommendations
    - read_results(): Listens to experiments
    - write_results(): Routinely updates a controller_data.json to help Handlers
    """

    def __init__(
        self,
        root,
        network,
        brain,
        automl_context,
        max_recommendations,
        delete_intermediate_ckpt,
        metric,
        automl_algorithm,
        decrypted_workspace_metadata
    ):
        """Initialize the Automl Controller class

        Args:
            root: handler root
            network: model name
            brain: Bayesian/Hyperband class object
            automl_context: job context with regards to automl
            max_recommendations: max_recommendation parameter value (for Bayesian)
            delete_intermediate_ckpt: boolean value to delete/not-delete checkpoints which don't correspond to the
            best model
            metric: metric name which will be used to choose best models
            automl_algorithm: automl algorithm name
        """
        self.brain = brain

        self.recommendations = []
        self.automl_context = automl_context
        logger.info("automl_context.id: %s", self.automl_context.id)

        self.root = root
        self.network = network
        self.checkpoint_delimiter = ""
        if self.network in MISSING_EPOCH_FORMAT_NETWORKS:
            self.checkpoint_delimiter = "_"
        self.completed_recommendations = 0
        self.max_recommendations = int(max_recommendations)
        self.delete_intermediate_ckpt = bool(delete_intermediate_ckpt.lower() == "true")
        self.automl_algorithm = automl_algorithm
        self.decrypted_workspace_metadata = decrypted_workspace_metadata
        self.metric = metric
        if self.automl_algorithm in ("hyperband", "h") and self.network in NO_VAL_METRICS_DURING_TRAINING_NETWORKS:
            self.metric_key = "loss"
            self.metric = "loss"
        elif self.metric == "kpi":
            self.metric_key = get_monitoring_metric(self.network)
        else:
            self.metric_key = self.metric

        self.brain.reverse_sort = True
        self.min_max = max
        if self.metric == "loss" or self.metric_key in ("loss", "evaluation_cost") or "loss" in self.metric_key:
            self.brain.reverse_sort = False
            self.min_max = min

        self.total_epochs = 0
        self.first_epoch_number = -1
        self.best_epoch_number = {}
        self.brain_epoch_number = -2
        self.best_model_copied = False
        self.best_rec_id = -1
        self.ckpt_path = {}

        self.old_bracket = "0"
        self.hyperband_cancel_condition_seen = False

        self.eta = "Will be updated after completing one epoch"
        self.remaining_epochs_in_experiment = float("inf")
        self.average_time_per_epoch = float("inf")

        self.on_new_automl_job = lambda jc: on_new_automl_job(self.automl_context, jc)

        self.cs_instance, _ = create_cs_instance_with_decrypted_metadata(self.decrypted_workspace_metadata)
        self.retain_checkpoints_for_resume = automl_context.retain_checkpoints_for_resume

    def _get_checkpoint_config(self):
        """Get checkpoint config from network config"""
        network_config = get_network_config(self.network)
        return network_config.get("checkpoint", {})

    def _uses_folder_lookup(self):
        """Check if network config specifies folder lookup method"""
        checkpoint_config = self._get_checkpoint_config()
        return checkpoint_config.get("folder", "false") == "true"

    def _inject_automl_env_vars(self):
        """Inject AutoML-specific environment variables into experiment docker_env_vars."""
        # Get current experiment metadata
        experiment_metadata = get_handler_metadata(self.automl_context.handler_id, "experiments")

        # Initialize docker_env_vars if not present
        if "docker_env_vars" not in experiment_metadata:
            experiment_metadata["docker_env_vars"] = {}

        # Set AutoML triggered flag and client type
        experiment_metadata["docker_env_vars"]["TAO_AUTOML_TRIGGERED"] = "true"
        if "TAO_CLIENT_TYPE" not in experiment_metadata["docker_env_vars"]:
            # Set to api since AutoML is triggered by API server
            experiment_metadata["docker_env_vars"]["TAO_CLIENT_TYPE"] = "api"

        # Save updated metadata
        write_handler_metadata(self.automl_context.handler_id, experiment_metadata, "experiments")
        logger.info(
            "Injected TAO_AUTOML_TRIGGERED=true for experiment %s",
            self.automl_context.handler_id
        )

    def _get_checkpoint_format(self):
        """Get checkpoint format from network config"""
        checkpoint_config = self._get_checkpoint_config()
        return checkpoint_config.get("format", "")

    def _select_best_epoch_folder(self, epoch_folder_files, all_files, specific_format=None):
        """Select the best epoch folder that contains the required checkpoint format

        Args:
            epoch_folder_files: Files matching the epoch pattern
            all_files: All available files
            specific_format: Optional specific format to look for (e.g., 'safetensors', 'pth')

        Returns:
            str: Path to the selected folder, or None if not found
        """
        checkpoint_format = specific_format or self._get_checkpoint_format()

        # Extract unique folder paths
        folder_paths = list(set(
            "/".join(f.split("/")[:-1]) for f in epoch_folder_files
        ))
        logger.info("Found potential epoch folders: %s", folder_paths)

        # Find the folder that contains files with the correct format
        for folder_path in folder_paths:
            folder_files = [f for f in all_files if f.startswith(folder_path + "/")]

            if checkpoint_format:
                # Check for specific format files
                format_files = [f for f in folder_files if f.endswith(f".{checkpoint_format}")]
                if format_files:
                    logger.info("Selected folder with %s files: %s", checkpoint_format, folder_path)
                    return folder_path
            else:
                # Check for any checkpoint files if no specific format
                checkpoint_files_in_folder = filter_files(
                    folder_files, network_name=self.network
                )
                if checkpoint_files_in_folder:
                    logger.info("Selected folder with checkpoint files: %s", folder_path)
                    return folder_path

        logger.info("No folder found containing required format files")
        return None

    def cancel_recommendation_jobs(self):
        """Cleanup recommendation jobs"""
        backend = os.getenv("TAO_EXECUTION_BACKEND", "local-k8s")
        for rec in self.recommendations:
            job_name = rec.job_id
            logger.info("Deleting %s", job_name)
            if not job_name:
                continue
            if not os.getenv("CI_PROJECT_DIR", None):
                logger.info("Cancelling automl job at end of controller %s", job_name)
                on_cancel_automl_job(rec.job_id)

            # Stop log monitoring for this experiment
            if backend in ("local-k8s", "local-docker"):
                try:
                    from nvidia_tao_core.microservices.utils.log_monitor_service import stop_monitoring_job
                    logger.info(f"[AUTOML-CONTROLLER] Stopping log monitoring for experiment job {job_name}")
                    stop_monitoring_job(job_name)
                    logger.info(f"[AUTOML-CONTROLLER] Stopped log monitoring for experiment job {job_name}")
                except Exception as e:
                    logger.warning(f"[AUTOML-CONTROLLER] Failed to stop log monitoring for {job_name}: {e}")

        logger.info("Deleting automl job %s", self.automl_context.id)
        on_delete_automl_job(self.automl_context.id)

    def start(self):
        """Starts the automl controller"""
        try:
            # Report initial health beat
            report_health_beat(self.automl_context.id, "AutoML controller starting")

            update_job_message(
                self.automl_context.handler_id,
                self.automl_context.id,
                "experiments",
                "AutoML train started, more details in automl_brain_info of response"
            )
            self._execute_loop()
            status = "Error"
            result_metadata = get_handler_job_metadata(self.automl_context.id)
            result_metadata["job_details"][self.automl_context.id] = {
                "detailed_status": {
                    "message": f"Checkpoint file doesn't exist in best model folder /results/{self.automl_context.id}",
                    "status": "FAILURE"
                }
            }
            if self.best_model_copied:
                status = "Done"
                result_metadata["job_details"][self.automl_context.id] = {
                    "detailed_status": {
                        "message": (
                            f"AutoML run is successful with best checkpoints under /results/{self.automl_context.id}"
                        ),
                        "status": "SUCCESS"
                    }
                }

            write_job_metadata(self.automl_context.id, result_metadata)
            update_job_status(self.automl_context.handler_id, self.automl_context.id, status=status, kind="experiments")
            self.cancel_recommendation_jobs()

            # Clean up health beat on completion
            delete_health_beat(self.automl_context.id)

        except Exception:
            logger.error(
                "AutoMLpipeline loop for network %s with job id %s failed due to exception %s",
                self.network, self.automl_context.id, traceback.format_exc()
            )
            result_metadata = get_handler_job_metadata(self.automl_context.id)
            result_metadata["job_details"][self.automl_context.id] = {
                "detailed_status": {
                    "message": "AutoML train failed due to run-time exception",
                    "status": "FAILURE"
                }
            }
            write_job_metadata(self.automl_context.id, result_metadata)
            self.cancel_recommendation_jobs()
            update_job_status(
                self.automl_context.handler_id,
                self.automl_context.id,
                status="Error",
                kind="experiments"
            )

            # Clean up health beat on error
            delete_health_beat(self.automl_context.id)

    def refresh_recommendations(self):
        """Refresh the recommendations"""
        self.recommendations = []
        recs_dict = get_automl_controller_info(self.automl_context.id)
        for rec_dict in recs_dict:
            rec = Recommendation(rec_dict["id"], rec_dict["specs"], self.metric_key)
            rec.update_result(rec_dict["result"])
            rec.update_status(rec_dict["status"])
            rec.assign_job_id(rec_dict["job_id"])
            self.recommendations.append(rec)

    def save_state(self):
        """Save the self.recommendations into automl brain DB"""
        recs_dict = [ele.__dict__ for ele in self.recommendations]
        metadata = get_handler_job_metadata(self.automl_context.id)
        current_status = metadata.get("status", "")
        logger.debug(
            f"[CONTROLLER-SAVE-STATE] About to save state: automl_job_id={self.automl_context.id}, "
            f"brain_status={current_status}, num_recs={len(recs_dict)}"
        )

        # Check if brain is being stopped
        if current_status in ("canceled", "canceling", "pausing", "paused"):
            logger.debug(
                f"[CONTROLLER-SAVE-STATE] Skipping save due to brain status: "
                f"automl_job_id={self.automl_context.id}, status={current_status}"
            )
            return

        # Check if any recommendations in MongoDB are already marked as canceled (race condition prevention)
        existing_recs = get_automl_controller_info(self.automl_context.id)
        if existing_recs:
            for existing_rec in existing_recs:
                if existing_rec.get("status") in ("canceled", "canceling"):
                    logger.debug(
                        f"[CONTROLLER-SAVE-STATE] Skipping save due to canceled recommendation in MongoDB: "
                        f"automl_job_id={self.automl_context.id}, rec_id={existing_rec.get('id')}, "
                        f"status={existing_rec.get('status')}"
                    )
                    return

        save_automl_controller_info(self.automl_context.id, recs_dict)
        logger.debug(
            f"[CONTROLLER-SAVE-STATE] Saved state to MongoDB: automl_job_id={self.automl_context.id}, "
            f"num_recs={len(recs_dict)}"
        )

    @staticmethod
    def load_state(
        root,
        network,
        brain,
        automl_context,
        max_recommendations,
        delete_intermediate_ckpt,
        metric,
        automl_algorithm,
        decrypted_workspace_metadata
    ):
        """Loads a Controller object from pre-existing root"""
        logger.debug(
            f"[CONTROLLER-LOAD-STATE] Starting controller load_state: "
            f"automl_job_id={automl_context.id}, algorithm={automl_algorithm}"
        )
        ctrl = Controller(
            root,
            network,
            brain,
            automl_context,
            max_recommendations,
            delete_intermediate_ckpt,
            metric,
            automl_algorithm,
            decrypted_workspace_metadata
        )
        ctrl.recommendations = []
        # Restore the recommendations
        recs_dict = get_automl_controller_info(automl_context.id)
        logger.debug(
            f"[CONTROLLER-LOAD-STATE] Retrieved recommendations from controller info: "
            f"automl_job_id={automl_context.id}, num_recs={len(recs_dict) if recs_dict else 0}"
        )
        if recs_dict:
            logger.debug(
                f"[CONTROLLER-LOAD-STATE] Raw recommendations from MongoDB: "
                f"automl_job_id={automl_context.id}, "
                f"recs={[{'id': r['id'], 'status': r['status'], 'job_id': r['job_id']} for r in recs_dict]}"
            )

        for idx, rec_dict in enumerate(recs_dict):
            rec = Recommendation(rec_dict["id"], rec_dict["specs"], ctrl.metric_key)
            rec.update_result(rec_dict["result"])
            logger.debug(
                f"[CONTROLLER-LOAD-STATE] Before update_status: automl_job_id={automl_context.id}, "
                f"rec_id={rec_dict['id']}, rec.status={rec.status}, db_status={rec_dict['status']}"
            )
            rec.update_status(rec_dict["status"])
            logger.debug(
                f"[CONTROLLER-LOAD-STATE] After update_status: automl_job_id={automl_context.id}, "
                f"rec_id={rec_dict['id']}, rec.status={rec.status}"
            )
            rec.assign_job_id(rec_dict["job_id"])
            ctrl.recommendations.append(rec)
            ctrl.best_epoch_number[rec_dict["id"]] = (
                rec_dict.get("best_epoch_number") if rec_dict.get("best_epoch_number") else 0
            )
            logger.debug(
                f"[CONTROLLER-LOAD-STATE] Restored recommendation {idx}: automl_job_id={automl_context.id}, "
                f"rec_id={rec_dict['id']}, status={rec_dict['status']}, job_id={rec_dict['job_id']}"
            )

        # Handle temp_rec
        # temp_rec is a recommendation that started, but never ended
        # Usually, if the controller is stopped before a recommendation is done,
        # it might have to be started / resumed again
        temp_rec = get_automl_current_rec(automl_context.id)
        logger.debug(
            f"[CONTROLLER-LOAD-STATE] Retrieved current recommendation index: "
            f"automl_job_id={automl_context.id}, temp_rec={temp_rec}, "
            f"num_recommendations={len(ctrl.recommendations)}"
        )
        # if ctrl.recommendations[temp_rec].status != JobStates.canceled:
        #     ctrl.recommendations[temp_rec].update_status(JobStates.success)
        ctrl.save_state()

        if temp_rec is not None and temp_rec < len(ctrl.recommendations):
            rec_status = ctrl.recommendations[temp_rec].status
            logger.debug(
                f"[CONTROLLER-LOAD-STATE] Checking if should resume recommendation: "
                f"automl_job_id={automl_context.id}, temp_rec={temp_rec}, status={rec_status}, "
                f"is_canceled={rec_status == JobStates.canceled}"
            )
            if rec_status == JobStates.canceled:
                logger.debug(
                    f"[CONTROLLER-LOAD-STATE] Resuming stopped automl sub-experiment: "
                    f"automl_job_id={automl_context.id}, rec_id={temp_rec}, "
                    f"job_id={ctrl.recommendations[temp_rec].job_id}"
                )
                if ctrl.automl_algorithm == "hyperband":
                    ctrl.brain.track_id = temp_rec
                    logger.debug(
                        f"[CONTROLLER-LOAD-STATE] Set Hyperband brain track_id: "
                        f"automl_job_id={automl_context.id}, track_id={temp_rec}"
                    )
                ctrl.on_new_automl_job(ctrl.recommendations[temp_rec])
                logger.debug(
                    f"[CONTROLLER-LOAD-STATE] Queued recommendation for resume: "
                    f"automl_job_id={automl_context.id}, rec_id={temp_rec}"
                )
            else:
                logger.debug(
                    f"[CONTROLLER-LOAD-STATE] NOT resuming recommendation (status is not canceled): "
                    f"automl_job_id={automl_context.id}, temp_rec={temp_rec}, status={rec_status}"
                )
        else:
            logger.warning(
                f"[CONTROLLER-LOAD-STATE] Invalid temp_rec or no recommendations to resume: "
                f"automl_job_id={automl_context.id}, temp_rec={temp_rec}, "
                f"num_recommendations={len(ctrl.recommendations)}"
            )

        logger.debug(
            f"[CONTROLLER-LOAD-STATE] Controller load_state completed: automl_job_id={automl_context.id}"
        )
        return ctrl

    def _execute_loop(self):
        """A loop that does the 3 things in order

        1.See if any new recommendation is up to execute
        2.Reads results of newly done experiments
        3.Writes AutoML status into a file which can be shown to the end user
        """
        update_job_status(self.automl_context.handler_id, self.automl_context.id, status="Running", kind="experiments")

        # Report health beat at start of loop
        report_health_beat(self.automl_context.id, "AutoML execute loop started")

        while True:
            # Report health beat on each iteration
            report_health_beat(
                self.automl_context.id,
                f"AutoML loop iteration (completed: {self.completed_recommendations}/{self.max_recommendations})"
            )

            metadata = get_handler_job_metadata(self.automl_context.id)
            current_status = metadata.get("status", "")
            automl_status = get_automl_controller_info(self.automl_context.id)
            if current_status in ("canceled", "canceling"):
                return
            if automl_status:
                self.completed_recommendations = len(automl_status)
                if (
                    self.completed_recommendations == self.max_recommendations and
                    automl_status[self.max_recommendations - 1]['status'] in ('success', 'failure') and
                    self.automl_algorithm in ("bayesian", "b")
                ) or (
                    self.automl_algorithm in ("hyperband", "h") and self.brain.done()
                ):
                    # Find best model based on mAP
                    logger.info("Finding best model")
                    self.best_rec_id = self.find_best_model()
                    logger.info("best_model_copied result %s", self.best_model_copied)

                    if self.best_model_copied:
                        # Delete final extra checkpoints after finish training
                        for rec in self.recommendations:
                            expt_root = os.path.join("/results", rec.job_id)
                            self.get_best_checkpoint_path(expt_root, rec)
                            if self.delete_intermediate_ckpt:
                                self.delete_not_best_model_checkpoints(expt_root, rec, True)
                        handler_metadata = get_handler_metadata(
                            self.automl_context.handler_id, "experiments"
                        )
                        handler_metadata["checkpoint_epoch_number"][
                            f"best_model_{self.automl_context.id}"
                        ] = self.best_epoch_number[self.best_rec_id]
                        handler_metadata["checkpoint_epoch_number"][
                            f"latest_model_{self.automl_context.id}"
                        ] = self.best_epoch_number[self.best_rec_id]
                        write_handler_metadata(self.automl_context.handler_id, handler_metadata, "experiments")

                    self.eta = 0.0
                    self.remaining_epochs_in_experiment = 0.0
                    self.write_results(final=True)
                    return

            self.run_experiments()
            self.read_results()
            self.write_results()
            if not os.getenv("CI_PROJECT_DIR", None):
                time.sleep(10)

    def run_experiments(self):
        """Generate recommendation from brain

        if a new job is requested, add it to self.recommendations and execute it (add it to workflow)
        if a resume is requested, add the relevant recommendation to the workflow
        """
        report_health_beat(self.automl_context.id, "Running experiments")

        if (self.automl_algorithm in ("bayesian", "b") and
                len(self.recommendations) == self.max_recommendations):
            return
        history = deepcopy(self.recommendations)
        recommended_specs = self.brain.generate_recommendations(history)
        assert len(recommended_specs) in [0, 1], "At most one recommendation"
        for spec in recommended_specs:
            logger.info("Recommendation received for %s", self.network)
            if type(spec) is dict:
                # Save brain state and update current recommendation
                self.hyperband_cancel_condition_seen = False
                self.brain.save_state()
                if self.automl_algorithm in ("hyperband", "h"):
                    self.automl_context.early_stop_epoch = self.brain.epoch_number
                # update temp_rec
                new_id = len(self.recommendations)
                self.best_epoch_number[new_id] = 0
                save_automl_current_rec(self.automl_context.id, new_id)

                # Run new recommendation
                rec = Recommendation(new_id, spec, self.metric_key)
                job_id = str(uuid.uuid4())  # Assign job_id for this recommendation
                rec.assign_job_id(job_id)
                self.recommendations.append(rec)
                self.save_state()

                # Inject TAO_AUTOML_TRIGGERED flag into experiment docker_env_vars
                self._inject_automl_env_vars()

                self.on_new_automl_job(rec)

            elif type(spec) is ResumeRecommendation:
                logger.debug(
                    f"[AUTOML-CONTROLLER-RESUME] Resume recommendation received: "
                    f"automl_job_id={self.automl_context.id}, rec_id={spec.id}"
                )
                self.hyperband_cancel_condition_seen = False
                rec_id = spec.id
                self.best_epoch_number[rec_id] = 0
                logger.debug(
                    f"[AUTOML-CONTROLLER-RESUME] Reset best epoch number for recommendation: "
                    f"automl_job_id={self.automl_context.id}, rec_id={rec_id}"
                )

                # Save brain state and update current recommendation
                self.brain.save_state()
                logger.debug(
                    f"[AUTOML-CONTROLLER-RESUME] Saved brain state: "
                    f"automl_job_id={self.automl_context.id}"
                )

                if self.automl_algorithm in ("hyperband", "h"):
                    self.automl_context.early_stop_epoch = self.brain.epoch_number
                    logger.debug(
                        f"[AUTOML-CONTROLLER-RESUME] Set early_stop_epoch for Hyperband: "
                        f"automl_job_id={self.automl_context.id}, rec_id={rec_id}, "
                        f"early_stop_epoch={self.brain.epoch_number}"
                    )

                # update temp_rec
                save_automl_current_rec(self.automl_context.id, rec_id)
                logger.debug(
                    f"[AUTOML-CONTROLLER-RESUME] Updated current recommendation: "
                    f"automl_job_id={self.automl_context.id}, rec_id={rec_id}"
                )

                assert (self.recommendations[rec_id].id == rec_id), (
                    f"Recommendation ID mismatch: expected {rec_id} but got "
                    f"{self.recommendations[rec_id].id}"
                )
                self.recommendations[rec_id].specs = spec.specs.copy()
                self.recommendations[rec_id].update_status(JobStates.pending)
                logger.debug(
                    f"[AUTOML-CONTROLLER-RESUME] Updated recommendation specs and status to pending: "
                    f"automl_job_id={self.automl_context.id}, rec_id={rec_id}"
                )

                # Remove previous files (except checkpoints) from experiment folder.
                def remove_files(local_expt_path, cloud_expt_path, rec_id):
                    logger.debug(
                        f"[AUTOML-CONTROLLER-RESUME] Cleaning up previous experiment files: "
                        f"automl_job_id={self.automl_context.id}, rec_id={rec_id}, "
                        f"local_path={local_expt_path}, cloud_path={cloud_expt_path}"
                    )
                    expt_file_name = get_file_list_from_cloud_storage(
                        self.decrypted_workspace_metadata, cloud_expt_path)
                    regex_pattern = r'.*(?:lightning_logs|events).*$|.*\.(json)$'
                    expt_file_name = filter_files(expt_file_name, regex_pattern)
                    logger.debug(
                        f"[AUTOML-CONTROLLER-RESUME] Found {len(expt_file_name)} files to delete from cloud: "
                        f"automl_job_id={self.automl_context.id}, rec_id={rec_id}"
                    )
                    for file_name in expt_file_name:
                        self.cs_instance.delete_file(file_name)
                    if os.path.exists(local_expt_path):
                        expt_file_name = glob.glob(local_expt_path + "/**/*.txt", recursive=True)
                        # Filter out microservices_log.txt - we want to keep logs from log streaming
                        expt_file_name = [f for f in expt_file_name if not f.endswith('microservices_log.txt')]
                        logger.debug(
                            f"[AUTOML-CONTROLLER-RESUME] Removing {len(expt_file_name)} log files "
                            f"(excluding microservices_log.txt): "
                            f"automl_job_id={self.automl_context.id}, rec_id={rec_id}, files={expt_file_name}"
                        )
                        for file_name in expt_file_name:
                            if os.path.isfile(file_name):
                                os.remove(file_name)
                    else:
                        logger.debug(
                            f"[AUTOML-CONTROLLER-RESUME] Local experiment path does not exist: "
                            f"automl_job_id={self.automl_context.id}, rec_id={rec_id}, path={local_expt_path}"
                        )
                    delete_dnn_status(self.automl_context.id, automl=True, experiment_number=str(rec_id))
                    logger.debug(
                        f"[AUTOML-CONTROLLER-RESUME] Deleted DNN status for experiment: "
                        f"automl_job_id={self.automl_context.id}, rec_id={rec_id}"
                    )

                expt_name = "experiment_" + str(rec_id)
                remove_files(
                    os.path.join(self.root, expt_name),
                    os.path.join("/results", self.recommendations[rec_id].job_id),
                    rec_id
                )
                logger.debug(
                    f"[AUTOML-CONTROLLER-RESUME] Completed cleanup for resumed experiment: "
                    f"automl_job_id={self.automl_context.id}, rec_id={rec_id}"
                )

                self.save_state()
                logger.debug(
                    f"[AUTOML-CONTROLLER-RESUME] Saved controller state: "
                    f"automl_job_id={self.automl_context.id}"
                )

                # Inject TAO_AUTOML_TRIGGERED flag into experiment docker_env_vars
                self._inject_automl_env_vars()
                logger.debug(
                    f"[AUTOML-CONTROLLER-RESUME] Injected AutoML environment variables: "
                    f"automl_job_id={self.automl_context.id}"
                )

                logger.debug(
                    f"[AUTOML-CONTROLLER-RESUME] Queueing resumed recommendation job: "
                    f"automl_job_id={self.automl_context.id}, rec_id={rec_id}, "
                    f"rec_job_id={self.recommendations[rec_id].job_id}"
                )
                self.on_new_automl_job(self.recommendations[rec_id])
                logger.debug(
                    f"[AUTOML-CONTROLLER-RESUME] Resume recommendation processing completed: "
                    f"automl_job_id={self.automl_context.id}, rec_id={rec_id}"
                )

    def read_results(self):
        """Update results for each recommendation"""
        report_health_beat(self.automl_context.id, "Reading results")

        flag = False
        self.refresh_recommendations()
        for rec in self.recommendations:
            # Report health beat for each experiment being processed
            report_health_beat(
                self.automl_context.id,
                f"Processing experiment {rec.id} (status: {rec.status})"
            )

            old_status = rec.status

            job_name = rec.job_id
            if not job_name:
                continue

            expt_name = "experiment_" + str(rec.id)
            local_expt_root = os.path.join(self.root, expt_name)
            cloud_expt_root = os.path.join("/results", rec.job_id)

            # If rec already changed to Success, no need to check
            if rec.status in [JobStates.success, JobStates.failure]:
                # Apply penalty for failed experiments if result is still 0.0
                if rec.status == JobStates.failure and rec.result == 0.0:
                    if self.brain.reverse_sort:
                        penalty_value = 1e-7  # Low penalty for metrics where higher is better
                    else:
                        penalty_value = 1e7  # High penalty for metrics where lower is better
                    logger.warning(
                        f"AutoML experiment {rec.id} (job {rec.job_id}) failed with result 0.0. "
                        f"Assigning penalty value {penalty_value} based on metric '{self.metric_key}' "
                        f"(reverse_sort={self.brain.reverse_sort}) to enable AutoML optimization to continue."
                    )
                    rec.update_result(penalty_value)
                    self.save_state()

                if self.delete_intermediate_ckpt:
                    self.delete_checkpoint_files(cloud_expt_root, rec)
                    # Remove the checkpoints from not best model
                    brain_dict = get_automl_brain_info(self.automl_context.id)
                    if brain_dict:
                        if (
                            self.automl_algorithm in ("bayesian", "b") or
                            self.old_bracket != brain_dict.get("bracket", "0")
                        ):
                            flag = self.delete_not_best_model_checkpoints(cloud_expt_root, rec, flag)
                continue

            status_parser = StatusParser(self.network, local_expt_root, self.first_epoch_number)

            new_results = status_parser.update_results(
                experiment_number=str(rec.id),
                automl=True,
                job_id=self.automl_context.id,
                rec_job_id=rec.job_id
            )
            report_health_beat(self.automl_context.id, f"Recieved updated results for experiment {rec.id}")
            self.calculate_eta(new_results, rec.job_id, rec.id)
            metadata = get_handler_job_metadata(self.automl_context.id)
            results = metadata.get("job_details", {})
            brain_dict = get_automl_brain_info(self.automl_context.id)
            self.brain_epoch_number = float(brain_dict.get("epoch_number", float('inf')))
            # Calculate last_seen_epoch and ensure it's non-negative
            last_seen_epoch_value = max(0, self.total_epochs - self.remaining_epochs_in_experiment)
            new_results = status_parser.update_results(
                experiment_number=str(rec.id),
                total_epochs=self.total_epochs,
                eta=self.eta,
                last_seen_epoch=last_seen_epoch_value,
                automl=True,
                job_id=self.automl_context.id,
                previous_result_metadata=results,
                automl_brain=True
            )
            new_results = status_parser.update_results(
                experiment_number=str(rec.id),
                total_epochs=self.total_epochs,
                last_seen_epoch=last_seen_epoch_value,
                automl=True,
                job_id=self.automl_context.id,
                rec_job_id=rec.job_id,
                previous_result_metadata=results
            )
            if status_parser.first_epoch_number != -1:
                self.first_epoch_number = status_parser.first_epoch_number
            detailed_status_message = (
                metadata.get("job_details", {})
                .get(self.automl_context.id, {})
                .get("detailed_status", {})
                .get("message", "")
            )
            if "Invalid schema" not in detailed_status_message:
                update_job_metadata(
                    self.automl_context.handler_id,
                    self.automl_context.id,
                    metadata_key="job_details",
                    data=new_results,
                    kind="experiments"
                )

            # Status is read from the status DB and not from K8s
            status = ""
            if rec.status == JobStates.success:
                status = JobStates.success
            elif new_results[rec.job_id].get("detailed_status"):
                status = new_results[rec.job_id]["detailed_status"].get("status", JobStates.pending).lower()
            if not status:
                status = JobStates.pending
            if status in [JobStates.success, JobStates.failure]:
                logger.info("Post processing of job %s under automl algorithm %s", rec.job_id, self.automl_algorithm)
                brain_epoch_number = self.brain_epoch_number
                if self.automl_algorithm in ("bayesian", "b"):
                    self.brain.num_epochs_per_experiment = get_total_epochs(
                        self.automl_context,
                        os.path.dirname(self.root),
                    )
                    brain_epoch_number = self.brain.num_epochs_per_experiment
                report_health_beat(
                    self.automl_context.id,
                    f"Reading final metrics for experiment {rec.id}"
                )
                validation_map, self.best_epoch_number[rec.id], _ = status_parser.read_metric(
                    results=new_results[rec.job_id],
                    metric=self.metric,
                    automl_algorithm=self.automl_algorithm,
                    automl_brain_job_id=self.automl_context.id,
                    brain_epoch_number=brain_epoch_number
                )
                if status == JobStates.failure:
                    if self.brain.reverse_sort:
                        validation_map = 1e-7
                    else:
                        validation_map = 1e7
                    logger.warning(
                        f"AutoML experiment {rec.id} (job {rec.job_id}) failed. "
                        f"Assigning penalty value {validation_map} to enable Bayesian optimization to continue."
                    )
                rec.update_result(validation_map)
                self.save_state()

                # Enhanced logging for job cancellation with full context
                logger.debug(
                    f"{'-' * 80}\n"
                    f"AUTOML CONTROLLER: CANCELLING EXPERIMENT JOB\n"
                    f"Brain Job ID: {self.automl_context.id}\n"
                    f"Experiment ID: {rec.id}\n"
                    f"Experiment Job ID: {rec.job_id}\n"
                    f"Final Status: {status}\n"
                    f"Final Result: {validation_map}\n"
                    f"Reason: Experiment completed with status={status}\n"
                    f"Action: Calling on_cancel_automl_job to delete StatefulSet\n"
                    f"{'-' * 80}"
                )

                report_health_beat(
                    self.automl_context.id,
                    f"Cancelling completed job {rec.job_id} (experiment {rec.id})"
                )
                on_cancel_automl_job(rec.job_id)
            if old_status != status:
                rec.update_status(status)
                self.save_state()
                if status == JobStates.success:
                    container_log_file = f"{self.root}/{rec.job_id}/log.txt"
                    if os.getenv("BACKEND") == "NVCF":
                        overwrite_job_logs_from_bcp(container_log_file, rec.job_id)
                    if os.path.exists(container_log_file):
                        with open(container_log_file, "a", encoding='utf-8') as f:
                            f.write("\nEOF\n")

            if rec.status in [JobStates.success, JobStates.failure] and self.delete_intermediate_ckpt:
                # Retain the latest checkpoint and remove others in experiment folder
                self.delete_checkpoint_files(cloud_expt_root, rec)

        if self.automl_algorithm in ("hyperband", "h"):
            if brain_dict:
                self.old_bracket = brain_dict.get("bracket", "0")

    def calculate_eta(self, new_results, rec_job_id, rec_id):
        """Calculate estimated time remaining for automl job"""
        report_health_beat(self.automl_context.id, f"Calculating ETA for experiment {rec_id}")

        global time_per_epoch  # pylint: disable=global-statement
        global time_per_epoch_counter  # pylint: disable=global-statement
        self.total_epochs = 0
        if self.automl_algorithm in ("bayesian", "b"):
            self.total_epochs = self.max_recommendations * self.brain.num_epochs_per_experiment
        elif self.automl_algorithm in ("hyperband", "h"):
            for key in self.brain.ni:
                experiments = self.brain.ni[key]
                epochs = self.brain.ri[key]
                for i, num_epochs in enumerate(epochs):
                    if i == 0:
                        self.total_epochs += experiments[i] * num_epochs
                    else:
                        self.total_epochs += experiments[i] * (epochs[i] - epochs[i - 1])
            self.total_epochs *= self.brain.epoch_multiplier

        for result_key in new_results.get(rec_job_id, {}).keys():
            if result_key in ("epoch", "cur_iter") and new_results[rec_job_id].get(result_key):
                current_epoch = new_results[rec_job_id].get(result_key)
                if result_key == "cur_iter":
                    time_per_key = "time_per_iter"
                else:
                    time_per_key = "time_per_epoch"
                time_per_epoch_string = new_results[rec_job_id].get(time_per_key, "0:0:0.0")
                if time_per_epoch_string:
                    format_time_per_epoch = time.strptime(time_per_epoch_string.split(".")[0], '%H:%M:%S')
                    time_per_epoch += (
                        format_time_per_epoch.tm_hour * 60 * 60 +
                        format_time_per_epoch.tm_min * 60 +
                        format_time_per_epoch.tm_sec
                    )
                else:
                    time_per_epoch = 0
                time_per_epoch_counter += 1
                self.average_time_per_epoch = time_per_epoch / time_per_epoch_counter

                if self.automl_algorithm in ("bayesian", "b"):
                    current_experiment_epoch = get_total_epochs(
                        rec_job_id,
                        os.path.dirname(self.root),
                        automl=True,
                        automl_experiment_id=str(rec_id)
                    )
                    remaining_epochs = current_experiment_epoch - current_epoch
                    self.remaining_epochs_in_experiment = (
                        remaining_epochs +
                        (self.max_recommendations - self.completed_recommendations) *
                        (self.brain.num_epochs_per_experiment)
                    )
                    self.eta = self.remaining_epochs_in_experiment * self.average_time_per_epoch

                elif self.automl_algorithm in ("hyperband", "h"):
                    # Calculate completed epochs for completed sh sessions
                    completed_epochs = 0
                    for bracket in range(0, int(self.brain.bracket) + 1):
                        local_sh_iter = len(self.brain.ni[str(bracket)])
                        if bracket == int(self.brain.bracket):
                            local_sh_iter = self.brain.sh_iter
                        for sh in range(0, local_sh_iter):
                            if (sh == 0):
                                completed_epochs += self.brain.ni[str(bracket)][sh] * self.brain.ri[str(bracket)][sh]
                            else:
                                completed_epochs += (
                                    self.brain.ni[str(bracket)][sh] *
                                    (self.brain.ri[str(bracket)][sh] - self.brain.ri[str(bracket)][sh - 1])
                                )

                    # Calculate completed epochs for current sh session
                    current_sh_allowed_epochs = (
                        self.brain.ri[self.brain.bracket][self.brain.sh_iter] *
                        self.brain.epoch_multiplier
                    )
                    if self.brain.sh_iter > 0:
                        current_sh_allowed_epochs = (
                            (self.brain.ri[self.brain.bracket][self.brain.sh_iter] -
                             self.brain.ri[self.brain.bracket][self.brain.sh_iter - 1]) *
                            self.brain.epoch_multiplier
                        )
                    completed_epochs += self.brain.expt_iter * current_sh_allowed_epochs

                    self.remaining_epochs_in_experiment = max(0, self.total_epochs - completed_epochs)
                    self.eta = self.remaining_epochs_in_experiment * self.average_time_per_epoch

        if self.remaining_epochs_in_experiment == float("inf") or self.remaining_epochs_in_experiment == float("-inf"):
            self.remaining_epochs_in_experiment = self.total_epochs

    def write_results(self, final=False):
        """Update stats value and write to job metadata"""
        report_health_beat(self.automl_context.id, "Writing results" if not final else "Writing final results")

        # Best mAP seen till now
        result_dict = {}
        try:
            # Filter recommendations to only those with completed status
            valid_recs = [r for r in self.recommendations
                          if r.status in (JobStates.success, JobStates.failure)]

            if valid_recs:
                best_metric_value = self.min_max(valid_recs, key=lambda rec: rec.result).result
            else:
                best_metric_value = 0.0

            result_dict[f"best_{self.metric_key}"] = best_metric_value
        except Exception as e:
            logger.error("Exception thrown in write_results is %s", str(e))
            result_dict[f"best_{self.metric_key}"] = 0.0

        if type(self.eta) is float:
            self.eta = str(timedelta(seconds=self.eta))
        result_dict["Estimated time for automl completion"] = str(self.eta)
        result_dict["Current experiment id"] = len(self.recommendations)

        if self.network in _ITER_MODELS:
            result_dict["Number of iters yet to start"] = self.remaining_epochs_in_experiment
            result_dict["Time per iter in seconds"] = round(self.average_time_per_epoch, 2)
        else:
            result_dict["Number of epochs yet to start"] = self.remaining_epochs_in_experiment
            result_dict["Time per epoch in seconds"] = round(self.average_time_per_epoch, 2)

        # Only consider successful recommendations (not failures with penalties)
        completed_recs = [
            rec for rec in self.recommendations
            if rec.status == JobStates.success
        ]
        if completed_recs:
            try:
                best_rec = self.min_max(completed_recs, key=lambda rec: rec.result)
                self.best_rec_id = best_rec.id
                logger.debug(
                    f"Updated best_rec_id to {self.best_rec_id} with "
                    f"{self.metric_key}={best_rec.result}"
                )
            except Exception as e:
                logger.error("Exception while updating best_rec_id: %s", str(e))

        # Add best experiment id (always, not just at the end)
        if self.best_rec_id != -1:
            result_dict["Best experiment id"] = self.best_rec_id

        update_automl_stats(self.automl_context.id, result_dict)

    def find_best_model(self):
        """Find best model based on metric value chosen and move those artifacts to best_model folder"""
        report_health_beat(self.automl_context.id, "Finding best model")

        logger.info("Finding best recommendation config")
        try:
            best_mAP = self.min_max(self.recommendations, key=lambda rec: rec.result).result
        except Exception as e:
            logger.error("Exception thrown in find_best_model is %s", str(e))
            best_mAP = 0.0
            return -1

        logger.info("Best metric value %s", best_mAP)
        for rec in self.recommendations:
            # Report health beat while processing each recommendation
            report_health_beat(self.automl_context.id, f"Checking experiment {rec.id} for best model")
            logger.info("\nRecommendation in function find_best_model %s", rec)
            job_name = rec.job_id
            if not job_name:
                continue
            expt_folder = os.path.join("/results", rec.job_id)
            checkpoint_files = get_file_list_from_cloud_storage(self.decrypted_workspace_metadata, expt_folder)

            # Use network config for filtering if available
            if self._uses_folder_lookup():
                checkpoint_files = filter_files(checkpoint_files, network_name=self.network)
                # For folder lookup, we want to find the folder containing checkpoints
                if checkpoint_files:
                    # Get unique folder paths
                    checkpoint_folders = list(set(os.path.dirname(f) for f in checkpoint_files))
                    checkpoint_files = checkpoint_folders
            else:
                regex_pattern = r'^(?!.*lightning_logs).*\.(pth|tlt|hdf5)$'
                checkpoint_files = filter_files(checkpoint_files, regex_pattern)
            logger.info("Experiment folder %s", expt_folder)
            logger.info("Checkpoints in find best_model %s", checkpoint_files)

            if checkpoint_files and (rec.status == JobStates.success and rec.result == best_mAP):
                cloud_best_model_folder = f"/results/{self.automl_context.id}"
                logger.info("cloud_best_model_folder %s chosen for rec %s", cloud_best_model_folder, rec.id)

                # Clean up invalid checkpoint folders before moving
                self.delete_checkpoint_files(expt_folder, rec, filter_by_format=True)

                report_health_beat(
                    self.automl_context.id,
                    f"Moving best model folder for experiment {rec.id} to {cloud_best_model_folder}"
                )
                # Move folder but exclude log files - experiment logs should stay in experiment folder
                # The brain job has its own logs, experiment logs should not be moved to brain folder
                exclude_log_files = ['microservices_log.txt', 'log.txt']
                logger.info(f"Moving best experiment folder excluding log files: {exclude_log_files}")
                self.cs_instance.move_folder(
                    expt_folder[1:],
                    cloud_best_model_folder,
                    job_id=self.automl_context.id,
                    exclude_files=exclude_log_files
                )
                report_health_beat(
                    self.automl_context.id,
                    f"Completed moving best model folder for experiment {rec.id}"
                )
                best_specs = get_job_specs(job_name, automl=True, automl_experiment_id=str(rec.id))
                save_automl_best_rec_info(self.automl_context.id, rec.id, rec.job_id)
                save_job_specs(self.automl_context.id, specs=best_specs, automl=True, automl_experiment_id="-1")
                (find_trained_tlt,
                 find_trained_hdf5,
                 find_trained_pth,
                 _,
                 find_trained_safetensors) = self.get_checkpoint_paths_matching_epoch_number(
                    cloud_best_model_folder,
                    rec.id
                )
                if find_trained_tlt or find_trained_hdf5 or find_trained_pth or find_trained_safetensors:
                    self.best_model_copied = True
                    return rec.id
                logger.info("Best model checkpoints couldn't be moved")
                return -1
        return -1

    def get_checkpoint_paths_matching_epoch_number(self, path, rec_id):
        """Get checkpoints from cloud_path and filter based on epoch number

        For networks with folder-based checkpoints (like cosmos-rl), this will find
        folders for different formats (.pth, .safetensors) separately.
        """
        checkpoint_files = get_file_list_from_cloud_storage(self.decrypted_workspace_metadata, path)
        format_epoch_number = format_epoch(self.network, self.best_epoch_number[rec_id])
        if self._uses_folder_lookup():
            # For folder lookup, look for epoch-numbered folders (e.g., epoch_1/, epoch_2/)
            if self.network in MISSING_EPOCH_FORMAT_NETWORKS:
                folder_pattern = fr".*{self.checkpoint_delimiter}{format_epoch_number}/"
            else:
                folder_pattern = fr".*epoch_{format_epoch_number}(?:_step_\d+)?/"

            # Find files inside the epoch folder to identify the folder exists
            epoch_folder_files = [f for f in checkpoint_files if re.match(folder_pattern, f)]

            if epoch_folder_files:
                # For cosmos-rl and similar networks, try to find both safetensors and pth folders
                safetensors_folder = self._select_best_epoch_folder(
                    epoch_folder_files, checkpoint_files, specific_format="safetensors"
                )
                pth_folder = self._select_best_epoch_folder(
                    epoch_folder_files, checkpoint_files, specific_format="pth"
                )

                # If specific formats not found, fall back to default format
                if not safetensors_folder and not pth_folder:
                    selected_folder = self._select_best_epoch_folder(
                        epoch_folder_files, checkpoint_files
                    )
                    if selected_folder:
                        return (
                            [selected_folder], [selected_folder], [selected_folder],
                            [selected_folder], [selected_folder]
                        )
                    return [], [], [], [], []

                # Return format-specific folders
                return (
                    [safetensors_folder] if safetensors_folder else [],  # tlt
                    [safetensors_folder] if safetensors_folder else [],  # hdf5
                    [pth_folder] if pth_folder else [],  # pth
                    [safetensors_folder] if safetensors_folder else [],  # ckzip
                    [safetensors_folder] if safetensors_folder else []   # safetensors
                )
            logger.info("No epoch folder found for pattern: %s", folder_pattern)
            return [], [], [], [], []
        # Traditional epoch-based filtering for files
        if self.network in MISSING_EPOCH_FORMAT_NETWORKS:
            regex_pattern = fr".*{self.checkpoint_delimiter}{format_epoch_number}"
        else:
            regex_pattern = fr".*epoch_{format_epoch_number}(?:_step_\d+)?"
        find_trained_tlt = filter_files(checkpoint_files, regex_pattern=fr'.*{regex_pattern}\.tlt$')
        find_trained_hdf5 = filter_files(checkpoint_files, regex_pattern=fr'.*{regex_pattern}\.hdf5$')
        find_trained_pth = filter_files(checkpoint_files, regex_pattern=fr'.*{regex_pattern}\.pth$')
        find_trained_ckzip = filter_files(checkpoint_files, regex_pattern=fr'.*{regex_pattern}\.ckzip$')
        find_trained_safetensors = filter_files(checkpoint_files, regex_pattern=fr'.*{regex_pattern}\.safetensors$')
        return find_trained_tlt, find_trained_hdf5, find_trained_pth, find_trained_ckzip, find_trained_safetensors

    def get_best_checkpoint_path(self, path, recommendation, filter_by_format=False):
        """Get the path to the best checkpoint.

        Args:
            path: Path to search for checkpoints
            recommendation: Recommendation object containing experiment info
            filter_by_format: If True and using folder lookup, only save the configured checkpoint format.
                            Used in best model workflow to clean up non-matching formats.

        Returns:
            None: Updates internal checkpoint path mapping
        """
        self.ckpt_path[path] = {}
        format_epoch_number = format_epoch(self.network, self.best_epoch_number[recommendation.id])
        recommendation.best_epoch_number = format_epoch_number
        self.save_state()

        if self._uses_folder_lookup():
            logger.info(
                "Using folder-based checkpoint lookup for epoch %s at %s",
                recommendation.best_epoch_number, path
            )
        else:
            logger.info("Best epoch number %s %s", recommendation.best_epoch_number, path)
        (find_trained_tlt,
         find_trained_hdf5,
         find_trained_pth,
         find_trained_ckzip,
         find_trained_safetensors) = self.get_checkpoint_paths_matching_epoch_number(
            path, recommendation.id
        )

        # Check if a specific checkpoint format is configured
        checkpoint_format = self._get_checkpoint_format()

        # Map format names to their corresponding found paths
        format_map = {
            "tlt": find_trained_tlt,
            "hdf5": find_trained_hdf5,
            "pth": find_trained_pth,
            "ckzip": find_trained_ckzip,
            "safetensors": find_trained_safetensors
        }

        if filter_by_format and checkpoint_format and self._uses_folder_lookup():
            # Only save the configured checkpoint format for folder-based checkpoints
            logger.info("Filtering: saving only %s format to ckpt_path", checkpoint_format)
            if checkpoint_format in format_map and format_map[checkpoint_format]:
                self.ckpt_path[path][checkpoint_format] = format_map[checkpoint_format][0]
        else:
            # Save all available formats (default behavior)
            if find_trained_tlt:
                self.ckpt_path[path]["tlt"] = find_trained_tlt[0]
            if find_trained_hdf5:
                self.ckpt_path[path]["hdf5"] = find_trained_hdf5[0]
            if find_trained_pth:
                self.ckpt_path[path]["pth"] = find_trained_pth[0]
            if find_trained_ckzip:
                self.ckpt_path[path]["ckzip"] = find_trained_ckzip[0]
            if find_trained_safetensors:
                self.ckpt_path[path]["safetensors"] = find_trained_safetensors[0]

    def delete_checkpoint_files(self, path, rec, filter_by_format=False):
        """Remove the extra checkpoints generated after the on_cancel_automl_job"""
        report_health_beat(
            self.automl_context.id,
            f"Starting checkpoint cleanup for experiment {rec.id}"
        )
        if not os.getenv("CI_PROJECT_DIR", None):
            time.sleep(30)  # Mounted paths can take time to reflect files generated on remote locally
        trained_files = get_file_list_from_cloud_storage(self.decrypted_workspace_metadata, path)

        regex_pattern = r'.*\.(tlt|hdf5|pth|ckzip|safetensors|resume|lightning_logs)$'

        trained_files = filter_files(trained_files, regex_pattern)
        logger.info("Available checkpoints in delete_checkpoint_files function %s", trained_files)
        self.get_best_checkpoint_path(path, rec, filter_by_format=filter_by_format)
        logger.info("self.ckpt_path in delete_checkpoint_files function %s", self.ckpt_path)
        logger.info("RETAIN_CHECKPOINTS_FOR_RESUME setting: %s", self.retain_checkpoints_for_resume)

        for files in trained_files:
            should_delete = True

            if self._uses_folder_lookup():
                # For folder-based checkpointing, check if file has prefix of any checkpoint folder
                for checkpoint_folder in self.ckpt_path[path].values():
                    if files.startswith(checkpoint_folder + "/"):
                        should_delete = False
                        break
            else:
                # For file-based checkpointing, use exact match
                if files in self.ckpt_path[path].values():
                    should_delete = False

            if should_delete:
                logger.info("Removing item in delete_checkpoint_files function %s", files)
                if self.cs_instance.is_file(files):
                    logger.info("Removing file in delete_checkpoint_files function %s", files)
                    self.cs_instance.delete_file(files)
                    report_health_beat(
                        self.automl_context.id,
                        f"Deleting checkpoint file {files} for experiment {rec.id}"
                    )
                elif self._uses_folder_lookup():
                    logger.info("Removing folder in delete_checkpoint_files function %s", files)
                    self.cs_instance.delete_folder(files[1:])
                    report_health_beat(
                        self.automl_context.id,
                        f"Deleting checkpoint file {files} for experiment {rec.id}"
                    )

    def delete_not_best_model_checkpoints(self, path, rec, flag):
        """Remove the checkpoints which don't correspond to the best result"""
        try:
            valid_recs = [r for r in self.recommendations
                          if r.status in (JobStates.success, JobStates.failure)]

            if not valid_recs:
                # No completed experiments yet
                logger.warning(
                    "No completed experiment results available yet for best model selection. "
                    "All experiments are still pending or running."
                )
                best_mAP = 0.0
            else:
                # Find the best result from completed recommendations
                best_mAP = self.min_max(valid_recs, key=lambda rec: rec.result).result
        except Exception as e:
            logger.error("Exception thrown in delete_not_best_model_checkpoints is %s", str(e))
            best_mAP = 0.0

        logger.info("delete_not_best_model_checkpoints function arguments %s %s %s", path, rec, flag)
        if rec.result != best_mAP or bool(flag):
            trained_files = get_file_list_from_cloud_storage(self.decrypted_workspace_metadata, path)
            regex_pattern = r'.*(?:lightning_logs|events).*$|.*\.(tlt|hdf5|pth|ckzip|safetensors|resume)$'
            trained_files = filter_files(trained_files, regex_pattern)
            logger.info("Available checkpoints in delete_not_best_model_checkpoints function %s", trained_files)
            for files in trained_files:
                if self.cs_instance.is_file(files):
                    logger.info("Removing files in delete_not_best_model_checkpoints function %s", files)
                    self.cs_instance.delete_file(files)
        else:
            flag = True
        return flag
