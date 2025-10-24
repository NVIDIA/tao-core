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

from nvidia_tao_core.microservices.automl.utils import Recommendation, ResumeRecommendation, JobStates
from nvidia_tao_core.microservices.utils import get_monitoring_metric
from nvidia_tao_core.microservices.constants import (
    _ITER_MODELS,
    NO_VAL_METRICS_DURING_TRAINING_NETWORKS,
    MISSING_EPOCH_FORMAT_NETWORKS
)
if os.getenv("BACKEND") == "NVCF":
    from nvidia_tao_core.microservices.dgx_controller import overwrite_job_logs_from_bcp
from nvidia_tao_core.microservices.handlers.cloud_storage import create_cs_instance_with_decrypted_metadata
from nvidia_tao_core.microservices.handlers.utilities import (
    StatusParser,
    get_total_epochs,
    get_file_list_from_cloud_storage,
    filter_files,
    format_epoch,
    get_network_config
)
from nvidia_tao_core.microservices.handlers.stateless_handlers import (
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
    update_automl_stats
)
from nvidia_tao_core.microservices.job_utils.automl_job_utils import (
    on_new_automl_job,
    on_delete_automl_job,
    on_cancel_automl_job
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
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

        self.root = root
        self.network = network
        self.checkpoint_delimiter = ""
        if self.network in MISSING_EPOCH_FORMAT_NETWORKS:
            self.checkpoint_delimiter = "_"
        self.completed_recommendations = 0
        self.max_recommendations = int(max_recommendations)
        self.delete_intermediate_ckpt = bool(delete_intermediate_ckpt)
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

    def _get_checkpoint_config(self):
        """Get checkpoint config from network config"""
        network_config = get_network_config(self.network)
        return network_config.get("checkpoint", {})

    def _uses_folder_lookup(self):
        """Check if network config specifies folder lookup method"""
        checkpoint_config = self._get_checkpoint_config()
        return checkpoint_config.get("folder", "false") == "true"

    def _get_checkpoint_format(self):
        """Get checkpoint format from network config"""
        checkpoint_config = self._get_checkpoint_config()
        return checkpoint_config.get("format", "")

    def _select_best_epoch_folder(self, epoch_folder_files, all_files):
        """Select the best epoch folder that contains the required checkpoint format"""
        checkpoint_format = self._get_checkpoint_format()

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
        for rec in self.recommendations:
            job_name = rec.job_id
            logger.info("Deleting %s", job_name)
            if not job_name:
                continue
            if not os.getenv("CI_PROJECT_DIR", None):
                logger.info("Cancelling automl job at end of controller %s", job_name)
                on_cancel_automl_job(rec.job_id)
        on_delete_automl_job(self.automl_context.id)

    def start(self):
        """Starts the automl controller"""
        try:
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

        except Exception:
            result_metadata = get_handler_job_metadata(self.automl_context.id)
            result_metadata["job_details"][self.automl_context.id] = {
                "detailed_status": {
                    "message": "AutoML train failed due to run-time exception",
                    "status": "FAILURE"
                }
            }
            write_job_metadata(self.automl_context.id, result_metadata)
            self.cancel_recommendation_jobs()
            logger.error(
                "AutoMLpipeline loop for network %s failed due to exception %s",
                self.network, traceback.format_exc()
            )
            update_job_status(
                self.automl_context.handler_id,
                self.automl_context.id,
                status="Error",
                kind="experiments"
            )

    def save_state(self):
        """Save the self.recommendations into automl brain DB"""
        recs_dict = [ele.__dict__ for ele in self.recommendations]
        metadata = get_handler_job_metadata(self.automl_context.id)
        current_status = metadata.get("status", "")
        if current_status not in ("canceled", "canceling"):
            save_automl_controller_info(self.automl_context.id, recs_dict)

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

        for rec_dict in recs_dict:
            rec = Recommendation(rec_dict["id"], rec_dict["specs"], ctrl.metric_key)
            rec.update_result(rec_dict["result"])
            rec.update_status(rec_dict["status"])
            rec.assign_job_id(rec_dict["job_id"])
            ctrl.recommendations.append(rec)
            ctrl.best_epoch_number[rec_dict["id"]] = (
                rec_dict.get("best_epoch_number") if rec_dict.get("best_epoch_number") else 0
            )

        # Handle temp_rec
        # temp_rec is a recommendation that started, but never ended
        # Usually, if the controller is stopped before a recommendation is done,
        # it might have to be started / resumed again
        temp_rec = get_automl_current_rec(automl_context.id)
        # if ctrl.recommendations[temp_rec].status != JobStates.canceled:
        #     ctrl.recommendations[temp_rec].update_status(JobStates.success)
        ctrl.save_state()
        if ctrl.recommendations[temp_rec].status == JobStates.canceled:
            logger.info("Resuming stopped automl sub-experiment %s", temp_rec)
            if ctrl.automl_algorithm == "hyperband":
                ctrl.brain.track_id = temp_rec
            ctrl.on_new_automl_job(ctrl.recommendations[temp_rec])

        return ctrl

    def _execute_loop(self):
        """A loop that does the 3 things in order

        1.See if any new recommendation is up to execute
        2.Reads results of newly done experiments
        3.Writes AutoML status into a file which can be shown to the end user
        """
        update_job_status(self.automl_context.handler_id, self.automl_context.id, status="Running", kind="experiments")
        while True:
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
                            self.delete_not_best_model_checkpoints(expt_root, rec, True)
                        handler_metadata = get_handler_metadata(self.automl_context.handler_id, "experiments")
                        handler_metadata["checkpoint_epoch_number"][f"best_model_{self.automl_context.id}"] = (
                            self.best_epoch_number[self.best_rec_id]
                        )
                        handler_metadata["checkpoint_epoch_number"][f"latest_model_{self.automl_context.id}"] = (
                            self.best_epoch_number[self.best_rec_id]
                        )
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
        if self.automl_algorithm in ("bayesian", "b") and len(self.recommendations) == self.max_recommendations:
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
                self.on_new_automl_job(rec)

            elif type(spec) is ResumeRecommendation:
                self.hyperband_cancel_condition_seen = False
                rec_id = spec.id
                self.best_epoch_number[rec_id] = 0
                # Save brain state and update current recommendation
                self.brain.save_state()
                # update temp_rec
                save_automl_current_rec(self.automl_context.id, rec_id)
                assert (self.recommendations[rec_id].id == rec_id), (
                    f"Recommendation ID mismatch: expected {rec_id} but got "
                    f"{self.recommendations[rec_id].id}"
                )
                self.recommendations[rec_id].specs = spec.specs.copy()
                self.recommendations[rec_id].update_status(JobStates.pending)

                # Remove previous files (except checkpoints) from experiment folder.
                def remove_files(local_expt_path, cloud_expt_path, rec_id):
                    expt_file_name = get_file_list_from_cloud_storage(
                        self.decrypted_workspace_metadata, cloud_expt_path)
                    regex_pattern = r'.*(?:lightning_logs|events).*$|.*\.(json)$'
                    expt_file_name = filter_files(expt_file_name, regex_pattern)
                    for file_name in expt_file_name:
                        self.cs_instance.delete_file(file_name)
                    if os.path.exists(local_expt_path):
                        expt_file_name = glob.glob(local_expt_path + "/**/*.txt", recursive=True)
                        logger.info("Removing log files: %s", expt_file_name)
                        for file_name in expt_file_name:
                            if os.path.isfile(file_name):
                                os.remove(file_name)
                    delete_dnn_status(self.automl_context.id, automl=True, experiment_number=str(rec_id))
                expt_name = "experiment_" + str(rec_id)
                remove_files(
                    os.path.join(self.root, expt_name),
                    os.path.join("/results", self.recommendations[rec_id].job_id),
                    rec_id
                )

                self.save_state()
                self.on_new_automl_job(self.recommendations[rec_id])

    def read_results(self):
        """Update results for each recommendation"""
        flag = False
        for rec in self.recommendations:
            old_status = rec.status

            job_name = rec.job_id
            if not job_name:
                continue

            expt_name = "experiment_" + str(rec.id)
            local_expt_root = os.path.join(self.root, expt_name)
            cloud_expt_root = os.path.join("/results", rec.job_id)

            # If rec already changed to Success, no need to check
            if rec.status in [JobStates.success, JobStates.failure]:
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
            self.calculate_eta(new_results, rec.job_id)
            metadata = get_handler_job_metadata(self.automl_context.id)
            results = metadata.get("job_details", {})
            new_results = status_parser.update_results(
                experiment_number=str(rec.id),
                total_epochs=self.total_epochs,
                eta=self.eta,
                last_seen_epoch=self.total_epochs - self.remaining_epochs_in_experiment,
                automl=True,
                job_id=self.automl_context.id,
                previous_result_metadata=results,
                automl_brain=True
            )
            new_results = status_parser.update_results(
                experiment_number=str(rec.id),
                total_epochs=self.total_epochs,
                last_seen_epoch=self.total_epochs - self.remaining_epochs_in_experiment,
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

            validation_map_processed = False
            # Force termination of the case for hyperband training
            if self.automl_algorithm in ("hyperband", "h"):
                brain_dict = get_automl_brain_info(self.automl_context.id)
                if brain_dict:
                    # if the experiment is in the last set of bracket, do not cancel job.
                    for result_key in new_results[rec.job_id].keys():
                        if self.hyperband_cancel_condition_seen or result_key in ("epoch", "cur_iter"):
                            if not isinstance(new_results[rec.job_id].get(result_key, None), type(None)):
                                self.brain_epoch_number = float(brain_dict.get("epoch_number", float('inf')))
                                ni_list = brain_dict.get("ni", [str(float('-inf'))])
                                bracket_key = str(brain_dict.get("bracket", 0))
                                sh_iter = brain_dict.get("sh_iter", float('inf'))
                                if len(ni_list[bracket_key]) != (sh_iter + 1):
                                    if (
                                        self.hyperband_cancel_condition_seen or
                                        new_results[rec.job_id].get(result_key) > self.brain_epoch_number or
                                        (self.network == "pointpillars" and
                                         new_results[rec.job_id].get(result_key) + 1 >= self.brain_epoch_number)
                                    ):
                                        self.hyperband_cancel_condition_seen = True
                                        # Cancel the current running job and change the job state to success
                                        validation_map, self.best_epoch_number[rec.id], _ = status_parser.read_metric(
                                            results=new_results[rec.job_id],
                                            metric=self.metric,
                                            automl_algorithm=self.automl_algorithm,
                                            automl_brain_job_id=self.automl_context.id,
                                            brain_epoch_number=self.brain_epoch_number
                                        )
                                        if validation_map != 0.0:
                                            format_epoch_number = format_epoch(
                                                self.network,
                                                self.best_epoch_number[rec.id]
                                            )
                                            trained_files = get_file_list_from_cloud_storage(
                                                self.decrypted_workspace_metadata,
                                                cloud_expt_root
                                            )

                                            if self._uses_folder_lookup():
                                                # Look for epoch-numbered folders
                                                if self.network in MISSING_EPOCH_FORMAT_NETWORKS:
                                                    folder_pattern = (
                                                        fr".*{self.checkpoint_delimiter}{format_epoch_number}/"
                                                    )
                                                else:
                                                    folder_pattern = (
                                                        fr".*epoch_{format_epoch_number}(?:_step_\d+)?/"
                                                    )

                                                # Find files inside the epoch folder to identify the folder exists
                                                epoch_folder_files = [
                                                    f for f in trained_files if re.match(folder_pattern, f)
                                                ]
                                                if epoch_folder_files:
                                                    selected_folder = self._select_best_epoch_folder(
                                                        epoch_folder_files, trained_files
                                                    )
                                                    trained_files = [selected_folder] if selected_folder else []
                                                else:
                                                    trained_files = []
                                                logger.info("Trained files in read_results function: %s", trained_files)
                                            else:
                                                # Traditional epoch-based filtering
                                                regex_pattern = (
                                                    fr'^(?!.*lightning_logs).*{self.checkpoint_delimiter}'
                                                    fr'{format_epoch_number}\.(pth|tlt|hdf5)$'
                                                )
                                                trained_files = filter_files(trained_files, regex_pattern)
                                            if trained_files:
                                                rec.update_status(JobStates.success)
                                                validation_map_processed = True
                                                self.hyperband_cancel_condition_seen = False
                                                logger.info("Cancelling hyperband automl job %s", rec.job_id)
                                                on_cancel_automl_job(rec.job_id)
                                                self.delete_checkpoint_files(cloud_expt_root, rec)
                                                break

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
                if not validation_map_processed:
                    brain_epoch_number = self.brain_epoch_number
                    if self.automl_algorithm in ("bayesian", "b"):
                        self.brain.num_epochs_per_experiment = get_total_epochs(
                            self.automl_context,
                            os.path.dirname(self.root),
                            automl=True,
                            automl_experiment_id=rec.id
                        )
                        brain_epoch_number = self.brain.num_epochs_per_experiment
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
                        validation_map = float('inf')
                if validation_map != 0.0:
                    rec.update_result(validation_map)
                self.save_state()
                logger.info("Cancelling automl job with status %s and job id %s", status, rec.job_id)
                on_cancel_automl_job(rec.job_id)
            if old_status != status:
                rec.update_status(status)
                self.save_state()
                if status == JobStates.success:
                    container_log_file = f"{self.root}/experiment_{rec.id}/log.txt"
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

    def calculate_eta(self, new_results, rec_job_id):
        """Calculate estimated time remaining for automl job"""
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
                    remaining_epochs = self.brain.num_epochs_per_experiment - current_epoch
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
        # Best mAP seen till now
        result_dict = {}
        try:
            if self.recommendations[-1].result == 0.0:
                best_metric_value = 0.0
                if self.recommendations[:-1]:
                    best_metric_value = self.min_max(self.recommendations[:-1], key=lambda rec: rec.result).result
                result_dict[f"best_{self.metric_key}"] = best_metric_value
            else:
                best_metric_value = 0.0
                if self.recommendations:
                    best_metric_value = self.min_max(self.recommendations, key=lambda rec: rec.result).result
                result_dict[f"best_{self.metric_key}"] = best_metric_value
        except Exception as e:
            logger.error("Exception thrown in write_results is %s", str(e))
            result_dict[f"best_{self.metric_key}"] = 0.0

        if type(self.eta) is float:
            self.eta = str(timedelta(seconds=self.eta))
        result_dict["Estimated time for automl completion"] = str(self.eta)
        result_dict["Current experiment number"] = len(self.recommendations)

        if self.network in _ITER_MODELS:
            result_dict["Number of iters yet to start"] = self.remaining_epochs_in_experiment
            result_dict["Time per iter in seconds"] = round(self.average_time_per_epoch, 2)
        else:
            result_dict["Number of epochs yet to start"] = self.remaining_epochs_in_experiment
            result_dict["Time per epoch in seconds"] = round(self.average_time_per_epoch, 2)

        if final and self.best_rec_id != -1:
            result_dict["Best experiment number"] = self.best_rec_id + 1

        update_automl_stats(self.automl_context.id, result_dict)

    def find_best_model(self):
        """Find best model based on metric value chosen and move those artifacts to best_model folder"""
        logger.info("Finding best recommendation config")
        try:
            best_mAP = self.min_max(self.recommendations, key=lambda rec: rec.result).result
        except Exception as e:
            logger.error("Exception thrown in find_best_model is %s", str(e))
            best_mAP = 0.0
            return -1

        logger.info("Best metric value %s", best_mAP)
        for rec in self.recommendations:
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
                logger.info("cloud_best_model_folder %s", cloud_best_model_folder)

                self.cs_instance.move_folder(expt_folder[1:], cloud_best_model_folder)
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
        """Get checkpoints from cloud_path and filter based on epoch number"""
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
                selected_folder = self._select_best_epoch_folder(
                    epoch_folder_files, checkpoint_files
                )
                if selected_folder:
                    return [selected_folder], [selected_folder], [selected_folder], [selected_folder], [selected_folder]
                return [], [], [], [], []
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

    def get_best_checkpoint_path(self, path, recommendation):
        """Get the path to the best checkpoint.

        Args:
            path: Path to search for checkpoints
            recommendation: Recommendation object containing experiment info

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

    def delete_checkpoint_files(self, path, rec):
        """Remove the extra checkpoints generated after the on_cancel_automl_job"""
        if not os.getenv("CI_PROJECT_DIR", None):
            time.sleep(30)  # Mounted paths can take time to reflect files generated on remote locally
        trained_files = get_file_list_from_cloud_storage(self.decrypted_workspace_metadata, path)

        regex_pattern = r'.*\.(tlt|hdf5|pth|ckzip|safetensors|resume|lightning_logs)$'

        trained_files = filter_files(trained_files, regex_pattern)
        logger.info("Available checkpoints in delete_checkpoint_files function %s", trained_files)
        self.get_best_checkpoint_path(path, rec)
        logger.info("self.ckpt_path in delete_checkpoint_files function %s", self.ckpt_path)
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
                elif self._uses_folder_lookup():
                    logger.info("Removing folder in delete_checkpoint_files function %s", files)
                    self.cs_instance.delete_folder(files[1:])

    def delete_not_best_model_checkpoints(self, path, rec, flag):
        """Remove the checkpoints which don't correspond to the best result"""
        try:
            if self.recommendations[-1].result == 0.0:
                best_mAP = self.min_max(self.recommendations[:-1], key=lambda rec: rec.result).result
            else:
                best_mAP = self.min_max(self.recommendations, key=lambda rec: rec.result).result
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
