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

"""AutoML main handler"""
import argparse
import traceback
import json
import logging
import os

from nvidia_tao_core.microservices.automl.controller import Controller
from nvidia_tao_core.microservices.automl.bayesian import Bayesian
from nvidia_tao_core.microservices.automl.hyperband import HyperBand
from nvidia_tao_core.microservices.automl.params import generate_hyperparams_to_search
from nvidia_tao_core.microservices.utils.handler_utils import JobContext
from nvidia_tao_core.microservices.utils.stateless_handler_utils import (
    update_job_status,
    update_job_metadata,
    get_job_specs
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
logger.info(f"Logging configured at level: {TAO_LOG_LEVEL}")


def automl_start(
    root,
    network,
    jc,
    resume,
    automl_algorithm,
    automl_max_recommendations,
    automl_delete_intermediate_ckpt,
    automl_R,
    automl_nu,
    metric,
    epoch_multiplier,
    automl_hyperparameters,
    override_automl_disabled_params,
    decrypted_workspace_metadata
):
    """Starts the automl controller"""
    parameters, parameter_names = generate_hyperparams_to_search(
        jc,
        automl_hyperparameters,
        "/".join(root.split("/")[0:-2]),
        override_automl_disabled_params
    )

    # Check if automl algorithm is valid for specific use-cases
    if network == "classification_pyt":
        if "model.head.type" in parameter_names and automl_algorithm == "hyperband":
            error_message = (
                "Hyperband not supported when non-epoch based models are chosen. "
                "Change algorithm to bayesian"
            )
            result = {"message": error_message}
            update_job_metadata(jc.handler_id, jc.id, metadata_key="job_details", data=result, kind="experiments")
            raise ValueError(error_message)

    logger.debug(
        f"[AUTOML-START] AutoML starting: job_id={jc.id}, resume={resume}, "
        f"algorithm={automl_algorithm}, network={network}"
    )
    if resume:
        logger.debug(f"[AUTOML-START] Entering RESUME path: job_id={jc.id}, algorithm={automl_algorithm}")
        if automl_algorithm.lower() in ("hyperband", "h"):
            brain = HyperBand.load_state(
                job_context=jc,
                root=root,
                network=network,
                parameters=parameters,
                R=int(automl_R),
                nu=int(automl_nu),
                epoch_multiplier=int(epoch_multiplier)
            )
        elif automl_algorithm.lower() in ("bayesian", "b"):
            brain = Bayesian.load_state(jc, root, network, parameters)
        else:
            raise ValueError(f"AutoML Algorithm {automl_algorithm} is not valid")
        controller = Controller.load_state(
            root,
            network,
            brain,
            jc,
            automl_max_recommendations,
            automl_delete_intermediate_ckpt,
            metric,
            automl_algorithm.lower(),
            decrypted_workspace_metadata
        )
        logger.debug(f"[AUTOML-START] Resume path completed, starting controller: job_id={jc.id}")
        controller.start()

    else:
        logger.debug(f"[AUTOML-START] Entering NEW (non-resume) path: job_id={jc.id}, algorithm={automl_algorithm}")
        if automl_algorithm.lower() in ("hyperband", "h"):
            brain = HyperBand(
                job_context=jc,
                root=root,
                network=network,
                parameters=parameters,
                R=int(automl_R),
                nu=int(automl_nu),
                epoch_multiplier=int(epoch_multiplier)
            )
        elif automl_algorithm.lower() in ("bayesian", "b"):
            brain = Bayesian(jc, root, network, parameters)
        else:
            raise ValueError(f"AutoML Algorithm {automl_algorithm} is not valid")
        controller = Controller(
            root,
            network,
            brain,
            jc,
            automl_max_recommendations,
            automl_delete_intermediate_ckpt,
            metric,
            automl_algorithm.lower(),
            decrypted_workspace_metadata
        )
        controller.start()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='AutoML controller', description='Run AutoML.')
    parser.add_argument(
        '--user_id',
        type=str,
    )
    parser.add_argument(
        '--org_name',
        type=str,
    )
    parser.add_argument(
        '--name',
        type=str,
    )
    parser.add_argument(
        '--root',
        type=str,
    )
    parser.add_argument(
        '--automl_job_id',
        type=str,
    )
    parser.add_argument(
        '--network',
        type=str,
    )
    parser.add_argument(
        '--experiment_id',
        type=str,
    )
    parser.add_argument(
        '--resume',
        type=str,
    )
    parser.add_argument(
        '--automl_algorithm',
        type=str,
    )
    parser.add_argument(
        '--automl_max_recommendations',
        type=str,
    )
    parser.add_argument(
        '--automl_delete_intermediate_ckpt',
        type=str,
    )
    parser.add_argument(
        '--automl_R',
        type=str,
    )
    parser.add_argument(
        '--automl_nu',
        type=str,
    )
    parser.add_argument(
        '--metric',
        type=str,
    )
    parser.add_argument(
        '--epoch_multiplier',
        type=str,
    )
    parser.add_argument(
        '--automl_hyperparameters',
        type=str,
    )
    parser.add_argument(
        '--override_automl_disabled_params',
        type=str,
    )
    parser.add_argument(
        '--retain_checkpoints_for_resume',
        type=str,
        default='False'
    )
    parser.add_argument(
        '--timeout_minutes',
        type=str,
        default='60'
    )
    parser.add_argument(
        '--decrypted_workspace_metadata',
        type=json.loads,
    )

    parser.add_argument(
        '--platform_id',
        type=str,
    )

    args = parser.parse_args()
    automl_job_id = args.automl_job_id
    handler_id = args.experiment_id
    network = args.network
    user_id = args.user_id
    org_name = args.org_name
    try:
        root = args.root
        name = args.name
        platform_id = args.platform_id
        specs = get_job_specs(automl_job_id)

        # Get retain_checkpoints_for_resume from CLI argument
        retain_checkpoints_for_resume = args.retain_checkpoints_for_resume.lower() in ("true", "1")
        timeout_minutes = int(args.timeout_minutes)

        from nvidia_tao_core.microservices.utils.handler_utils import get_num_gpus_from_spec
        num_gpu = get_num_gpus_from_spec(specs, "train", network=network, default=-1)
        logger.debug(
            f"[AUTOML-START] AutoML brain job {automl_job_id}: num_gpu from spec = {num_gpu}, "
            f"NUM_GPU_PER_NODE={os.getenv('NUM_GPU_PER_NODE', '0')}"
        )

        jc = JobContext(
            automl_job_id,
            None,
            network,
            "train",
            handler_id,
            user_id,
            org_name,
            "experiment",
            name=name,
            num_gpu=num_gpu,
            platform_id=platform_id,
            specs=specs,
            retain_checkpoints_for_resume=retain_checkpoints_for_resume,
            timeout_minutes=timeout_minutes
        )
        resume = args.resume == "True"
        logger.debug(
            f"[AUTOML-START-ARGS] Parsed resume argument: args.resume='{args.resume}', "
            f"resume={resume}, job_id={automl_job_id}"
        )
        automl_algorithm = args.automl_algorithm
        automl_max_recommendations = args.automl_max_recommendations
        automl_delete_intermediate_ckpt = args.automl_delete_intermediate_ckpt
        automl_R = args.automl_R
        automl_nu = args.automl_nu
        metric = args.metric
        epoch_multiplier = args.epoch_multiplier
        # Parse automl_hyperparameters - normalized to JSON by handler
        automl_hyperparameters = json.loads(args.automl_hyperparameters)
        override_automl_disabled_params = args.override_automl_disabled_params == "True"
        decrypted_workspace_metadata = args.decrypted_workspace_metadata
        automl_start(
            root=root,
            network=network,
            jc=jc,
            resume=resume,
            automl_algorithm=automl_algorithm,
            automl_max_recommendations=automl_max_recommendations,
            automl_delete_intermediate_ckpt=automl_delete_intermediate_ckpt,
            automl_R=automl_R,
            automl_nu=automl_nu,
            metric=metric,
            epoch_multiplier=epoch_multiplier,
            automl_hyperparameters=automl_hyperparameters,
            override_automl_disabled_params=override_automl_disabled_params,
            decrypted_workspace_metadata=decrypted_workspace_metadata)

    except Exception:
        logger.error("AutoML start for network %s failed due to exception %s", network, traceback.format_exc())
        update_job_status(handler_id, automl_job_id, status="Error", kind="experiments")
