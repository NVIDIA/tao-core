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

"""Defines a dictionary mapping docker image to an internal tag"""
import os

DOCKER_IMAGE_MAPPER = {
    "MAXINE_DLDK": os.getenv('IMAGE_MAXINE_DLDK', default='nvcr.io/0544357712065245/maxine-dldk-trainer_ram:latest'),
    "MAXINE_DEPLOY": os.getenv(
        'IMAGE_MAXINE_DEPLOY',
        default='nvcr.io/0544357712065245/maxine-dldk-trainer_ram:latest'
    ),
    "TAO_PYTORCH": os.getenv('IMAGE_TAO_PYTORCH', default='nvcr.io/nvidia/tao/tao-toolkit:6.0.0-pyt'),
    "TAO_DEPLOY": os.getenv('IMAGE_TAO_DEPLOY', default='nvcr.io/nvidia/tao/tao-toolkit:6.0.0-deploy'),
    "": os.getenv('IMAGE_DEFAULT', default='nvcr.io/nvidia/tao/tao-toolkit:6.0.0-pyt'),  # Default
    "API": os.getenv('IMAGE_API', default='nvcr.io/nvidia/tao/tao-toolkit:6.0.0-api'),
    "TAO_DS": os.getenv('IMAGE_TAO_DS', default='nvcr.io/nvidia/tao/tao-toolkit:6.0.0-data-services'),
    "COSMOS_RL": os.getenv('IMAGE_COSMOS_RL', default='nvcr.io/nvstaging/tao/cosmos_rl_ram_dev:latest'),
    "VILA": os.getenv('IMAGE_VILA', default='nvcr.io/nvidia/tao/tao-toolkit:6.0.0-vila'),
    "tensorboard": os.getenv('IMAGE_TAO_PYTORCH', default='nvcr.io/nvidia/tao/tao-toolkit:6.0.0-pyt')
}


DOCKER_IMAGE_VERSION = {  # (Release tao version for DNN framework, Overriden version for this model)
    "action_recognition": ("6.25.7", "6.0.0"),
    "depth_net_mono": ("6.25.10", "6.25.10"),
    "depth_net_stereo": ("6.25.10", "6.25.10"),
    "bevfusion": ("6.25.7", "6.0.0"),
    "centerpose": ("6.25.7", "6.0.0"),
    "classification_pyt": ("6.25.7", "6.0.0"),
    "deformable_detr": ("6.25.7", "6.0.0"),
    "dino": ("6.25.7", "6.0.0"),
    "image": ("6.25.7", "6.0.0"),
    "mae": ("6.25.7", "6.0.0"),
    "mal": ("6.25.7", "6.0.0"),
    "nvdinov2": ("6.25.7", "6.0.0"),
    "ml_recog": ("6.25.7", "6.0.0"),
    "object_detection": ("6.25.7", "6.0.0"),
    "ocdnet": ("6.25.7", "6.0.0"),
    "ocrnet": ("6.25.7", "6.0.0"),
    "optical_inspection": ("6.25.7", "6.0.0"),
    "pointpillars": ("6.25.7", "6.0.0"),
    "pose_classification": ("6.25.7", "6.0.0"),
    "re_identification": ("6.25.7", "6.0.0"),
    "rtdetr": ("6.25.7", "6.0.0"),
    "segformer": ("6.25.7", "6.0.0"),
    "sparse4d": ("6.25.7", "6.25.7"),
    "stylegan_xl": ("6.25.7", "6.0.0"),
    "visual_changenet_classify": ("6.25.7", "6.0.0"),
    "visual_changenet_segment": ("6.25.7", "6.0.0"),
}
