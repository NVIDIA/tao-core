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

"""LoRA-specific AutoML utilities"""
import logging
logger = logging.getLogger(__name__)

VISION_MODEL_LORA_MODULES = [
    "attn.qkv", "attn.proj"
]


def apply_lora_constraints(parent_params: dict, selected_items: list):
    """Apply cosmos+LoRA constraint to target_modules valid_options, then use standard subset_list logic.

    Since modules_to_save is processed first (parameter sorting), we know its value.
    Simple rule: If modules_to_save has "visual", exclude vision modules from valid_options.
    """
    modules_to_save = parent_params.get("modules_to_save")
    if modules_to_save and "visual" in modules_to_save:
        # Remove vision modules from valid options
        selected_items = [opt for opt in selected_items if opt not in VISION_MODEL_LORA_MODULES]

    return selected_items
