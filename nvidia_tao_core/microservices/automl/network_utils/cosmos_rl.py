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

"""Cosmos-RL specific AutoML utilities"""
import logging
import numpy as np

logger = logging.getLogger(__name__)

# Vision model LoRA modules that should be excluded when visual modules are fully tuned
VISION_MODEL_LORA_MODULES = [
    "attn.qkv", "attn.proj"
]

# Common regex patterns for LoRA module matching
COMMON_LORA_PATTERNS = [
    "visual\\..*",      # Visual/vision tower modules
    "attn\\..*",        # Attention modules
    "mlp\\..*",         # MLP modules
    ".*proj",           # Projection layers
    ".*_proj",          # Projection layers with underscore
    "embed.*",          # Embedding layers
]


def apply_lora_constraints(parent_params: dict, selected_items: list):
    """Apply cosmos-rl LoRA constraint to target_modules valid_options.

    Since modules_to_save is processed first (parameter sorting), we know its value.
    Simple rule: If modules_to_save has "visual", exclude vision modules from valid_options.

    Args:
        parent_params: Dictionary of parent parameters already processed
        selected_items: List of selected LoRA target modules

    Returns:
        List of modules after applying constraints, or "all-linear" if list becomes empty
    """
    modules_to_save = parent_params.get("modules_to_save")
    if modules_to_save and "visual" in modules_to_save:
        # Remove vision modules from valid options
        selected_items = [opt for opt in selected_items if opt not in VISION_MODEL_LORA_MODULES]

    # If list becomes empty after filtering, return "all-linear" as a safe fallback
    if selected_items == []:
        logger.info("All target_modules filtered out by constraints. Returning 'all-linear' as fallback.")
        return "all-linear"

    return selected_items


def is_full_sft_training(default_train_spec):
    """Check if this is full SFT training (no LoRA) by checking original train spec.

    Args:
        default_train_spec: The original train specification dictionary

    Returns:
        bool: True if policy.lora is not present in config (full SFT), False otherwise
    """
    # Check if policy.lora exists in the original train spec
    if "policy" not in default_train_spec:
        return False

    policy_config = default_train_spec.get("policy", {})
    # If lora key is not present or is None/empty, it's full SFT
    lora_config = policy_config.get("lora", None)

    is_full_sft = lora_config is None or (isinstance(lora_config, dict) and len(lora_config) == 0)
    logger.info(f"Cosmos-RL training mode: {'Full SFT' if is_full_sft else 'LoRA'}")
    return is_full_sft


def generate_lora_pattern(base_value, param_type="alpha", valid_multipliers=None):
    """Generate random pattern dictionary for LoRA configuration.

    Args:
        base_value: The base value to use as reference
        param_type: Type of parameter - "alpha" (float) or "rank" (int)
        valid_multipliers: List of valid multipliers. If None, uses defaults based on param_type

    Returns:
        dict or None: Dictionary mapping regex patterns to values, or None
    """
    # 50% chance to return None (no pattern override)
    lora_pattern_chance = np.random.random()
    logger.info(f"Lora pattern chance: {lora_pattern_chance}")
    if lora_pattern_chance < 0.5:
        return None

    # Set default multipliers based on parameter type
    if valid_multipliers is None:
        if param_type == "alpha":
            valid_multipliers = [0.5, 1.0, 2.0, 4.0]
        else:  # rank
            valid_multipliers = [0.5, 1.0, 2.0]

    # Randomly select 1-3 patterns
    num_patterns = np.random.randint(1, 4)
    logger.info(f"Number of patterns: {num_patterns}")
    selected_patterns = np.random.choice(COMMON_LORA_PATTERNS, size=num_patterns, replace=False)
    logger.info(f"Selected patterns: {selected_patterns}")
    # Generate pattern values
    pattern_dict = {}
    for pattern in selected_patterns:
        # Convert numpy string to regular Python string
        pattern = str(pattern)
        multiplier = np.random.choice(valid_multipliers)
        value = base_value * multiplier

        if param_type == "alpha":
            # Float values for alpha
            value = float(value)
            value = max(1.0, value)  # Ensure minimum of 1.0
        else:  # rank
            # Integer values for rank, rounded to power of 2
            value = int(value)
            value = max(1, value)  # Ensure minimum of 1
            # Round to nearest power of 2
            if value > 1:
                value = 2 ** round(np.log2(value))

        pattern_dict[pattern] = value

    logger.info(f"Generated {param_type}_pattern: {pattern_dict}")
    return pattern_dict


def apply_optm_lr_logic(parameter_name, lr_value, v_max, default_train_spec, parent_params=None):
    """Apply cosmos-rl specific logic for optm_lr parameter.

    For full SFT training (no LoRA), there's a 30% chance to generate dual learning rates
    [llm_lr, vision_lr] where vision_lr is 5-20x higher than llm_lr.

    Args:
        parameter_name: The parameter name to check
        lr_value: The generated learning rate value
        v_max: Maximum valid value for learning rate
        default_train_spec: The original train specification to check for LoRA presence
        parent_params: Dictionary of already-sampled parameters (optional, not used here)

    Returns:
        Either the original lr_value (float) or a list [llm_lr, vision_lr]
    """
    # Only apply for cosmos-rl optm_lr parameter
    if parameter_name != "train.optm_lr":
        return lr_value

    # Check if this is full SFT training (no LoRA in original config)
    if not is_full_sft_training(default_train_spec):
        return lr_value

    # 80% chance to use dual learning rates for vision-language models
    dual_learning_rates_chance = np.random.random()
    logger.info(f"Dual learning rates chance: {dual_learning_rates_chance}")
    if dual_learning_rates_chance < 0.8:
        # Generate two learning rates: [llm_lr, vision_lr]
        llm_lr = float(lr_value)  # Convert numpy type to Python float
        # Vision LR is typically 5-20x higher than LLM LR
        vision_multiplier = float(np.random.uniform(5.0, 20.0))
        vision_lr = float(llm_lr * vision_multiplier)
        # Don't cap vision_lr to v_max - it needs to be higher for effective vision fine-tuning
        # Only ensure it stays within reasonable bounds (not infinity)
        vision_lr = min(vision_lr, 1.0)  # Cap at 1.0 for safety
        logger.info(f"Generated dual learning rates for Full SFT: LLM={llm_lr:.2e}, Vision={vision_lr:.2e}")
        return [llm_lr, vision_lr]

    return lr_value


def apply_dict_logic(parameter_name, value, parent_params):
    """Apply cosmos-rl specific logic for dict parameters (LoRA patterns).

    Generates alpha_pattern or r_pattern dictionaries based on the base values.

    Args:
        parameter_name: The parameter name to check
        value: The generated parameter value (not used, we generate new dict)
        parent_params: Dictionary of parent parameters to get base values

    Returns:
        Generated pattern dictionary or None
    """
    logger.info(f"Applying dict logic for parameter: {parameter_name}")
    logger.info(f"Value: {value}")
    logger.info(f"Parent params: {parent_params}")
    if "alpha_pattern" in parameter_name:
        # Get the base lora_alpha value
        base_alpha = parent_params.get("policy.lora.lora_alpha", 8)
        return generate_lora_pattern(base_alpha, param_type="alpha")

    if "r_pattern" in parameter_name:
        # Get the base r value
        base_r = parent_params.get("policy.lora.r", 8)
        return generate_lora_pattern(base_r, param_type="rank")

    # For other dict types, 80% chance to return None
    lora_pattern_chance = np.random.random()
    logger.info(f"Lora pattern chance: {lora_pattern_chance}")
    if lora_pattern_chance < 0.8:
        return None
    return {}


# Register handlers for cosmos-rl
HANDLERS = {
    "float": apply_optm_lr_logic,
    "dict": apply_dict_logic,
    "collection": apply_dict_logic,
}
