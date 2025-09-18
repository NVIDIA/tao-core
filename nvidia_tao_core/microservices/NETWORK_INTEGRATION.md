# TAO Network Integration Guide

This guide provides comprehensive documentation for integrating new networks into the TAO Finetuning Microservices (FTMS). The network configuration system allows you to declaratively define how your network interacts with datasets, handles different data formats, and integrates with the TAO ecosystem without requiring code changes to the core infrastructure.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Quick Start](#quick-start)
3. [Configuration File Structure](#configuration-file-structure)
4. [Core Configuration Sections](#core-configuration-sections)
   - [API Parameters](#1-api-parameters-api_params)
   - [Data Sources](#2-data-sources-configuration-data_sources)
   - [Dataset Validation](#3-dataset-validation-dataset_validation)
   - [Dynamic Configuration](#4-dynamic-configuration-dynamic_config)
   - [Additional Downloads](#5-additional-downloads-additional_download)
   - [Upload Strategy](#6-upload-strategy-upload_strategy)
   - [Actions Mapping](#7-actions-mapping-actions_mapping)
   - [Spec Parameters](#8-spec-parameters-spec_params)
   - [AutoML Spec Parameters](#9-automl-spec-parameters-automl_spec_params)
   - [Metrics](#10-metrics-metrics)
5. [Complete Examples](#complete-examples)
6. [Testing and Validation](#testing-and-validation)
7. [Troubleshooting Guide](#troubleshooting-guide)
8. [Best Practices](#best-practices)

## Prerequisites

Before integrating a new network, ensure you have:

### 1. DataClass Configuration
Create a network configuration dataclass in TAO Core repository. Register all configurations supported by all actions within the dataclass Python file. [VILA example](https://gitlab-master.nvidia.com/nvidia-tao-toolkit/tao-core/-/blob/main/nvidia_tao_core/config/vila/default_config.py?ref_type=heads)

**Requirements:**
- The folder name should be the network name which is present in the entrypoint of the docker container
- Dataset paths in the configuration should accept absolute paths - avoid using separate root_dir and relative path fields; pass the entire path to a single config parameter

### 2. Network Configuration File
Create `<network_name>.config.json` under `tao-core-repo/nvidia_tao_core/microservices/handlers/network_configs/`

## Quick Start

To integrate a new network:

1. Create the network configuration JSON file
2. Define basic API parameters (actions, formats, Docker image)
3. Configure data source mappings for your datasets
4. Set up dataset validation rules
5. Add spec parameter mappings
6. Test with a simple dataset

## Overview

The network configuration JSON file defines:

- **API Parameters**: Basic network metadata and supported actions
- **Data Sources**: How to map dataset files to configuration parameters
- **Dataset Validation**: Rules for validating uploaded datasets
- **Dynamic Configuration**: Advanced logic for modifying configurations based on conditions
- **Additional Downloads**: Supplementary files needed beyond main data sources
- **Upload Strategy**: How results are uploaded during job execution
- **Actions Mapping**: Redirecting actions to specialized networks
- **Spec Parameters**: Parameter mappings for experiment specifications
- **AutoML Spec Parameters**: Specialized mappings for AutoML workflows
- **Metrics**: Evaluation metrics and monitoring configuration

## Configuration File Structure

The network configuration file follows a standardized JSON structure with well-defined sections. Each section serves a specific purpose in defining how the network integrates with the FTMS infrastructure. The top-level structure organizes all configuration aspects into logical groups that the handlers can process efficiently.

```json
{
    "api_params": { ... },
    "data_sources": { ... },
    "dataset_validation": { ... },
    "dynamic_config": { ... },
    "additional_download": { ... },
    "upload_strategy": { ... },
    "actions_mapping": { ... },
    "spec_params": { ... },
    "automl_spec_params": { ... },
    "metrics": { ... }
}
```

## Core Configuration Sections

### 1. API Parameters (`api_params`)

The `api_params` section serves as the foundation of your network configuration, defining the basic metadata that the FTMS needs to understand how to work with your network. This section tells the system what types of datasets your network can handle, what actions it supports, what Docker image to use for execution, and how API calls map to internal commands. Think of this as the configuration entry point that introduces your network to the TAO API system.

**Example from object_detection.config.json:** The object detection network demonstrates a comprehensive API parameter configuration with multiple formats and dataset actions, showing how to handle both training and data processing workflows.

```json
{
    "api_params": {
        "dataset_type": "object_detection",
        "actions": ["train", "evaluate", "export", "inference"],
        "formats": ["kitti", "coco", "raw"],
        "accepted_ds_intents": ["training", "evaluation", "testing"],
        "image": "TAO_PYTORCH",
        "spec_backend": "yaml",
        "actions_pipe": {
            "train": "train",
            "evaluate": "evaluate",
            "export": "export_with_spec",
            "inference": "inference"
        }
    }
}
```

#### Required Fields:

- **`dataset_type`**: Type of dataset this network works with  
- **`actions`**: List of supported actions for this network  
- **`formats`**: Supported dataset formats  
- **`accepted_ds_intents`**: Valid dataset usage intents  
- **`image`**: Docker image to use (TAO_PYTORCH, TAO_TF2, MONAI, VILA, etc.)  
- **`spec_backend`**: Configuration format (yaml, json)  
- **`actions_pipe`**: Maps API actions to internal command names

#### Optional Fields:

**Dataset Download Behavior**

**Example from vila.config.json:** Vision-Language Models may require recursive dataset file downloading for complex dataset structures. For normal networks like dino, centerpose, downloading the json file for dataset.train_data_sources.json_file is sufficient, but for vila, this json/yaml file should be downloaded first, and inside this it'll contain the actual dataset paths to be downloaded like videos folder or annotations.json

```json
{
    "api_params": {
        "recursive_dataset_file_download": true
    }
}
```

- **`recursive_dataset_file_download`**: Enable recursive downloading of dataset files

**Docker Image Override per Action**

**Example from sparse4d.config.json:** Some networks need different Docker images for different actions. This allows overriding the main image on a per-action basis.

```json
{
    "api_params": {
        "image": "TAO_PYTORCH",
        "image_override_per_action": {
            "dataset_convert": "TAO_DS"
        }
    }
}
```

- **`image_override_per_action`**: Override Docker image for specific actions

### 2. Data Sources Configuration (`data_sources`)

The `data_sources` section is the heart of your network configuration. It defines the precise mapping between your dataset files and the configuration parameters that your network expects.

This section allows you to declaratively specify how different types of dataset files should be incorporated into your network's configuration without requiring hardcoded path logic in Python functions. Each action (train, evaluate, inference, etc.) can have its own set of data source mappings.

#### Basic Structure

The most fundamental data source mapping connects a single dataset file to a configuration parameter. This pattern is used when you have a straightforward relationship between a dataset file and where it needs to appear in your network's configuration. The `source` field specifies which dataset collection to use, `multiple_sources` indicates whether to process one or many datasets, and `path` specifies the file within the dataset.

```json
{
    "data_sources": {
        "action_name": {
            "config.path": {
                "source": "train_datasets",
                "multiple_sources": false,
                "path": "images.tar.gz"
            }
        }
    }
}
```

#### Source Types

- **`train_datasets`**: Training datasets  
- **`eval_dataset`**: Evaluation dataset  
- **`inference_dataset`**: Inference dataset  
- **`calibration_dataset`**: Calibration dataset  
- **`id`**: If the action is on the dataset itself - like dataset_convert

#### Simple Path Mapping

When you need to map a single file from a dataset directly to a configuration parameter, use this simple pattern. This is the most common use case and handles the majority of dataset-to-config mappings.

The example below shows how to take the `train.tar.gz` file from training datasets and place it at the `dataset.train_dataset_dir` path in your network's configuration.  
    
**Example from action_recognition.config.json:** The action recognition network uses simple path mappings to connect dataset files directly to configuration parameters, demonstrating the most straightforward approach to data source configuration.

```json
{
    "dataset.train_dataset_dir": {
        "source": "train_datasets",
        "multiple_sources": false,
        "path": "train.tar.gz"
    }
}
```

#### Multiple Sources

Some networks need to combine data from multiple datasets into a single configuration parameter. For instance, you might want to train on several datasets simultaneously.

The `multiple_sources: true` setting tells the system to process all datasets from the specified source and create a list of mappings. Each dataset will contribute its files according to the mapping structure you define.

**Example from deformable_detr.config.json:** The Deformable DETR network demonstrates multiple source handling by processing several training datasets and creating structured mappings for each one, allowing training on combined datasets.

```json
{
    "dataset.train_data_sources": {
        "source": "train_datasets",
        "multiple_sources": true,
        "mapping": {
            "image_dir": {
                "path": "images.tar.gz"
            },
            "json_file": {
                "path": "annotations.json"
            }
        }
    }
}
```

#### Complex Mapping with Nested Objects

When your network expects complex nested structures in its configuration, you can define sophisticated mappings that include transformations and multiple file types. The mapping creates a structured object with multiple properties, each potentially sourced from different files within the dataset. Transforms can be applied to individual paths to handle special cases like TAR file processing.

**Example from deformable_detr.config.json:** This example shows how the Deformable DETR network handles complex mappings with transforms, creating structured objects that include both image directories and annotation files with appropriate transformations.


```json
{
    "dataset.train_data_sources": {
        "source": "train_datasets",
        "multiple_sources": true,
        "mapping": {
            "image_dir": {
                "path": "images.tar.gz",
                "transform": "handle_tar_path"
            },
            "annotations_file": {
                "path": "annotations.json"
            }
        }
    }
}
```

  ### Conditional Logic

  #### Format-based paths:

  Different dataset formats require different file paths and structures. Format-based path selection allows your network to automatically choose the correct file based on the dataset's format. This is essential for networks that support multiple input formats like KITTI and COCO. The asterisk (\*) serves as a fallback option when the specific format isn't explicitly handled.


  **Example from object\_detection.config.json:** The object detection network supports multiple annotation formats and automatically selects the appropriate file based on the dataset format, demonstrating robust format handling.


```json
{
    "data.ann_path": {
        "source": "id",
        "multiple_sources": false,
        "path_from_format": {
            "kitti": "labels.tar.gz",
            "coco": "annotations.json",
            "*": "annotations.json"
        }
    }
}
```

  #### Intent-based paths:

  Dataset intent (training, evaluation, testing) often determines which files should be used from a dataset. Intent-based path selection automatically chooses the appropriate file based on how the dataset is intended to be used. This is particularly useful for datasets that contain multiple data splits in different files.


```json
{
    "dataset.train_dataset": {
        "source": "id",
        "multiple_sources": false,
        "path_from_intent": {
            "training": "train.tar.gz",
            "evaluation": "test.tar.gz"
        }
    }
}
```

  #### Type-based paths:

  When the dataset type determines the file structure or naming convention, type-based path selection automatically selects the correct path. This is useful for networks that can work with different types of data (images, point clouds, etc.) where the file organization differs based on the data type.


```json
{
    "config.path": {
        "source": "train_datasets",
        "multiple_sources": false,
        "path_from_type": {
            "segmentation": "masks/",
            "classification": "images/"
        }
    }
}
```

  #### Source-based paths:

  Some networks need different file paths based on which dataset source is being used. This allows different behavior when the same parameter is populated from different dataset types.


  **Example from classification\_tf2.config.json:** The TensorFlow classification network demonstrates source-based path selection for calibration datasets, using different files depending on whether the calibration dataset is the evaluation dataset or another source.


```json
{
    "gen_trt_engine.tensorrt.calibration.cal_image_dir": {
        "source": "calibration_dataset",
        "multiple_sources": false,
        "path_from_source": {
            "eval_dataset": "images_val.tar.gz",
            "*": "images_train.tar.gz"
        }
    }
}
```

  #### Model type-based paths:

  Some networks support multiple model architectures or input modalities, each requiring different data files. Model type-based path selection reads the model type from your network's configuration and selects the appropriate data files accordingly. This eliminates the need for conditional logic in your network code.


  **Example from pose\_classification.config.json:** The pose classification network supports both "nvidia" and "openpose" graph layouts, with each requiring different data file structures. The configuration automatically selects the appropriate paths based on the model type.


```json
{
    "dataset.train_dataset": {
        "source": "train_datasets",
        "multiple_sources": false,
        "path_from_model_type": {
            "nvidia": "nvidia/train_data.npy",
            "openpose": "kinetics/train_data.npy"
        }
    }
}
```

  ### Value from Metadata

  Sometimes you need to set configuration values based on dataset metadata rather than file paths. The `value_from_metadata` feature reads metadata properties from the dataset and maps them to configuration values. This is useful for setting parameters like dataset type, number of classes, or processing modes based on the dataset's characteristics.


  **Example from object\_detection.config.json:** The object detection network demonstrates metadata-based value assignment, automatically setting the dataset type configuration parameter based on the dataset's format metadata.


```json
{
    "data.dataset_type": {
        "source": "id",
        "multiple_sources": false,
        "value_from_metadata": {
            "key": "format",
            "default": "coco",
            "mapping": {
                "kitti": "kitti",
                "coco": "coco",
                "*": "coco"
            }
        }
    }
}
```

  ### Path from Convert Job Specification

  When your network requires paths that depend on the output of dataset conversion jobs, you can use `path_from_convert_job_spec` to dynamically determine paths based on the conversion job's specifications.

  **Example from sparse4d.config.json:** The Sparse4D network uses conversion job specifications to determine the correct split paths based on the conversion job's configuration.

```json
{
    "dataset.data_root": {
        "source": "train_datasets",
        "multiple_sources": false,
        "path_from_convert_job_spec": {
            "spec_path": "aicity.split",
            "mapping": {
                "train": "train",
                "val": "val", 
                "test": "test",
                "*": "train"
            }
        }
    }
}
```

- **`path_from_convert_job_spec`**: Determine path based on dataset conversion job specifications
- **`spec_path`**: Path to the specification value in the conversion job
- **`mapping`**: Maps specification values to actual paths

  ### Static Values

  For configuration parameters that don't depend on dataset files but need to be set to specific values or patterns, use static value assignment. This is particularly useful for output paths, temporary directories, or parameters that incorporate job-specific identifiers. Placeholders like `{job_id}` will be automatically replaced with actual values at runtime.


  **Example from object\_detection.config.json:** The object detection network sets a static output path that incorporates the job ID, demonstrating how to create dynamic paths without relying on dataset files.


```json
{
    "output.path": {
        "source": "id",
        "multiple_sources": false,
        "value": "/results/{job_id}/output.json"
    }
}
```

  ### Transforms

Transforms allow you to modify path values programmatically to handle special cases or format requirements. They're applied after the basic path is determined but before the final configuration is generated.

Multiple transforms can be chained together and are applied in the order specified. This provides a clean way to handle edge cases without cluttering the main mapping logic.


  **Example from deformable\_detr.config.json:** The Deformable DETR network chains multiple transforms to handle data service integration and list formatting, showing how transforms can solve complex path processing requirements.


```json
{
    "config.path": {
        "source": "train_datasets",
        "path": "images.tar.gz",
        "transform": ["data_services_as_parent_possible", "wrap_in_list"]
    }
}
```

  #### Available Transforms:

  **Basic Transforms:**


- **`handle_tar_path`**: Handles special cases for TAR files, adjusting paths for extraction  
- **`wrap_in_list`**: Wraps the value in a list for parameters expecting arrays  
    
  **Dataset Conversion Transforms:** **Example from efficientdet\_tf2.config.json:** Networks that require dataset conversion can use specialized transforms to reference conversion job outputs.


```json
{
    "dataset.train_tfrecords": {
        "source": "train_datasets",
        "path": "results/{dataset_convert_job_id}/dataset_convert",
        "transform": ["use_dataset_convert_job", "wrap_in_list"]
    }
}
```


- **`use_dataset_convert_job`**: References paths from dataset conversion jobs  
- **`data_services_as_parent_possible`**: Handles data service integration scenarios

  ### Conditionals

  Conditionals provide fine-grained control over when specific mappings should be applied. They evaluate dataset metadata or other conditions and only apply the mapping if the condition is met. This prevents incorrect mappings from being applied to incompatible datasets and ensures that your network only receives appropriate data configurations.  
    
  **Example from object\_detection.config.json:** The object detection network uses conditionals to ensure that COCO-specific annotation mappings are only applied to datasets with the COCO format, preventing format mismatches.


```json
{
    "config.path": {
        "source": "id",
        "conditional": {
            "metadata_key": "format",
            "equals": "coco"
        },
        "path": "annotations.json"
    }
}
```

  ### Optional Paths

  Some networks can function with or without certain files. Optional paths allow you to specify files that should be included if they exist but won't cause validation errors if missing.


```json
{
    "config.optional_file": {
        "source": "train_datasets",
        "path": "optional_metadata.json",
        "optional": true
    }
}
```

### 3. Dataset Validation (`dataset_validation`)

  The `dataset_validation` section ensures that uploaded datasets contain all the necessary files and follow the expected structure for your network. This validation happens before any processing begins, preventing runtime errors and providing clear feedback to users about dataset requirements. The validation rules are organized by dataset format, allowing different formats to have different requirements while sharing common validation logic.

  ### Basic Structure

  The validation system is organized around dataset formats, with each format defining its own set of required files and structures. The `required_files` object maps format names to arrays of validation rules. Each rule can specify individual files, combinations of files, or complex validation logic with alternatives and conditionals.


```json
{
    "dataset_validation": {
        "required_files": {
            "format_name": [
                {
                    "path": "images.tar.gz",
                    "type": "file"
                }
            ]
        }
    }
}
```

  ### File Types

- **`file`**: Regular file  
- **`folder`**: Directory  
- **`regex`**: Files matching a pattern (use with `regex` field)

  ### Validation Rules

  #### All files required:

When your network requires multiple files to be present simultaneously, use the `all_of` rule. This ensures that every specified file exists before validation passes.

This is common for datasets that need both images and corresponding labels or annotations. All files in the `all_of` array must be present for the validation to succeed.  
    
  **Example from object\_detection.config.json:** The object detection network requires both images and annotations for COCO format datasets, ensuring complete dataset integrity before processing begins.


```json
{
    "all_of": [
        {"path": "images.tar.gz", "type": "file"},
        {"path": "annotations.json", "type": "file"}
    ]
}
```

  #### Any file from alternatives:

Some networks can work with different types of annotation files or formats. The `any_of` rule validates that at least one of the specified alternatives is present.

This provides flexibility for users while ensuring that some form of required data is available. This is useful when your network can accept multiple annotation formats.


**Example from centerpose.config.json:** The CenterPose network demonstrates flexible validation where either training or validation data files can satisfy the requirement.


```json
{
    "any_of": [
        {"path": "train.tar.gz", "type": "file"},
        {"path": "val.tar.gz", "type": "file"}
    ]
}
```

  #### Intent-based validation:

  Different dataset intents (training, evaluation, testing) may require different files. Intent-based validation checks for files specific to how the dataset will be used. This allows the same dataset format to have different validation requirements depending on its intended use, providing more precise validation feedback.


```json
{
    "intent_based_path": {
        "training": {"path": "train.tar.gz", "type": "file"},
        "evaluation": {"path": "test.tar.gz", "type": "file"}
    }
}
```

  #### Intent restrictions:

Some file types or dataset configurations should only be used with specific intents. Intent restrictions enforce these constraints by ensuring that certain files or configurations are only accepted when the dataset has the appropriate intent. This prevents misuse of raw or unlabeled data for training purposes.


  **Example from object\_detection.config.json:** The object detection network restricts raw format datasets to testing intent only, preventing users from accidentally trying to train on unlabeled data.


```json
{
    "path": "raw_images.tar.gz",
    "type": "file",
    "intent_restriction": ["testing"]
}
```

  #### Regex pattern matching:

  When you need to validate the presence of files matching a pattern rather than specific filenames, use regex validation. This is useful for checking that a directory contains files of a certain type or that model files with varying names are present. The regex validation looks for any files matching the specified pattern.

  **Example from maxine\_dataset.config.json:** The Maxine dataset validates video files using regex patterns.

```json
{
    "path": ".",
    "type": "regex",
    "regex": "mp4"
}
```

  **Example from sparse4d.config.json:** Complex regex patterns for validating nested directory structures.

```json
{
    "all_of": [
        {"path": ".", "type": "regex", "regex": "train/*/videos/*.mp4"},
        {"path": ".", "type": "regex", "regex": "train/*/calibration.json"},
        {"path": ".", "type": "regex", "regex": "train/*/ground_truth.json"}
    ]
}
```

  #### Complex Nested Validation:

  **Example from ml\_recog.config.json:** The metric learning recognition network demonstrates complex validation for deeply nested dataset structures.


```json
{
    "all_of": [
        {
            "path": "metric_learning_recognition/retail-product-checkout-dataset_classification_demo/known_classes/train.tar.gz",
            "type": "file"
        },
        {
            "path": "metric_learning_recognition/retail-product-checkout-dataset_classification_demo/known_classes/reference.tar.gz",
            "type": "file"
        }
    ]
}
```

  ### Example Complete Validation

  A comprehensive validation configuration demonstrates how different validation rules work together to handle multiple dataset formats with varying requirements. Each format can have its own validation logic, from simple file checks to complex combinations with intent restrictions. This example shows the flexibility of the validation system in handling diverse dataset requirements.


  **Example from pose\_classification.config.json:** The pose classification network demonstrates complex validation with alternative folder structures, accommodating both "kinetics" and "nvidia" data organizations while ensuring all required files are present.


```json
{
    "dataset_validation": {
        "required_files": {
            "default": [
                {
                    "any_of": [
                        {
                            "all_of": [
                                {"path": "kinetics", "type": "folder"},
                                {"path": "kinetics/train_data.npy", "type": "file"},
                                {"path": "kinetics/train_label.pkl", "type": "file"},
                                {"path": "kinetics/val_data.npy", "type": "file"},
                                {"path": "kinetics/val_label.pkl", "type": "file"}
                            ]
                        },
                        {
                            "all_of": [
                                {"path": "nvidia", "type": "folder"},
                                {"path": "nvidia/train_data.npy", "type": "file"},
                                {"path": "nvidia/train_label.pkl", "type": "file"},
                                {"path": "nvidia/val_data.npy", "type": "file"},
                                {"path": "nvidia/val_label.pkl", "type": "file"}
                            ]
                        }
                    ]
                }
            ]
        }
    }
}
```

### 4. Dynamic Configuration (`dynamic_config`)

  The `dynamic_config` section handles complex configuration logic that goes beyond simple file mappings. This is where you define rules for modifying configurations based on model types, parent job actions, or other runtime conditions. Dynamic configuration eliminates the need for complex conditional logic in your network code by handling these adjustments declaratively in the configuration file.

  ### Model Type Rules

  When your network supports multiple model architectures or processing modes, different configurations may be needed for each type. Model type rules automatically adjust the configuration based on the selected model type, removing incompatible parameters and applying type-specific transformations. The `model_type_key` specifies where to find the model type in the configuration, and the rules define what changes to make for each type.


  **Example from action\_recognition.config.json:** The action recognition network supports RGB, optical flow, and joint processing modes, with each requiring different parameter sets. The dynamic config removes incompatible parameters for each mode.


```json
{
    "dynamic_config": {
        "model_type_key": "model.model_type",
        "rules": {
            "rgb": {
                "remove": [
                    "model.of_seq_length",
                    "model.of_pretrained_model_path"
                ],
                "remove_if_action": {
                    "train": ["model.of_pretrained_num_classes"]
                }
            },
            "joint": {
                "transform": "split_pretrained_paths"
            }
        }
    }
}
```

  #### Model Type Transform Rules:

  **Example from pose\_classification.config.json:** The pose classification network demonstrates model type-specific parameter removal based on graph layout.


```json
{
    "dynamic_config": {
        "model_type_key": "model.graph_layout",
        "rules": {
            "nvidia": {
                "remove": [
                    "dataset.random_choose",
                    "dataset.random_move", 
                    "dataset.window_size"
                ]
            },
            "openpose": {
                "remove": [
                    "model.pretrained_model_path"
                ]
            }
        }
    }
}
```

  ### Parent Action Rules

  TAO API jobs often build upon previous jobs, and the configuration may need to reference outputs from parent jobs. Parent action rules detect when a job has a parent with a specific action and automatically configure paths to use the parent's outputs. This enables powerful workflow chaining without manual path configuration from users.


  **Example from object\_detection.config.json:** The object detection network demonstrates parent action handling for annotation format conversion, automatically updating annotation paths when the parent job converted annotations to COCO format.


```json
{
    "dynamic_config": {
        "parent_action_rules": {
            "annotation_format_convert": {
                "check_parent_specs": {
                    "data.output_format": "COCO",
                    "if_match": {
                        "set_value": {
                            "data.ann_path": "results/{parent_id}/annotations.json"
                        }
                    }
                },
                "action_restriction": {
                    "data.ann_path": ["analyze", "validate_annotations"]
                }
            }
        }
    }
}
```

  #### Simple Parent Action Rules:

  Some parent action rules can be simpler, just setting values without complex conditions.


```json
{
    "parent_action_rules": {
        "auto_label": {
            "set_value": {
                "data.ann_path": "results/{parent_id}/label.json"
            },
            "conditional": {
                "metadata_key": "format",
                "equals": "coco"
            },
            "action_restriction": ["augment"]
        }
    }
}
```

  ### Action-specific Rules

  Different actions (train, evaluate, inference) may require different configuration adjustments. Action-specific rules automatically modify the configuration based on the current action being performed. This allows you to enable inference-specific settings, remove training-only parameters during evaluation, or set action-appropriate defaults.


  **Example from deformable\_detr.config.json:** The Deformable DETR network removes training-specific parameters during evaluation and inference actions, preventing conflicts and ensuring clean configurations for each action type.


```json
{
    "dynamic_config": {
        "action_rules": {
            "inference": {
                "remove": [
                    "model.pretrained_backbone_path",
                    "train.resume_training_checkpoint_path"
                ]
            }
        }
    }
}
```

  ### Defaults

  Default values ensure that essential configuration parameters are set even when not explicitly provided in the user's specification. Defaults are applied after all other configuration processing is complete, filling in any missing values with sensible defaults. This reduces the configuration burden on users while ensuring your network has all required parameters.


  **Example from action\_recognition.config.json:** The action recognition network provides default label mappings, ensuring that the network has proper class definitions even when not explicitly provided by the user.


```json
{
    "dynamic_config": {
        "defaults": {
            "dataset.label_map": {"catch": 0, "smile": 1}
        }
    }
}
```

### 5. Additional Downloads (`additional_download`)

  The `additional_download` section specifies additional files that need to be downloaded for specific actions beyond the main data source mappings. This is useful when actions require supplementary files from dataset conversion jobs or other sources.

  **Example from sparse4d.config.json:** The Sparse4D network requires additional converted data files for training and evaluation.

```json
{
    "additional_download": {
        "train": [
            {
                "source": "train_datasets",
                "path": "/results/{dataset_convert_job_id}/convert_selective.tar.gz"
            },
            {
                "source": "eval_dataset", 
                "path": "/results/{dataset_convert_job_id}/convert_selective.tar.gz"
            }
        ],
        "evaluate": [
            {
                "source": "eval_dataset",
                "path": "/results/{dataset_convert_job_id}/convert_selective.tar.gz"
            }
        ]
    }
}
```

  ### Additional Download Features:

- **Direct path specification**: Simple path strings for straightforward downloads
- **Path from convert job spec**: Similar to data sources, can use conversion job specifications  
- **Placeholders**: Support for `{dataset_convert_job_id}` and `{dataset_path}` replacement

### 6. Upload Strategy (`upload_strategy`)

  The `upload_strategy` section controls how results are uploaded during job execution. Different strategies can be applied per action to optimize upload behavior.

  **Example from sparse4d.config.json:** The Sparse4D network uses selective tarball upload for dataset conversion.

```json
{
    "upload_strategy": {
        "dataset_convert": {
            "default": "continuous",
            "selective_tarball": {
                "patterns": [
                    "**/Camera*/**"
                ],
                "base_path": "data"
            }
        }
    }
}
```

  ### Upload Strategy Options:

- **`continuous`**: Default continuous upload behavior
- **`tarball_after_completion`**: Create tarball only after job completion
- **`selective_tarball`**: Create selective tarballs based on patterns
  - **`patterns`**: Array of glob patterns to include
  - **`base_path`**: Base path for pattern matching

### 7. Actions Mapping (`actions_mapping`)

  The `actions_mapping` section handles cases where dataset actions should be processed by different networks or with different action names. This is common when certain dataset operations (like augmentation or analysis) are handled by specialized networks rather than the main network. The mapping redirects these actions to the appropriate network while preserving the user's intent.


  **Example from object\_detection.config.json:** The object detection network delegates specialized dataset operations to dedicated networks, such as sending augmentation requests to the augmentation network and analysis requests to the analytics network.


```json
{
    "actions_mapping": {
        "augment": {
            "network": "augmentation",
            "action": "generate"
        },
        "analyze": {
            "network": "analytics"
        },
        "convert_efficientdet_tf2": {
            "network": "efficientdet_tf2",
            "action": "dataset_convert"
        }
    }
}
```

  ### Wildcard Action Mapping

  **Example from visual_changenet_segment.config.json:** You can use wildcards to map all actions to a different network. This is used when you create config files for each sub-task of a network, for example, visual_changenet_classify and visual_changenet_segment.

```json
{
    "actions_mapping": {
        "*": {
            "network": "visual_changenet"
        }
    }
}
```

  ### Action Mapping Patterns:

  #### Network and Action Mapping:

  When an action should be handled by a different network with a different action name.


```json
{
    "augment": {
        "network": "augmentation", 
        "action": "generate"
    }
}
```

  #### Network Mapping Only:

  When an action should be handled by a different network but keep the same action name.


```json
{
    "analyze": {
        "network": "analytics"
    }
}
```

  #### Format-Specific Mappings:

  Some actions may be mapped differently based on the dataset format or other conditions.


```json
{
    "auto_labeling": {
        "network": "auto_label",
        "action": "generate"
    }
}
```

### 8. Spec Parameters (`spec_params`)

The `spec_params` section defines how configuration paths map to parameter types used by the FTMS's experiment specification system. This mapping tells the API how to handle different types of parameters (model paths, output directories, encryption keys, etc.) and ensures that the right parameter handling logic is applied.

Each action can have its own parameter mappings to handle action-specific requirements. This section is crucial for proper parameter processing, model management, and workflow orchestration.

  ### Common Parameter Types

  The FTMS uses standardized parameter types that handle different aspects of experiment configuration:

  #### Core Parameters

- **`output_dir`**: Standard output directory for all results  
- **`key`**: Encryption key for model security (TensorFlow networks)  
- **`encryption_key`**: Encryption key for model security (PyTorch networks)

  #### Model Management Parameters

- **`ptm_if_no_resume_model`**: Use pretrained model if there is model to resume the training  
- **`resume_model`**: Resume from a previous training checkpoint  
- **`parent_model`**: Use model from parent job in workflow  
- **`parent_model_evaluate`**: Use model from parent job specifically for evaluation  
- **`automl_assign_ptm`**: AutoML-specific pretrained model assignment  
- **`automl_resume_model`**: Resume model handling for AutoML experiments

  #### File Creation Parameters

- **`create_onnx_file`**: Generate ONNX export filename  
- **`create_engine_file`**: Generate TensorRT engine filename  
- **`create_cal_cache`**: Generate calibration cache filename  
- **`create_cal_data_file`**: Generate calibration data filename  
- **`create_inference_result_file_pose`**: Generate pose-specific inference result filename

  #### AutoML Interval Parameters

- **`assign_const_value,train.num_epochs`**: Set parameter to training epochs value  
- **`assign_const_value,train.num_epochs,train.checkpoint_interval`**: Complex value assignment

  ### Training Action Parameters

Training actions typically require the most comprehensive parameter mapping, handling pretrained models, resume capabilities, and output configuration.

  **Example from classification\_pyt.config.json:** The PyTorch classification network shows how to handle multiple pretrained model paths and the different encryption key parameter name used in PyTorch networks.


```json
{
    "train": {
        "results_dir": "output_dir",
        "model.backbone.pretrained_backbone_path": "ptm_if_no_resume_model",
        "train.pretrained_model_path": "ptm_if_no_resume_model",
        "train.resume_training_checkpoint_path": "resume_model"
    }
}
```

  ### Evaluation Action Parameters

Evaluation actions focus on using models from parent jobs and handling both checkpoint and TensorRT engine formats.


**Example from centerpose.config.json:** The CenterPose network shows typical evaluation parameter mapping, supporting both checkpoint and TensorRT engine inference paths.


```json
{
    "evaluate": {
        "results_dir": "output_dir",
        "evaluate.checkpoint": "parent_model",
        "evaluate.trt_engine": "parent_model",
        "encryption_key": "key"
    }
}
```

  ### Export Action Parameters

Export actions handle model conversion to different formats, particularly ONNX for interoperability.


  **Example from deformable\_detr.config.json:** The Deformable DETR network shows standard export parameter mapping with automatic ONNX filename generation.


```json
{
    "export": {
        "results_dir": "output_dir",
        "export.checkpoint": "parent_model",
        "export.onnx_file": "create_onnx_file",
        "encryption_key": "key"
    }
}
```

  ### TensorRT Engine Generation Parameters

  TensorRT engine generation requires specialized parameter handling for optimization and calibration.


  **Example from classification\_tf2.config.json:** The TensorFlow classification network demonstrates comprehensive TensorRT engine generation with calibration file handling.


```json
{
    "gen_trt_engine": {
        "results_dir": "output_dir",
        "encryption_key": "key",
        "gen_trt_engine.onnx_file": "parent_model",
        "gen_trt_engine.trt_engine": "create_engine_file",
        "gen_trt_engine.tensorrt.calibration.cal_data_file": "create_cal_data_file",
        "gen_trt_engine.tensorrt.calibration.cal_cache_file": "create_cal_cache"
    }
}
```

  ### Specialized Action Parameters

  Some networks have unique actions requiring specialized parameter handling.


  **Example from efficientdet\_tf2.config.json (Pruning):** The EfficientDet network supports pruning operations with specific parameter requirements.


```json
{
    "prune": {
        "results_dir": "output_dir",
        "prune.checkpoint": "parent_model",
        "encryption_key": "key"
    }
}
```


  **Example from classification\_pyt.config.json (Distillation):** The PyTorch classification network supports knowledge distillation with teacher model parameter handling.


```json
{
    "distill": {
        "results_dir": "output_dir",
        "distill.pretrained_teacher_model_path": "resume_model"
    }
}
```


  **Example from pointpillars.config.json (Retrain after Pruning):** The PointPillars network demonstrates retraining after pruning with pruned model path handling.


```json
{
    "retrain": {
        "key": "key",
        "results_dir": "output_dir",
        "train.pruned_model_path": "parent_model"
    }
}
```

  ### Dataset Processing Parameters

  Some networks include dataset processing actions with specific parameter requirements.


  **Example from stylegan\_xl.config.json:** The StyleGAN XL network includes dataset conversion with custom output filename specification.


```json
{
    "dataset_convert": {
        "results_dir": "output_dir",
        "dest_file_name": "stylegan_dsconvert_output"
    }
}
```

### 9. AutoML Spec Parameters (`automl_spec_params`)

The `automl_spec_params` section defines parameter mappings specifically for AutoML workflows. AutoML requires specialized parameter handling for automated model selection, hyperparameter optimization, and experiment management. These parameters ensure proper integration with the AutoML system while maintaining compatibility with manual workflows.

  ### AutoML-Specific Parameters

  AutoML workflows use specialized parameter types that handle automated experiment management:


- **`automl_output_dir`**: Specialized output directory for AutoML experiments  
- **`automl_assign_ptm`**: Automated pretrained model assignment based on AutoML logic  
- **`automl_resume_model`**: Resume model handling for AutoML experiments  
- **`assign_const_value,parameter_path`**: Dynamic value assignment from other parameters  
    
  **Example from action\_recognition.config.json:** The action recognition network demonstrates comprehensive AutoML parameter mapping with interval assignments.


```json
{
    "automl_spec_params": {
        "results_dir": "automl_output_dir",
        "model.rgb_pretrained_model_path": "automl_assign_ptm",
        "model.of_pretrained_model_path": "automl_assign_ptm",
        "train.resume_training_checkpoint_path": "automl_resume_model",
        "train.checkpoint_interval": "assign_const_value,train.num_epochs",
        "train.validation_interval": "assign_const_value,train.num_epochs,train.checkpoint_interval",
        "encryption_key": "key"
    }
}
```

### 10. Metrics (`metrics`)

The `metrics` section defines the evaluation metrics that your network supports and which metric should be used for monitoring training progress and AutoML optimization. This section is crucial for proper experiment tracking, model selection, and automated hyperparameter optimization. The metrics configuration tells the FTMS which metrics to extract from training logs and which metric to use for determining the best model.

  ### Metric Configuration Structure

  The metrics section contains two main components:


- **`available_metrics`**: List of all metrics that the network can produce during training/evaluation  
- **`monitoring_metric`**: The primary metric used for model selection and AutoML optimization

  ### Common Metric Types

  Different network types typically use different evaluation metrics based on their task:

  #### Segmentation Metrics

  **Example from segformer.config.json:** Segmentation networks typically use IoU (Intersection over Union) based metrics.


```json
{
    "metrics": {
        "available_metrics": ["val_miou", "val_mAcc", "val_acc"],
        "monitoring_metric": "val_miou"
    }
}
```

  ### Metric Usage in FTMS

  The metrics configuration serves several important purposes:


1. **Training Monitoring**: The `monitoring_metric` is used to track training progress and determine the best model checkpoint  
2. **AutoML Optimization**: AutoML uses the `monitoring_metric` to optimize hyperparameters and select the best configuration  
3. **Choose Appropriate Metrics**: Select metrics that align with your task's evaluation criteria  
4. **Clear Naming**: Use clear, descriptive metric names that indicate their purpose  
5. **Consistency**: Maintain consistent metric naming across similar network types

  ### Dynamic Metric Patterns

  **Example from sparse4d.config.json:** Some networks generate metrics with dynamic names that follow specific patterns. You can specify regex patterns to capture these metrics automatically.

```json
{
    "metrics": {
        "available_metrics": ["val_hota", "val_mAP"],
        "monitoring_metric": "val_mAP",
        "dynamic_metric_patterns": [
            "^lr$",
            "^loss_cls_\\d+$",
            "^loss_box_\\d+$",
            "^img_bbox_NuScenes/\\w+_AP_dist_[0-9.]+$",
            "^img_bbox_NuScenes/mAP$"
        ]
    }
}
```

- **`dynamic_metric_patterns`**: Array of regex patterns to match dynamically generated metric names

## Complete Examples

### Simple Network Configuration

This example shows a basic network configuration with essential components:

```json
{
    "api_params": {
        "dataset_type": "my_network",
        "actions": ["train", "evaluate", "inference"],
        "formats": ["custom", "standard"],
        "accepted_ds_intents": ["training", "evaluation", "testing"],
        "image": "TAO_PYTORCH",
        "spec_backend": "yaml",
        "actions_pipe": {
            "train": "train",
            "evaluate": "evaluate", 
            "inference": "inference"
        }
    },
    
    "data_sources": { ... },
    "dataset_validation": { ... },
    "dynamic_config": { ... },
    "additional_download": { ... },
    "upload_strategy": { ... },
    "actions_mapping": { ... },
    "spec_params": { ... },
    "automl_spec_params": { ... },
    "metrics": { ... }
}
```

### Comprehensive Network Configuration

This comprehensive example demonstrates how all the configuration sections work together to define a complete network integration. The example shows a hypothetical network with multiple dataset formats, conditional logic, validation rules, and proper parameter mappings. This serves as a template for creating your own network configurations.

```json
{
    "api_params": {
        "dataset_type": "my_network",
        "actions": ["train", "evaluate", "inference"],
        "formats": ["custom", "standard"],
        "accepted_ds_intents": ["training", "evaluation", "testing"],
        "image": "TAO_PYTORCH",
        "spec_backend": "yaml",
        "actions_pipe": {
            "train": "train",
            "evaluate": "evaluate", 
            "inference": "inference"
        }
    },
    
    "data_sources": {
        "train": {
            "dataset.train_images": {
                "source": "train_datasets",
                "multiple_sources": false,
                "path": "images.tar.gz"
            },
            "dataset.train_labels": {
                "source": "train_datasets", 
                "multiple_sources": false,
                "path_from_format": {
                    "custom": "custom_labels.json",
                    "standard": "labels.tar.gz"
                }
            }
        },
        "evaluate": {
            "dataset.test_images": {
                "source": "eval_dataset",
                "multiple_sources": false,
                "path": "images.tar.gz"
            }
        }
    },
    
    "dataset_validation": {
        "required_files": {
            "default": [
                {"path": "images.tar.gz", "type": "file"}
            ],
            "custom": [
                {
                    "all_of": [
                        {"path": "images.tar.gz", "type": "file"},
                        {"path": "custom_labels.json", "type": "file"}
                    ]
                }
            ]
        }
    },
    
    "spec_params": {
        "train": {
            "results_dir": "output_dir",
            "model.pretrained_path": "ptm_if_no_resume_model",
            "train.resume_training_checkpoint_path": "resume_model",
            "encryption_key": "key"
        },
        "evaluate": {
            "results_dir": "output_dir",
            "evaluate.checkpoint": "parent_model",
            "encryption_key": "key"
        }
    },
    
    "automl_spec_params": {
        "results_dir": "automl_output_dir",
        "model.pretrained_path": "automl_assign_ptm",
        "train.resume_training_checkpoint_path": "automl_resume_model",
        "train.checkpoint_interval": "assign_const_value,train.num_epochs",
        "encryption_key": "key"
    },
    
    "metrics": {
        "available_metrics": ["val_acc", "val_loss"],
        "monitoring_metric": "val_acc"
    }
}
```

## Best Practices

### Configuration Design

1. **Start Simple**: Begin with basic path mappings and add complexity as needed
2. **Use Conditionals**: Leverage format/intent-based logic for different dataset types  
3. **Validate Thoroughly**: Include comprehensive validation rules for all supported formats
4. **Handle Edge Cases**: Use dynamic config for special cases and model-specific logic

### Development Workflow

5. **Test All Paths**: Ensure all combinations of formats, intents, and actions work correctly
6. **Document Assumptions**: Comment complex mappings and transformations in your code
7. **Reuse Patterns**: Follow established patterns from existing network configs
8. **Standardize Parameters**: Use consistent parameter types across similar actions

### Integration Strategy

9. **Plan for AutoML**: Include AutoML parameter mappings for automated workflows
10. **Choose Appropriate Metrics**: Select metrics that align with your task's evaluation criteria

This configuration system provides a flexible, maintainable way to integrate new networks into the Finetuning Microservices without requiring code changes to the core handlers.
