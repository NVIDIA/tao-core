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

"""Workflow manager for normal model actions"""
from nvidia_tao_core.microservices.constants import NO_SPEC_ACTIONS_MODEL
from nvidia_tao_core.microservices.handlers.stateless_handlers import get_handler_type, get_job
from nvidia_tao_core.microservices.handlers.utilities import JobContext
from nvidia_tao_core.microservices.job_utils.workflow import Dependency, Job, Workflow


def create_job_context(
    parent_job_id,
    action,
    job_id,
    handler_id,
    user_id,
    org_name,
    kind,
    handler_metadata=None,
    specs=None,
    name=None,
    description=None,
    num_gpu=-1,
    platform_id=None
):
    """Calls the create job contexts function"""
    network = get_handler_type(handler_metadata)
    if not network:
        raise ValueError(f"Handler {handler_id} not found for user {user_id}")

    if not specs and action not in NO_SPEC_ACTIONS_MODEL:
        raise ValueError(f"Specs are required to create a job context for {action} action.")

    # Create a jobcontext
    job_context = JobContext(
        job_id,
        parent_job_id,
        network,
        action,
        handler_id,
        user_id,
        org_name,
        kind,
        specs=specs,
        name=name,
        description=description,
        num_gpu=num_gpu,
        platform_id=platform_id
    )
    return job_context


def on_new_job(job_context):
    """Assigns dependencies for a new job

    Creates job_context dictionary and enqueues the job to workflow
    """
    deps = []
    deps.append(Dependency(type="parent"))
    deps.append(Dependency(type="specs"))
    deps.append(Dependency(type="model"))
    deps.append(Dependency(type="dataset"))

    num_gpu = job_context.num_gpu
    platform_id = job_context.platform_id

    if job_context.action not in ["convert", "dataset_convert", "kmeans", "annotation"]:
        num_gpu = 1
        platform_id = None
    elif job_context.action in ("convert", "gen_trt_engine"):
        if job_context.specs and job_context.specs.get("platform_id"):
            num_gpu = job_context.specs.get("num_gpu")
            platform_id = job_context.specs.get("platform_id")

    if num_gpu > 0:
        deps.append(Dependency(type="gpu", name=platform_id, num=num_gpu))

    job = {
        'user_id': job_context.user_id,
        'org_name': job_context.org_name,
        'num_gpu': job_context.num_gpu,
        'platform_id': job_context.platform_id,
        'kind': job_context.kind,
        'id': job_context.id,
        'parent_id': job_context.parent_id,
        'priority': 1,
        'action': job_context.action,
        'network': job_context.network,
        'handler_id': job_context.handler_id,
        'created_on': job_context.created_on,
        'last_modified': job_context.last_modified,
        'dependencies': deps,
        'specs': job_context.specs,
        'workflow_status': 'enqueued'
    }
    j = Job(**job)
    Workflow.enqueue(j)


def on_delete_job(job_id):
    """Dequeue a job"""
    job_metadata = get_job(job_id)
    job_dict = {
        'user_id': job_metadata["user_id"],
        'org_name': job_metadata["org_name"],
        'num_gpu': job_metadata["num_gpu"],
        'platform_id': job_metadata["platform_id"],
        'kind': job_metadata["kind"],
        'id': job_metadata["id"],
        'parent_id': job_metadata["parent_id"],
        'priority': 1,
        'action': job_metadata["action"],
        'network': job_metadata["network"],
        'handler_id': job_metadata["handler_id"],
        'created_on': job_metadata["created_on"],
        'last_modified': job_metadata["last_modified"],
        'specs': job_metadata["specs"],
        'workflow_status': job_metadata["workflow_status"],
    }
    job = Job(**job_dict)
    Workflow.dequeue(job)
