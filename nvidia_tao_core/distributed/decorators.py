# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Distributed decorators."""

from functools import wraps

from nvidia_tao_core.distributed.utils import is_master_node


def master_node_only(fn):
    """Decorator to ensure function only runs on master node.

    This decorator handles both PyTorch Lightning (NODE_RANK),
    PyTorch DDP/torchrun (RANK), and MPI (comm.Get_rank()) distributed training cases.

    Returns:
        The decorated function that only executes on rank zero.
    """
    @wraps(fn)
    def wrapped_fn(*args, **kwargs):
        if not is_master_node():
            return None

        return fn(*args, **kwargs)
    return wrapped_fn
