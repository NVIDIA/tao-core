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

import os
import pytest
from unittest.mock import patch, MagicMock

from nvidia_tao_core.distributed.utils import is_master_node


def test_is_master_node_default():
    """Test is_master_node returns True by default when no distributed setup exists."""
    assert is_master_node() is True


@pytest.mark.parametrize("rank", [1, 2, 3])
def test_is_master_node_rank_env(rank):
    """Test is_master_node with RANK environment variable."""
    with patch.dict(os.environ, {"RANK": str(rank)}):
        assert is_master_node() is False

    with patch.dict(os.environ, {"RANK": "0"}):
        assert is_master_node() is True


@pytest.mark.parametrize("node_rank", [1, 2, 3])
def test_is_master_node_node_rank_env(node_rank):
    """Test is_master_node with NODE_RANK environment variable."""
    with patch.dict(os.environ, {"NODE_RANK": str(node_rank)}):
        assert is_master_node() is False

    with patch.dict(os.environ, {"NODE_RANK": "0"}):
        assert is_master_node() is True


@patch("torch.distributed")
def test_is_master_node_pytorch(mock_dist):
    """Test is_master_node with PyTorch distributed setup."""
    mock_dist.is_available.return_value = True
    mock_dist.is_initialized.return_value = True

    # Test non-master node
    mock_dist.get_rank.return_value = 1
    assert is_master_node() is False

    # Test master node
    mock_dist.get_rank.return_value = 0
    assert is_master_node() is True


def test_is_master_node_mpi():
    """Test is_master_node with MPI setup."""
    mock_comm = MagicMock()

    with patch.dict('sys.modules', {'mpi4py': MagicMock()}):
        from mpi4py import MPI
        MPI.COMM_WORLD = mock_comm

        # Test non-master node
        mock_comm.Get_rank.return_value = 1
        assert is_master_node() is False

        # Test master node
        mock_comm.Get_rank.return_value = 0
        assert is_master_node() is True
