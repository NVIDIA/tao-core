# Build script for FTMS Docker image
#!/bin/bash
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

# Build script for FTMS Docker image

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_NAME="${IMAGE_NAME:-nvcr.io/nvstaging/tao/tao-ftms}"
IMAGE_TAG="${IMAGE_TAG:-dev}"

echo "Building FTMS Docker image: ${IMAGE_NAME}:${IMAGE_TAG}"
echo "Working directory: ${SCRIPT_DIR}"

cd "${SCRIPT_DIR}"

# Build the Docker image
docker build \
    -f Dockerfile \
    -t "${IMAGE_NAME}:${IMAGE_TAG}" \
    .

echo ""
echo "✅ Build complete!"
echo "To run the container:"
echo "  docker run -p 8000:8000 ${IMAGE_NAME}:${IMAGE_TAG}"
echo ""
echo "To push to registry:"
echo "  docker push ${IMAGE_NAME}:${IMAGE_TAG}"

