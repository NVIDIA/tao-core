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

"""Flask blueprints for organizing API endpoints.

This package uses versioned APIs with the following structure:

- v1/: Contains all v1 API blueprints including admin (current production)
- v2/: Contains all v2 API blueprints including admin (enhanced features)

All admin functionality is now versioned and integrated into v1 and v2.
Use the api_versions module for automatic registration of all versions.
"""

# Admin blueprints are now versioned and imported via v1/ and v2/ directories

# Versioned imports
try:
    from .v1 import *  # Import all v1 blueprints  # noqa: F403, F401
except ImportError:
    pass

try:
    from .v2 import *  # Import all v2 blueprints (when available)  # noqa: F403, F401
except ImportError:
    pass

__all__ = []

# Add versioned blueprints to __all__ if they exist
try:
    from .v1 import __all__ as v1_all
    __all__.extend(v1_all)
except ImportError:
    pass

try:
    from .v2 import __all__ as v2_all
    __all__.extend(v2_all)
except ImportError:
    pass
