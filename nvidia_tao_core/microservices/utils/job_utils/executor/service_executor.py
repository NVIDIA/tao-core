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

"""Service executor for Kubernetes Service operations"""
import time
import traceback
from kubernetes import client
from kubernetes.client.rest import ApiException

from nvidia_tao_core.microservices.utils.stateless_handler_utils import get_handler_job_metadata, get_dnn_status
from nvidia_tao_core.microservices.utils.handler_utils import get_statefulset_service_name

from .base_executor import BaseExecutor
from nvidia_tao_core.microservices.utils.executor_utils import (
    check_service_ready,
    check_endpoints_ready
)


class ServiceExecutor(BaseExecutor):
    """Handles Kubernetes Service operations"""

    def create_service_unified(self, service_name, selector, ports, service_type="ClusterIP", labels=None,
                               return_info=False, add_owner_reference=True):
        """Unified function to create Kubernetes services with flexible configuration"""
        try:
            name_space = self.get_namespace()
            api_instance = client.CoreV1Api()

            # Set default labels
            if labels is None:
                labels = {}

            # Handle different port formats
            if isinstance(ports, int):
                # Single port number
                service_ports = [client.V1ServicePort(port=ports, target_port=ports)]
            elif isinstance(ports, tuple) and len(ports) == 2:
                # Single (port, target_port) tuple
                service_ports = [client.V1ServicePort(port=ports[0], target_port=ports[1])]
            elif isinstance(ports, list):
                # List of port configurations
                service_ports = []
                for port_config in ports:
                    if isinstance(port_config, tuple):
                        if len(port_config) == 2:
                            # (port, target_port)
                            service_ports.append(client.V1ServicePort(
                                port=port_config[0],
                                target_port=port_config[1],
                                name=f"port-{port_config[0]}"
                            ))
                        elif len(port_config) == 3:
                            # (port, target_port, name)
                            service_ports.append(client.V1ServicePort(
                                port=port_config[0],
                                target_port=port_config[1],
                                name=port_config[2],
                                protocol="TCP"
                            ))
                    else:
                        # Just port number
                        service_ports.append(client.V1ServicePort(
                            port=port_config,
                            target_port=port_config,
                            name=f"port-{port_config}"
                        ))
            else:
                raise ValueError(f"Unsupported ports format: {ports}")

            # Configure service spec
            cluster_ip = "None" if service_type == "Headless" else None
            spec = client.V1ServiceSpec(
                type=service_type if service_type != "Headless" else "ClusterIP",
                ports=service_ports,
                selector=selector,
                cluster_ip=cluster_ip
            )

            # Configure metadata
            metadata_kwargs = {"name": service_name, "labels": labels}
            if add_owner_reference:
                metadata_kwargs["owner_references"] = [self.get_owner_reference()]

            # Create service object
            service = client.V1Service(
                api_version="v1",
                kind="Service",
                metadata=client.V1ObjectMeta(**metadata_kwargs),
                spec=spec
            )

            # Create the service
            api_response = api_instance.create_namespaced_service(
                namespace=name_space,
                body=service
            )
            self.logger.info(f"Service created: {service_name}")

            if return_info:
                return {
                    "service_name": service_name,
                    "cluster_ip": api_response.spec.cluster_ip,
                    "ports": ports
                }
            return None

        except Exception as e:
            self.logger.error(f"Failed to create service {service_name}: {e}")
            if return_info:
                raise
            # For backward compatibility with create_service, don't raise for non-return cases
            self.logger.error(traceback.format_exc())
            return None

    def create_service(self, service_name, selector, service_port, target_port, labels=None):
        """Legacy function - delegates to unified service creation"""
        if labels is None:
            labels = {}
        return self.create_service_unified(
            service_name=service_name,
            selector=selector,
            ports=(service_port, target_port),
            service_type="Headless",
            labels=labels,
            return_info=False,
            add_owner_reference=True
        )

    def create_flask_service(self, job_id):
        """Create a service for a microservice pod"""
        service_name = f"flask-service-{job_id}"
        selector = {
            "app": "flask",
            "job-id": job_id
        }
        return self.create_service_unified(
            service_name=service_name,
            selector=selector,
            ports=(8000, 8000),
            service_type="Headless",
            labels={},
            return_info=False,
            add_owner_reference=True
        )

    def create_statefulset_service(self, job_id, statefulset_type="multinode", ports=None, service_type="ClusterIP"):
        """Create a service for a statefulset with flexible configuration"""
        if statefulset_type == "inference_microservice":
            # Inference microservice service configuration
            service_name = f"ims-svc-{job_id}"
            statefulset_name = f"ims-{job_id}"
            selector = {"statefulset": statefulset_name}
            labels = {"app": "ims"}

            # Handle ports format for inference microservices
            if isinstance(ports, tuple) and len(ports) == 2:
                port_list = [
                    (ports[0], ports[0], "http-port"),
                    (ports[1], ports[1], "health-port")
                ]
            else:
                port_list = ports or [(8080, 8080, "http-port"), (8081, 8081, "health-port")]

            return self.create_service_unified(
                service_name=service_name,
                selector=selector,
                ports=port_list,
                service_type=service_type,
                labels=labels,
                return_info=True,
                add_owner_reference=False
            )

        # Multinode service configuration
        service_name = get_statefulset_service_name(job_id)
        selector = {
            "app": "multinode",
            "job-id": job_id
        }
        labels = {
            "app": "multinode",
            "job-id": job_id
        }
        # Default to single port if not specified
        port_info = ports[0] if ports else (8000, 8000)

        return self.create_service_unified(
            service_name=service_name,
            selector=selector,
            ports=port_info,
            service_type="Headless",
            labels=labels,
            return_info=False,
            add_owner_reference=True
        )

    def delete_service(self, job_id=None, service_name=None, service_type="default"):
        """Delete a microservice pod's service with flexible service name handling"""
        try:
            # Handle different service naming patterns
            if service_name is None:
                if job_id is None:
                    raise ValueError("Either job_id or service_name must be provided")

                if service_type == "flask":
                    service_name = f"flask-service-{job_id}"
                elif service_type == "statefulset":
                    service_name = get_statefulset_service_name(job_id)
                else:
                    # Default case
                    service_name = job_id
            elif service_type == "inference_microservice" and job_id is None:
                # Extract job_id from inference microservice service name pattern
                if service_name.startswith("ims-svc-"):
                    job_id = service_name.replace("ims-svc-", "")
                else:
                    job_id = service_name  # fallback

            name_space = self.get_namespace()
            core_v1 = client.CoreV1Api()
            service = core_v1.read_namespaced_service(name=service_name, namespace=name_space)
            if not service:
                self.logger.info(f"Service {service_name} not found in namespace {name_space}")
                return True  # Return True since the goal (service not existing) is achieved

            core_v1.delete_namespaced_service(name=service_name, namespace=name_space)
            self.logger.info(f"Successfully deleted service: {service_name}")
            return True
        except ApiException as e:
            if e.status == 404:
                # Service not found - deletion goal already achieved (likely already deleted by another process)
                self.logger.info(f"Service {service_name} not found (404) - already deleted. Deletion successful.")
                return True
            self.logger.error(f"ApiException thrown in delete_service is {str(e)}")
            self.logger.error(traceback.format_exc())
            return False
        except Exception as e:
            self.logger.error(f"Exception thrown in delete_service is {str(e)}")
            self.logger.error(traceback.format_exc())
            return False

    def check_service_ready(self, service_name, namespace):
        """Check if the specified service is ready."""
        return check_service_ready(service_name, namespace)

    def check_endpoints_ready(self, service_name, namespace):
        """Check if the specified service has ready endpoints."""
        return check_endpoints_ready(service_name, namespace)

    def wait_for_service(self, job_id, service_name=None):
        """Wait until the specified service is ready or timeout is reached."""
        if not service_name:
            service_name = get_statefulset_service_name(job_id)
        namespace = self.get_namespace()
        start_time = time.time()
        while time.time() - start_time < 300:
            # Check job metadata status - if job is already terminated, exit early
            job_metadata = get_handler_job_metadata(job_id)
            if job_metadata:
                metadata_status = job_metadata.get("status")
                # Check if job has been terminated (by timeout, cancellation, or completion)
                if metadata_status in ("Canceled", "Canceling", "Paused", "Pausing", "Error", "Done"):
                    self.logger.info(
                        f"Job {job_id} has status '{metadata_status}'. "
                        f"Exiting wait_for_service early (no need to wait for service that won't come up)."
                    )
                    return metadata_status

            # Also check DNN status to catch timeout terminations early
            try:
                dnn_status = get_dnn_status(job_id, automl=False)
                if dnn_status:
                    # Check the most recent status entry
                    latest_status = dnn_status[-1] if isinstance(dnn_status, list) and len(dnn_status) > 0 else {}
                    if isinstance(latest_status, dict):
                        status_msg = latest_status.get('status', '')
                        if isinstance(status_msg, str):
                            import json
                            try:
                                status_data = json.loads(status_msg)
                                job_status = status_data.get('status', '')
                                # If job has FAILURE status, it's been terminated
                                if job_status == 'FAILURE':
                                    self.logger.info(
                                        f"Job {job_id} has DNN status 'FAILURE'. "
                                        f"Exiting wait_for_service early (job terminated)."
                                    )
                                    return "Error"
                            except (json.JSONDecodeError, AttributeError):
                                pass
            except Exception as e:
                # Don't fail the wait if we can't check DNN status
                self.logger.debug(f"Could not check DNN status for {job_id}: {e}")

            # Check if service is ready
            if (self.check_service_ready(service_name, namespace) and
                    self.check_endpoints_ready(service_name, namespace)):
                self.logger.info(f"Service '{service_name}' is ready.")
                return "Running"
            self.logger.info(f"Waiting for service '{service_name}' to be ready...")
            time.sleep(10)
        self.logger.error(f"Timed out waiting for service '{service_name}' to be ready.")
        return "Error"
