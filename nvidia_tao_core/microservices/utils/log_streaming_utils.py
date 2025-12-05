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

"""Log streaming utilities for Kubernetes and Docker.

This module provides direct log streaming from Kubernetes pods and Docker containers
without relying on the container's internal log capture mechanism. This makes log
retrieval more reliable and real-time.
"""

import os
import logging
from typing import Optional, Generator

logger = logging.getLogger(__name__)


def get_k8s_pod_logs(job_id: str, namespace: Optional[str] = None, tail_lines: Optional[int] = None) -> Optional[str]:
    """Get logs directly from Kubernetes pod using Kubernetes Python client.

    For multi-node jobs (StatefulSet), this automatically identifies and retrieves
    logs from the master node (identified by -0 suffix).

    Args:
        job_id (str): The job ID (used as pod/job name)
        namespace (str, optional): Kubernetes namespace. If None, uses default from config
        tail_lines (int, optional): Number of lines to tail. If None, gets all logs

    Returns:
        str: The log content, or None if logs cannot be retrieved
    """
    logger.debug(
        f"[K8S_LOGS] Starting get_k8s_pod_logs for job_id={job_id}, "
        f"namespace={namespace}, tail_lines={tail_lines}"
    )
    try:
        from kubernetes import client, config as k8s_config
        from kubernetes.client.rest import ApiException

        # Load Kubernetes configuration
        logger.debug(f"[K8S_LOGS] Loading Kubernetes configuration for job_id={job_id}")
        try:
            k8s_config.load_incluster_config()
            logger.debug(f"[K8S_LOGS] Successfully loaded in-cluster config for job_id={job_id}")
        except Exception as e1:
            logger.debug(f"[K8S_LOGS] In-cluster config failed: {e1}, trying kube config")
            try:
                k8s_config.load_kube_config()
                logger.debug(f"[K8S_LOGS] Successfully loaded kube config for job_id={job_id}")
            except Exception as e2:
                logger.error(
                    f"[K8S_LOGS] Failed to load any Kubernetes config for job_id={job_id}: "
                    f"in-cluster={e1}, kube={e2}"
                )
                return None

        # Get namespace
        if namespace is None:
            try:
                namespace_file = '/var/run/secrets/kubernetes.io/serviceaccount/namespace'
                logger.debug(f"[K8S_LOGS] Reading namespace from {namespace_file}")
                with open(namespace_file, 'r', encoding='utf-8') as f:
                    namespace = f.read().strip()
                logger.debug(f"[K8S_LOGS] Got namespace from service account: {namespace}")
            except Exception as e:
                namespace = os.getenv("NAMESPACE", "default")
                logger.debug(f"[K8S_LOGS] Using fallback namespace: {namespace} (error: {e})")
        else:
            logger.debug(f"[K8S_LOGS] Using provided namespace: {namespace}")

        v1 = client.CoreV1Api()
        logger.debug(f"[K8S_LOGS] Created CoreV1Api client for job_id={job_id}, namespace={namespace}")

        # Try to find the pod - could be a Job pod or StatefulSet pod
        pod_name = None
        logger.debug(f"[K8S_LOGS] Searching for pod for job_id={job_id} in namespace={namespace}")
        try:
            # First, try to list pods with the job_id as a label selector
            label_selector = f"job-name={job_id}"
            logger.debug(f"[K8S_LOGS] Trying label selector: {label_selector}")
            pods = v1.list_namespaced_pod(namespace=namespace, label_selector=label_selector)

            if pods.items:
                # Found pods with job-name label (regular Kubernetes Job)
                pod_name = pods.items[0].metadata.name
                pod_status = pods.items[0].status.phase
                logger.info(
                    f"[K8S_LOGS] Found Job pod: {pod_name} (status={pod_status}) "
                    f"for job_id: {job_id}"
                )
                logger.debug(f"[K8S_LOGS] Pod details: namespace={namespace}, labels={pods.items[0].metadata.labels}")
            else:
                logger.debug(
                    f"[K8S_LOGS] No pods found with label selector {label_selector}, "
                    f"trying StatefulSet naming"
                )
                # Try StatefulSet naming convention
                # For StatefulSet, pods are named as <statefulset-name>-<ordinal>
                # For multi-node, master is -0, workers are -1, -2, etc.
                master_pod_name = f"{job_id}-0"
                logger.debug(f"[K8S_LOGS] Trying StatefulSet master pod: {master_pod_name}")
                try:
                    pod = v1.read_namespaced_pod(name=master_pod_name, namespace=namespace)
                    pod_name = master_pod_name
                    pod_status = pod.status.phase
                    logger.info(
                        f"[K8S_LOGS] Found StatefulSet master pod: {pod_name} "
                        f"(status={pod_status}) for job_id: {job_id}"
                    )
                    logger.debug(f"[K8S_LOGS] Pod details: namespace={namespace}, labels={pod.metadata.labels}")
                except ApiException as e1:
                    logger.debug(
                        f"[K8S_LOGS] Master pod {master_pod_name} not found "
                        f"(status={e1.status}), trying single pod name"
                    )
                    # Try single pod name (might be a single-node job)
                    try:
                        pod = v1.read_namespaced_pod(name=job_id, namespace=namespace)
                        pod_name = job_id
                        pod_status = pod.status.phase
                        logger.info(
                            f"[K8S_LOGS] Found single pod: {pod_name} (status={pod_status}) "
                            f"for job_id: {job_id}"
                        )
                        logger.debug(f"[K8S_LOGS] Pod details: namespace={namespace}, labels={pod.metadata.labels}")
                    except ApiException as e2:
                        logger.debug("[K8S_LOGS] Direct pod names failed, trying partial name match")
                        # Try partial name match - list all pods and find one containing the job_id
                        try:
                            all_pods = v1.list_namespaced_pod(namespace=namespace)
                            matching_pods = [p for p in all_pods.items if job_id in p.metadata.name]
                            if matching_pods:
                                # Prefer pods ending with -0 (master node) if multiple matches
                                master_pods = [p for p in matching_pods if p.metadata.name.endswith('-0')]
                                selected_pod = master_pods[0] if master_pods else matching_pods[0]
                                pod_name = selected_pod.metadata.name
                                pod_status = selected_pod.status.phase
                                logger.info(
                                    f"[K8S_LOGS] Found pod by partial name match: {pod_name} "
                                    f"(status={pod_status}) for job_id: {job_id}"
                                )
                                logger.debug(
                                    f"[K8S_LOGS] Pod details: namespace={namespace}, "
                                    f"labels={selected_pod.metadata.labels}"
                                )
                            else:
                                logger.error(
                                    f"[K8S_LOGS] Could not find pod for job_id={job_id}: "
                                    f"master_pod={e1.status}, single_pod={e2.status}, no partial matches"
                                )
                                logger.debug(f"[K8S_LOGS] Tried names: {master_pod_name}, {job_id}, and partial match")
                                return None
                        except Exception as e3:
                            logger.error(
                                f"[K8S_LOGS] Error during partial name match for job_id={job_id}: "
                                f"{type(e3).__name__}: {e3}"
                            )
                            logger.debug(f"[K8S_LOGS] Tried names: {master_pod_name}, {job_id}")
                            return None

        except Exception as e:
            logger.error(f"[K8S_LOGS] Exception finding pod for job_id={job_id}: {type(e).__name__}: {e}")
            logger.debug("[K8S_LOGS] Full traceback:", exc_info=True)
            return None

        if not pod_name:
            logger.error(f"[K8S_LOGS] No pod found for job_id={job_id} after all attempts")
            return None

        # Get logs from the pod
        logger.debug(f"[K8S_LOGS] Retrieving logs from pod={pod_name}, namespace={namespace}, tail_lines={tail_lines}")
        try:
            log_kwargs = {
                'name': pod_name,
                'namespace': namespace,
                'timestamps': False,
            }

            if tail_lines is not None:
                log_kwargs['tail_lines'] = tail_lines
                logger.debug(f"[K8S_LOGS] Using tail_lines={tail_lines}")

            logger.debug(f"[K8S_LOGS] Calling read_namespaced_pod_log with kwargs: {log_kwargs}")
            logs = v1.read_namespaced_pod_log(**log_kwargs)
            log_line_count = len(logs.splitlines())
            log_size_kb = len(logs) / 1024
            logger.info(
                f"[K8S_LOGS] Successfully retrieved {log_line_count} lines "
                f"({log_size_kb:.1f} KB) from pod {pod_name}"
            )
            return logs

        except ApiException as e:
            if e.status == 400:
                # Pod might not have started yet or container not ready
                logger.warning(
                    f"[K8S_LOGS] Pod {pod_name} logs not available yet: "
                    f"status={e.status}, reason={e.reason}"
                )
                logger.debug(f"[K8S_LOGS] ApiException details: {e.body}")
                return None
            logger.error(
                f"[K8S_LOGS] Error retrieving logs from pod {pod_name}: "
                f"status={e.status}, reason={e.reason}"
            )
            logger.debug(f"[K8S_LOGS] ApiException details: {e.body}")
            return None

    except ImportError as e:
        logger.error(
            f"[K8S_LOGS] Kubernetes Python client not installed: {e}. "
            f"Install with: pip install kubernetes"
        )
        return None
    except Exception as e:
        logger.error(
            f"[K8S_LOGS] Unexpected error getting Kubernetes pod logs for "
            f"job_id={job_id}: {type(e).__name__}: {e}"
        )
        logger.debug("[K8S_LOGS] Full traceback:", exc_info=True)
        return None


def stream_k8s_pod_logs(job_id: str, namespace: Optional[str] = None,
                        follow: bool = True) -> Optional[Generator[str, None, None]]:
    """Stream logs from Kubernetes pod in real-time.

    For multi-node jobs (StatefulSet), this automatically identifies and streams
    logs from the master node (identified by -0 suffix).

    Args:
        job_id (str): The job ID (used as pod/job name)
        namespace (str, optional): Kubernetes namespace. If None, uses default from config
        follow (bool): Whether to follow the log stream (tail -f behavior)

    Yields:
        str: Log lines as they are produced
    """
    try:
        from kubernetes import client, config as k8s_config, watch
        from kubernetes.client.rest import ApiException

        # Load Kubernetes configuration
        try:
            k8s_config.load_incluster_config()
        except Exception:
            try:
                k8s_config.load_kube_config()
            except Exception as e:
                logger.error(f"Failed to load Kubernetes config: {e}")
                return None

        # Get namespace
        if namespace is None:
            try:
                namespace_file = '/var/run/secrets/kubernetes.io/serviceaccount/namespace'
                with open(namespace_file, 'r', encoding='utf-8') as f:
                    namespace = f.read().strip()
            except Exception:
                namespace = os.getenv("NAMESPACE", "default")

        v1 = client.CoreV1Api()

        # Find the pod (same logic as get_k8s_pod_logs)
        pod_name = None
        try:
            label_selector = f"job-name={job_id}"
            pods = v1.list_namespaced_pod(namespace=namespace, label_selector=label_selector)

            if pods.items:
                pod_name = pods.items[0].metadata.name
            else:
                master_pod_name = f"{job_id}-0"
                try:
                    v1.read_namespaced_pod(name=master_pod_name, namespace=namespace)
                    pod_name = master_pod_name
                except ApiException:
                    try:
                        v1.read_namespaced_pod(name=job_id, namespace=namespace)
                        pod_name = job_id
                    except ApiException:
                        logger.error(f"Could not find pod for job_id: {job_id}")
                        return None

        except Exception as e:
            logger.error(f"Error finding pod for job_id {job_id}: {e}")
            return None

        if not pod_name:
            return None

        # Stream logs
        w = watch.Watch()
        try:
            for line in w.stream(
                v1.read_namespaced_pod_log,
                name=pod_name,
                namespace=namespace,
                follow=follow,
                timestamps=False
            ):
                yield line + '\n'
        except ApiException as e:
            if e.status == 400:
                logger.warning(f"Pod {pod_name} logs not available for streaming yet: {e.reason}")
            else:
                logger.error(f"Error streaming logs from pod {pod_name}: {e}")
            return None
        finally:
            w.stop()
        return None

    except ImportError:
        logger.error("Kubernetes Python client not installed")
        return None
    except Exception as e:
        logger.error(f"Unexpected error streaming Kubernetes pod logs: {e}")
        return None


def get_docker_container_logs(container_name: str, tail_lines: Optional[int] = None) -> Optional[str]:
    """Get logs directly from Docker container using Docker Python client.

    Args:
        container_name (str): The container name (usually same as job_id)
        tail_lines (int, optional): Number of lines to tail. If None, gets all logs

    Returns:
        str: The log content, or None if logs cannot be retrieved
    """
    logger.debug(
        f"[DOCKER_LOGS] Starting get_docker_container_logs for "
        f"container={container_name}, tail_lines={tail_lines}"
    )
    try:
        import docker
        from docker.errors import NotFound, APIError

        logger.debug(f"[DOCKER_LOGS] Initializing Docker client for container={container_name}")
        try:
            client = docker.from_env()
            logger.debug("[DOCKER_LOGS] Docker client initialized successfully")
        except Exception as e:
            logger.error(f"[DOCKER_LOGS] Failed to initialize Docker client: {type(e).__name__}: {e}")
            logger.debug("[DOCKER_LOGS] Docker client initialization failed", exc_info=True)
            return None

        try:
            logger.debug(f"[DOCKER_LOGS] Getting container: {container_name}")
            container = client.containers.get(container_name)
            container_status = container.status
            logger.info(f"[DOCKER_LOGS] Found container: {container_name} (status={container_status})")
            logger.debug(
                f"[DOCKER_LOGS] Container details: id={container.id[:12]}, "
                f"image={container.image.tags}, status={container_status}"
            )

            log_kwargs = {
                'stdout': True,
                'stderr': True,
                'timestamps': False,
            }

            if tail_lines is not None:
                log_kwargs['tail'] = tail_lines
                logger.debug(f"[DOCKER_LOGS] Using tail_lines={tail_lines}")

            logger.debug(f"[DOCKER_LOGS] Calling container.logs with kwargs: {log_kwargs}")
            logs = container.logs(**log_kwargs)

            # Docker returns bytes, decode to string
            if isinstance(logs, bytes):
                log_size_kb = len(logs) / 1024
                logger.debug(f"[DOCKER_LOGS] Decoding {log_size_kb:.1f} KB of log data")
                logs = logs.decode('utf-8', errors='replace')

            log_line_count = len(logs.splitlines())
            log_size_kb = len(logs) / 1024
            logger.info(
                f"[DOCKER_LOGS] Successfully retrieved {log_line_count} lines "
                f"({log_size_kb:.1f} KB) from container {container_name}"
            )
            return logs

        except NotFound as e:
            logger.error(f"[DOCKER_LOGS] Container not found: {container_name}")
            logger.debug(f"[DOCKER_LOGS] NotFound details: {e}")
            return None
        except APIError as e:
            logger.error(f"[DOCKER_LOGS] Docker API error retrieving logs from {container_name}: {e}")
            logger.debug(
                f"[DOCKER_LOGS] APIError details: "
                f"status_code={e.status_code if hasattr(e, 'status_code') else 'N/A'}"
            )
            return None

    except ImportError as e:
        logger.error(f"[DOCKER_LOGS] Docker Python client not installed: {e}. Install with: pip install docker")
        return None
    except Exception as e:
        logger.error(
            f"[DOCKER_LOGS] Unexpected error getting Docker container logs for "
            f"{container_name}: {type(e).__name__}: {e}"
        )
        logger.debug("[DOCKER_LOGS] Full traceback:", exc_info=True)
        return None


def stream_docker_container_logs(container_name: str, follow: bool = True) -> Optional[Generator[str, None, None]]:
    """Stream logs from Docker container in real-time.

    Args:
        container_name (str): The container name (usually same as job_id)
        follow (bool): Whether to follow the log stream (tail -f behavior)

    Yields:
        str: Log lines as they are produced
    """
    try:
        import docker
        from docker.errors import NotFound, APIError

        try:
            client = docker.from_env()
        except Exception as e:
            logger.error(f"Failed to initialize Docker client: {e}")
            return None

        try:
            container = client.containers.get(container_name)

            log_stream = container.logs(
                stdout=True,
                stderr=True,
                stream=True,
                follow=follow,
                timestamps=False
            )

            for log_chunk in log_stream:
                # Docker returns bytes, decode to string
                if isinstance(log_chunk, bytes):
                    log_chunk = log_chunk.decode('utf-8', errors='replace')
                yield log_chunk
            return None

        except NotFound:
            logger.error(f"Container not found: {container_name}")
            return None
        except APIError as e:
            logger.error(f"Docker API error streaming logs from {container_name}: {e}")
            return None

    except ImportError:
        logger.error("Docker Python client not installed")
        return None
    except Exception as e:
        logger.error(f"Unexpected error streaming Docker container logs: {e}")
        return None


def get_job_logs_from_backend(
    job_id: str, backend: Optional[str] = None,
    namespace: Optional[str] = None, tail_lines: Optional[int] = None
) -> Optional[str]:
    """Get logs from the appropriate backend (Kubernetes or Docker).

    This is a unified interface that automatically determines the backend
    and retrieves logs accordingly. For other backends (like NVCF), returns None
    so the caller can fallback to the traditional method.

    Args:
        job_id (str): The job ID
        backend (str, optional): Backend type ('local-k8s', 'local-docker', etc.)
                                If None, reads from BACKEND env var
        namespace (str, optional): Kubernetes namespace (only used for k8s backend)
        tail_lines (int, optional): Number of lines to tail

    Returns:
        str: The log content, or None if logs cannot be retrieved or backend not supported
    """
    if backend is None:
        backend = os.getenv("BACKEND", "local-k8s")
        logger.debug(f"[LOG_BACKEND] Backend not provided, using env var: {backend}")
    else:
        logger.debug(f"[LOG_BACKEND] Using provided backend: {backend}")

    logger.info(
        f"[LOG_BACKEND] Attempting to get logs for job_id={job_id} from backend={backend}, "
        f"namespace={namespace}, tail_lines={tail_lines}"
    )

    if backend == "local-k8s":
        logger.debug(f"[LOG_BACKEND] Calling get_k8s_pod_logs for job_id={job_id}")
        logs = get_k8s_pod_logs(job_id, namespace, tail_lines)
        if logs:
            logger.info(
                f"[LOG_BACKEND] Successfully retrieved {len(logs)} bytes from Kubernetes "
                f"for job_id={job_id}"
            )
        else:
            logger.warning(f"[LOG_BACKEND] Failed to retrieve logs from Kubernetes for job_id={job_id}")
        return logs
    if backend == "local-docker":
        logger.debug(f"[LOG_BACKEND] Calling get_docker_container_logs for job_id={job_id}")
        logs = get_docker_container_logs(job_id, tail_lines)
        if logs:
            logger.info(
                f"[LOG_BACKEND] Successfully retrieved {len(logs)} bytes from Docker "
                f"for job_id={job_id}"
            )
        else:
            logger.warning(f"[LOG_BACKEND] Failed to retrieve logs from Docker for job_id={job_id}")
        return logs
    logger.info(
        f"[LOG_BACKEND] Backend {backend} not supported for direct log streaming, "
        f"will use fallback method"
    )
    return None


def stream_job_logs_from_backend(
    job_id: str, backend: Optional[str] = None,
    namespace: Optional[str] = None, follow: bool = True
) -> Optional[Generator[str, None, None]]:
    """Stream logs from the appropriate backend (Kubernetes or Docker).

    This is a unified interface for streaming logs in real-time.

    Args:
        job_id (str): The job ID
        backend (str, optional): Backend type ('local-k8s', 'local-docker', etc.)
                                If None, reads from BACKEND env var
        namespace (str, optional): Kubernetes namespace (only used for k8s backend)
        follow (bool): Whether to follow the log stream

    Yields:
        str: Log lines as they are produced
    """
    if backend is None:
        backend = os.getenv("BACKEND", "local-k8s")

    logger.info(f"Attempting to stream logs for job_id={job_id} from backend={backend}")

    if backend == "local-k8s":
        stream = stream_k8s_pod_logs(job_id, namespace, follow)
        if stream is not None:
            for line in stream:
                yield line
        return None
    if backend == "local-docker":
        stream = stream_docker_container_logs(job_id, follow)
        if stream is not None:
            for line in stream:
                yield line
        return None
    logger.info(f"Backend {backend} not supported for direct log streaming")
    return None
