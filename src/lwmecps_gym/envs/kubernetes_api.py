from kubernetes import client, config
import time
import os
import logging
from kubernetes.client.rest import ApiException
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class k8s:
    def __init__(self, timeout: int = 30) -> None:
        """
        Initialize Kubernetes client.
        Uses in-cluster configuration when running inside Kubernetes,
        falls back to kubeconfig for local development.
        
        Args:
            timeout (int): Timeout in seconds for API operations
        """
        self.timeout = timeout
        try:
            # Try to load in-cluster config first
            config.load_incluster_config()
            logger.info("Using in-cluster Kubernetes configuration")
        except config.ConfigException:
            # Fall back to kubeconfig for local development
            config.load_kube_config()
            logger.info("Using local kubeconfig for Kubernetes configuration")
        
        # Initialize API clients
        self.core_api = client.CoreV1Api()
        self.app_api = client.AppsV1Api()
        
        # Test connection with timeout
        try:
            self.core_api.list_namespace(_request_timeout=self.timeout)
            logger.info("Successfully connected to Kubernetes API")
        except Exception as e:
            logger.error(f"Failed to connect to Kubernetes API: {str(e)}")
            raise

    def validate_node_state(self, node_state: Dict[str, Any]) -> bool:
        """
        Validate node state data.
        
        Args:
            node_state (Dict[str, Any]): Node state to validate
            
        Returns:
            bool: True if state is valid, False otherwise
        """
        required_fields = ['cpu', 'memory', 'deployments']
        return all(field in node_state for field in required_fields)

    def k8s_state(self) -> Dict[str, Any]:
        """
        Get current state of Kubernetes cluster.
        Raises ApiException if any API call fails.
        
        Returns:
            Dict[str, Any]: Current cluster state
        """
        try:
            state = {}
            logger.info("Fetching Kubernetes cluster state...")
            list_nodes = self.core_api.list_node(_request_timeout=self.timeout)
            logger.info(f"Found {len(list_nodes.items)} nodes")
            
            all_pods = self.core_api.list_pod_for_all_namespaces(_request_timeout=self.timeout)
            logger.info(f"Found {len(all_pods.items)} pods")
            
            all_namespaces = self.core_api.list_namespace(_request_timeout=self.timeout)
            logger.info(f"Found {len(all_namespaces.items)} namespaces")
            
            if not list_nodes.items:
                logger.error("No nodes found in the cluster")
                return None
            
            block_ns = ['kube-node-lease', 'kube-public', 'kube-system', 'kubernetes-dashboard']
            namespaces = [namespace.metadata.name for namespace in all_namespaces.items if namespace.metadata.name not in block_ns]
            if 'lwmecps-testapp' not in namespaces:
                namespaces.append('lwmecps-testapp')
            logger.info(f"Using namespaces: {namespaces}")

            pod_count = {namespace: {} for namespace in namespaces}

            for node in list_nodes.items:
                # Get node name from metadata
                node_name = node.metadata.name
                if not node_name:
                    logger.warning(f"Node {node.metadata.name} has no name, skipping")
                    continue
                    
                # Get node capacity
                if not hasattr(node, 'status'):
                    logger.warning(f"Node {node_name} has no status, skipping")
                    continue
                    
                if not hasattr(node.status, 'capacity'):
                    logger.warning(f"Node {node_name} has no capacity information, skipping")
                    continue
                    
                capacity = node.status.capacity
                if not capacity:
                    logger.warning(f"Node {node_name} has no capacity information, skipping")
                    continue
                    
                logger.info(f"Node {node_name} capacity: {capacity}")
                    
                # Extract CPU and memory values
                cpu = capacity.get('cpu')
                memory = capacity.get('memory')
                
                if not cpu or not memory:
                    logger.warning(f"Node {node_name} is missing required capacity fields (cpu or memory), skipping")
                    continue
                    
                state[node_name] = {
                    'cpu': cpu,
                    'memory': memory,
                    'deployments': {}
                }
                logger.info(f"Processing node {node_name} with capacity: CPU={cpu}, Memory={memory}")
                
                for pod in all_pods.items:
                    if pod.spec.node_name == node_name and pod.metadata.namespace in namespaces:
                        # Check for labels
                        if not pod.metadata.labels:
                            logger.debug(f"Pod {pod.metadata.name} in namespace {pod.metadata.namespace} has no labels")
                            continue
                            
                        app_label = pod.metadata.labels.get('app')
                        if not app_label:
                            logger.debug(f"Pod {pod.metadata.name} in namespace {pod.metadata.namespace} has no 'app' label")
                            continue
                            
                        namespace = pod.metadata.namespace
                        if app_label not in pod_count[namespace]:
                            pod_count[namespace][app_label] = 1
                        else:
                            pod_count[namespace][app_label] += 1
                        
                        if namespace not in state[node_name]['deployments']:
                            state[node_name]['deployments'][namespace] = {}
                        
                        state[node_name]['deployments'][namespace][app_label] = {
                            'replicas': pod_count[namespace][app_label]
                        }
            
            if not state:
                logger.error("No valid nodes found in the cluster")
                return None
                
            # Validate state
            for node_name, node_state in state.items():
                if not self.validate_node_state(node_state):
                    logger.warning(f"Invalid state for node {node_name}")
                    continue
                
            logger.info(f"Successfully collected state for nodes: {list(state.keys())}")
            return state
        except ApiException as e:
            logger.error(f"Failed to get Kubernetes state: {str(e)}")
            raise

    def k8s_action(self, namespace: str, deployment_name: str, replicas: int) -> None:
        """
        Perform Kubernetes action on deployment.
        
        Args:
            namespace (str): Namespace name
            deployment_name (str): Deployment name
            replicas (int): Number of replicas
        """
        try:
            # Validate input parameters
            if not all([namespace, deployment_name]):
                raise ValueError("Missing required parameters")
            if replicas < 0:
                raise ValueError("Replicas must be non-negative")
                
            deployment = self.app_api.read_namespaced_deployment(
                name=deployment_name,
                namespace=namespace,
                _request_timeout=self.timeout
            )
            
            if not deployment:
                raise Exception(f"Deployment {deployment_name} not found in namespace {namespace}")
                
            deployment.spec.replicas = replicas

            self.app_api.patch_namespaced_deployment(
                name=deployment_name,
                namespace=namespace,
                body=deployment,
                _request_timeout=self.timeout
            )
            logger.info(f"Updated deployment {deployment_name} to {replicas} replicas")
            
            # Wait for deployment to be ready with timeout
            start_time = time.time()
            while time.time() - start_time < self.timeout:
                if self.wait_for_deployment_to_be_ready(namespace, deployment_name):
                    logger.info(f'Deployment {deployment_name} is now running')
                    return
                time.sleep(2)
            raise TimeoutError(f"Deployment {deployment_name} failed to become ready within {self.timeout} seconds")
            
        except ApiException as e:
            logger.error(f"Failed to perform k8s action: {str(e)}")
            raise

    def wait_for_deployment_to_be_ready(self, namespace: str, deployment_name: str) -> bool:
        """
        Wait for deployment to be ready.
        
        Args:
            namespace (str): Namespace name
            deployment_name (str): Deployment name
            
        Returns:
            bool: True if deployment is ready, False otherwise
        """
        try:
            deployment = self.app_api.read_namespaced_deployment(
                name=deployment_name,
                namespace=namespace,
                _request_timeout=self.timeout
            )
            
            if not deployment:
                logger.error(f"Deployment {deployment_name} not found")
                return False
                
            if deployment.spec.replicas == 0:
                logger.info(f"Deployment {deployment_name} is scaled to 0")
                return True
            elif deployment.status.ready_replicas == deployment.spec.replicas:
                logger.info(f"Deployment {deployment_name} is ready")
                return True
            else:
                logger.info(f"Waiting for deployment {deployment_name} to be ready...")
                return False
                
        except ApiException as e:
            logger.error(f"Failed to check deployment status: {str(e)}")
            return False