from kubernetes import client, config
import time
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class k8s:
    def __init__(self) -> None:
        """
        Initialize Kubernetes client.
        Uses in-cluster configuration when running inside Kubernetes,
        falls back to kubeconfig for local development.
        """
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
        
        # Test connection
        try:
            self.core_api.list_namespace()
            logger.info("Successfully connected to Kubernetes API")
        except Exception as e:
            logger.error(f"Failed to connect to Kubernetes API: {str(e)}")
            raise

    def k8s_state(self):
        state = {}
        list_nodes = self.core_api.list_node()
        all_pods = self.core_api.list_pod_for_all_namespaces()
        all_namespaces = self.core_api.list_namespace()
        
        block_ns = ['kube-node-lease', 'kube-public', 'kube-system', 'kubernetes-dashboard']
        namespaces = [namespace.metadata.name for namespace in all_namespaces.items if namespace.metadata.name not in block_ns]

        pod_count = {namespace: {} for namespace in namespaces}

        for node in list_nodes.items:
            node_name = node.status.addresses[1].address
            state[node_name] = node.status.capacity
            state[node_name]['deployments'] = {}
            
            for pod in all_pods.items:
                if pod.spec.node_name == node_name and pod.metadata.namespace in namespaces:
                    app_label = pod.metadata.labels.get('app')
                    if app_label:
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
        return state

    def k8s_action(self, namespace, deployment_name, replicas, node):
        deployment = self.app_api.read_namespaced_deployment(
            name=deployment_name,
            namespace=namespace
        )
        deployment.spec.replicas = replicas
        deployment.spec.template.spec.node_selector = {
            "kubernetes.io/hostname" : node
        }

        self.app_api.patch_namespaced_deployment(
            name=deployment_name,
            namespace=namespace,
            body=deployment
        )
        time.sleep(2)
        while True:
            self.wait_for_deployment_to_be_ready(namespace, deployment_name)
            print('Working on ' + node)
            break
        pass

    def wait_for_deployment_to_be_ready(self, namespace, deployment_name):
        while True:
            deployment = self.app_api.read_namespaced_deployment(
                name=deployment_name,
                namespace=namespace
            )
            if deployment.spec.replicas == 0:
                print("Deployment " + deployment_name + " is scaled to 0.")
                break
            elif deployment.status.ready_replicas == deployment.spec.replicas:
                print("Deployment " + deployment_name + " is ready.")
                break
            else:
                print("Waiting for deployment to be ready...")
                time.sleep(5)  # Подождать 5 секунд перед следующей проверкой