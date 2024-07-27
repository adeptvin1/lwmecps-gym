
from kubernetes import client, config
import time

class k8s:
    def __init__(self, ) -> None:
        config.load_kube_config()
        self.core_api = client.CoreV1Api()
        self.app_api = client.AppsV1Api()

        pass

    def k8s_state(self):
        state = {}
        #   need to get per node 'cpu', 'ram', #not possible -> 'tx_bandwidth', 'rx_bandwidth', 'read_disks_bandwidth', 'write_disks_bandwidth'
        list_nodes = self.core_api.list_node()
        for node in list_nodes.items:
            state[node.status.addresses[1].address] = node.status.capacity
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
            break
        pass

    def wait_for_deployment_to_be_ready(self, namespace, deployment_name):
        while True:
            deployment = self.app_api.read_namespaced_deployment(
                name=deployment_name,
                namespace=namespace
            )
            if deployment.status.ready_replicas == deployment.spec.replicas:
                print("Deployment " + deployment_name + " is ready.")
                break
            else:
                print("Waiting for deployment to be ready...")
                time.sleep(5)  # Подождать 5 секунд перед следующей проверкой