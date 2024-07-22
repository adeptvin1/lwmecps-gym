
from kubernetes import client, config

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
        pass