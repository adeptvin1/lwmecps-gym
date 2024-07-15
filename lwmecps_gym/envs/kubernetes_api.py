
from kubernetes import client, config

class k8s:
    def __init__(self, namespace) -> None:
        config.load_kube_config()
        self.namespace = namespace
        self.api = client.CoreV1Api()

        pass

    def k8s_state(self):
        #   need to get per node 'cpu', 'ram', 'tx_bandwidth', 'rx_bandwidth', 'read_disks_bandwidth', 'write_disks_bandwidth'
        list_nodes = self.api.list_node()
        for node in list_nodes.items:
            print(node.status.capacity)
        pass