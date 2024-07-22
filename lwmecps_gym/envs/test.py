
from kubernetes_api import k8s
import time

minikube = k8s()

for i in range(1):
    state = minikube.k8s_state()
    print(state)
    time.sleep(1)

minikube.k8s_action(namespace='default', deployment_name='time-server', replicas=3, node='minikube-m02')