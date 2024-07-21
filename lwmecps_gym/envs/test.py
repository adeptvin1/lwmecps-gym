
from kubernetes_api import k8s
import time

minikube = k8s(namespace='default')

for i in range(10):
    minikube.k8s_state()
    time.sleep(1)
