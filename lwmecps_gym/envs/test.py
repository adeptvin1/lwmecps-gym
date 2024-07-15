
from kubernetes_api import k8s

minikube = k8s(namespace='default')
minikube.k8s_state()
