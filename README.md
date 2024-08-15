# lwmecps-gym

LWMECPS-gym is a collection of [OpenAI Gym](https://github.com/openai/gym) environments for reinforcement learning-based trading algorithms.

## Description

Данный репозиторий используется для создания окружения Kubernetes кластера и отработки использования RL в задачах опитимизации размещения вычислительных сервисов в геораспределенных узлах обработки данных (т.е k8s размазанный по условному городу, где сеть сотовой связи является транспортной).

## Installation

### From Repository

```bash
git clone https://github.com/adeptvin1/lwmecps-gym.git
cd lwmecps-gym
pip install -e .
```

### How to start work with lib

Вам необходимо сделать следующие вещи:
1. Установить minikube используя инструкцию -> [minikube install](https://minikube.sigs.k8s.io/docs/start/?arch=%2Fmacos%2Farm64%2Fstable%2Fbinary+download)
2. Установить kubectl.
3. Запустить minikube используя команду `minikube start --nodes 4 -p minikube`
4. Запустить деплой mock сервиса используя команду `kubectl apply -f deployment.yml` (вы можете поправить его при необходимости)
5. Проверить состояние подов можно используя команду `kubectl get pods`
6. Чтобы запустить тестовый сервис используйте команду `python3 ./test_service.py`

## Environment

### Action


## Resources

`https://www.comet.com/docs/v2/guides/quickstart/` - Интересная штука для визализации обучения
`https://gymnasium.farama.org/api/spaces/composite/` - Документация по Gymnasium

