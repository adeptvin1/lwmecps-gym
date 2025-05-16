# Развертывание

## Требования

### Системные требования
- Kubernetes кластер версии 1.20 или выше
- Helm 3
- kubectl
- Доступ к реестру контейнеров (опционально)

### Зависимости
- MongoDB 4.4 или выше
- Weights & Biases аккаунт
- Доступ к Kubernetes API

## Установка

### 1. Добавление Helm репозитория

```bash
helm repo add lwmecps-gym https://your-repo-url
helm repo update
```

### 2. Создание файла конфигурации

Создайте файл `values.yaml` со следующими настройками:

```yaml
# MongoDB настройки
mongodb:
  auth:
    enabled: true
    rootPassword: "your-secure-password"
    username: "lwmecps"
    password: "your-secure-password"
    database: "lwmecps"
  persistence:
    enabled: true
    size: 10Gi
  resources:
    requests:
      memory: "256Mi"
      cpu: "250m"
    limits:
      memory: "512Mi"
      cpu: "500m"

# Weights & Biases настройки
wandb:
  enabled: true
  apiKey: "your-wandb-api-key"
  project: "lwmecps-gym"
  entity: "your-entity"

# Настройки обучения
training:
  resources:
    requests:
      memory: "1Gi"
      cpu: "500m"
    limits:
      memory: "2Gi"
      cpu: "1000m"
  storage:
    models:
      size: 5Gi
    logs:
      size: 2Gi

# Настройки API
api:
  replicas: 2
  resources:
    requests:
      memory: "256Mi"
      cpu: "250m"
    limits:
      memory: "512Mi"
      cpu: "500m"
  ingress:
    enabled: true
    host: "lwmecps-gym.example.com"
    tls: true

# Настройки Kubernetes
kubernetes:
  namespace: "lwmecps-gym"
  serviceAccount:
    create: true
    name: "lwmecps-gym"
  rbac:
    enabled: true
```

### 3. Создание секретов

Создайте файл `secrets.yaml`:

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: lwmecps-gym-secrets
  namespace: lwmecps-gym
type: Opaque
data:
  mongodb-root-password: <base64-encoded-password>
  mongodb-password: <base64-encoded-password>
  wandb-api-key: <base64-encoded-api-key>
```

Примените секреты:

```bash
kubectl apply -f secrets.yaml
```

### 4. Установка Helm chart

```bash
helm install lwmecps-gym lwmecps-gym/lwmecps-gym \
  --namespace lwmecps-gym \
  --create-namespace \
  -f values.yaml
```

### 5. Проверка установки

```bash
# Проверка статуса подов
kubectl get pods -n lwmecps-gym

# Проверка логов
kubectl logs -n lwmecps-gym -l app.kubernetes.io/name=lwmecps-gym

# Проверка сервисов
kubectl get svc -n lwmecps-gym
```

## Конфигурация

### MongoDB

#### Аутентификация
```yaml
mongodb:
  auth:
    enabled: true
    rootPassword: "your-secure-password"
    username: "lwmecps"
    password: "your-secure-password"
    database: "lwmecps"
```

#### Персистентность
```yaml
mongodb:
  persistence:
    enabled: true
    size: 10Gi
    storageClass: "standard"
```

#### Ресурсы
```yaml
mongodb:
  resources:
    requests:
      memory: "256Mi"
      cpu: "250m"
    limits:
      memory: "512Mi"
      cpu: "500m"
```

### Weights & Biases

#### Базовая конфигурация
```yaml
wandb:
  enabled: true
  apiKey: "your-wandb-api-key"
  project: "lwmecps-gym"
  entity: "your-entity"
```

#### Дополнительные настройки
```yaml
wandb:
  tags:
    - "production"
    - "kubernetes"
  notes: "Production deployment"
  config:
    log_artifacts: true
    log_code: true
```

### Обучение

#### Ресурсы
```yaml
training:
  resources:
    requests:
      memory: "1Gi"
      cpu: "500m"
    limits:
      memory: "2Gi"
      cpu: "1000m"
```

#### Хранилище
```yaml
training:
  storage:
    models:
      size: 5Gi
      storageClass: "standard"
    logs:
      size: 2Gi
      storageClass: "standard"
```

### API

#### Масштабирование
```yaml
api:
  replicas: 2
  autoscaling:
    enabled: true
    minReplicas: 2
    maxReplicas: 10
    targetCPUUtilizationPercentage: 80
```

#### Ресурсы
```yaml
api:
  resources:
    requests:
      memory: "256Mi"
      cpu: "250m"
    limits:
      memory: "512Mi"
      cpu: "500m"
```

#### Ingress
```yaml
api:
  ingress:
    enabled: true
    host: "lwmecps-gym.example.com"
    tls: true
    annotations:
      kubernetes.io/ingress.class: nginx
      cert-manager.io/cluster-issuer: letsencrypt-prod
```

### Kubernetes

#### RBAC
```yaml
kubernetes:
  rbac:
    enabled: true
    rules:
      - apiGroups: [""]
        resources: ["pods", "services"]
        verbs: ["get", "list", "watch"]
      - apiGroups: ["apps"]
        resources: ["deployments"]
        verbs: ["get", "list", "watch"]
```

#### ServiceAccount
```yaml
kubernetes:
  serviceAccount:
    create: true
    name: "lwmecps-gym"
    annotations:
      eks.amazonaws.com/role-arn: "arn:aws:iam::123456789012:role/lwmecps-gym-role"
```

## Обновление

### 1. Обновление Helm репозитория

```bash
helm repo update
```

### 2. Обновление релиза

```bash
helm upgrade lwmecps-gym lwmecps-gym/lwmecps-gym \
  --namespace lwmecps-gym \
  -f values.yaml
```

### 3. Проверка обновления

```bash
# Проверка статуса подов
kubectl get pods -n lwmecps-gym

# Проверка версии
kubectl get deployment lwmecps-gym -n lwmecps-gym -o jsonpath='{.spec.template.spec.containers[0].image}'
```

## Удаление

### 1. Удаление релиза

```bash
helm uninstall lwmecps-gym -n lwmecps-gym
```

### 2. Удаление PVC (опционально)

```bash
kubectl delete pvc -n lwmecps-gym --all
```

### 3. Удаление namespace

```bash
kubectl delete namespace lwmecps-gym
```

## Мониторинг

### 1. Prometheus метрики

```yaml
api:
  metrics:
    enabled: true
    serviceMonitor:
      enabled: true
      interval: 30s
```

### 2. Логирование

```yaml
api:
  logging:
    level: INFO
    format: json
    output: stdout
```

### 3. Трейсинг

```yaml
api:
  tracing:
    enabled: true
    provider: jaeger
    sampling: 0.1
```

## Безопасность

### 1. Сетевые политики

```yaml
networkPolicy:
  enabled: true
  ingress:
    - from:
        - namespaceSelector:
            matchLabels:
              name: monitoring
      ports:
        - protocol: TCP
          port: 9090
```

### 2. Pod Security Policy

```yaml
podSecurityPolicy:
  enabled: true
  privileged: false
  readOnlyRootFilesystem: true
  runAsUser:
    rule: MustRunAsNonRoot
```

### 3. Сканирование уязвимостей

```yaml
security:
  scanning:
    enabled: true
    schedule: "0 0 * * *"
    provider: trivy
```

## Резервное копирование

### 1. MongoDB бэкап

```yaml
mongodb:
  backup:
    enabled: true
    schedule: "0 0 * * *"
    retention: 7d
    storage:
      size: 20Gi
```

### 2. Бэкап моделей

```yaml
training:
  backup:
    enabled: true
    schedule: "0 0 * * *"
    retention: 30d
    storage:
      size: 10Gi
```

## Устранение неполадок

### 1. Проверка логов

```bash
# Логи API
kubectl logs -n lwmecps-gym -l app.kubernetes.io/name=lwmecps-gym

# Логи MongoDB
kubectl logs -n lwmecps-gym -l app.kubernetes.io/name=mongodb

# Логи обучения
kubectl logs -n lwmecps-gym -l app.kubernetes.io/name=training
```

### 2. Проверка статуса

```bash
# Статус подов
kubectl get pods -n lwmecps-gym

# Статус сервисов
kubectl get svc -n lwmecps-gym

# Статус PVC
kubectl get pvc -n lwmecps-gym
```

### 3. Проверка событий

```bash
kubectl get events -n lwmecps-gym
```

### 4. Проверка конфигурации

```bash
# Проверка значений Helm
helm get values lwmecps-gym -n lwmecps-gym

# Проверка конфигурации подов
kubectl describe pod -n lwmecps-gym -l app.kubernetes.io/name=lwmecps-gym
```

## Производительность

### 1. Оптимизация ресурсов

```yaml
api:
  resources:
    requests:
      memory: "512Mi"
      cpu: "500m"
    limits:
      memory: "1Gi"
      cpu: "1000m"
  jvm:
    memory:
      min: "256m"
      max: "512m"
```

### 2. Кэширование

```yaml
api:
  caching:
    enabled: true
    type: redis
    ttl: 3600
```

### 3. Масштабирование

```yaml
api:
  autoscaling:
    enabled: true
    minReplicas: 2
    maxReplicas: 10
    targetCPUUtilizationPercentage: 80
    targetMemoryUtilizationPercentage: 80
``` 