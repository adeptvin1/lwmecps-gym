import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from gymnasium import spaces
import logging
from typing import List

from lwmecps_gym.envs import LWMECPSEnv

logger = logging.getLogger(__name__)

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_size=64):
        """
        obs_dim  : размерность вектора наблюдения
        act_dim  : размерность (число возможных действий)
        hidden_size: размер скрытых слоёв
        """
        super().__init__()

        # Актер (Policy)
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, act_dim),
            nn.Softmax(dim=-1)
        )

        # Критик (Value function)
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        action_probs = self.actor(x)
        value = self.critic(x)
        return action_probs, value

    def get_action_and_value(self, x):
        """
        Для заданного x возвращаем:
          - action (выбор из act_dim)
          - log_prob(action)
          - value (V(x))
          - распределение dist
        """
        action_probs, value = self(x)
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, dist, value


class RolloutBuffer:
    def __init__(self, n_steps, obs_dim):
        """
        n_steps: сколько шагов опыта мы собираем перед обновлением (n_steps)
        obs_dim: размер вектора наблюдения
        """
        self.n_steps = n_steps
        self.obs_dim = obs_dim
        self.reset()

    def reset(self):
        self.states = np.zeros((self.n_steps, self.obs_dim), dtype=np.float32)
        self.actions = np.zeros(self.n_steps, dtype=np.int64)
        self.rewards = np.zeros(self.n_steps, dtype=np.float32)
        self.values = np.zeros(self.n_steps, dtype=np.float32)
        self.log_probs = np.zeros(self.n_steps, dtype=np.float32)
        self.dones = np.zeros(self.n_steps, dtype=np.float32)
        self.advantages = np.zeros(self.n_steps, dtype=np.float32)
        self.returns = np.zeros(self.n_steps, dtype=np.float32)
        self.pos = 0

    def add(self, state, action, reward, value, log_prob, done):
        self.states[self.pos] = state
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.values[self.pos] = value
        self.log_probs[self.pos] = log_prob
        self.dones[self.pos] = done
        self.pos += 1

    def is_full(self):
        return self.pos >= self.n_steps

    def compute_advantages(self, gamma, lam):
        """
        gamma     : discount factor
        lam       : lambda для GAE
        """
        gae = 0
        for t in reversed(range(self.n_steps)):
            if t == self.n_steps - 1:
                next_value = 0
            else:
                next_value = self.values[t + 1]
            delta = self.rewards[t] + gamma * next_value * (1 - self.dones[t]) - self.values[t]
            gae = delta + gamma * lam * (1 - self.dones[t]) * gae
            self.advantages[t] = gae
            self.returns[t] = self.advantages[t] + self.values[t]


class PPO:
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_size: int = 64,
        lr: float = 3e-4,
        gamma: float = 0.99,
        lam: float = 0.95,
        clip_eps: float = 0.2,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        device: str = "cpu",
        deployments: List[str] = None
    ):
        """
        Инициализация PPO агента.
        """
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.lr = lr
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.device = device
        self.deployments = deployments or []  # Initialize deployments list

        # Модель
        self.model = ActorCritic(obs_dim, act_dim, hidden_size).to(device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        # Буфер
        self.buffer = RolloutBuffer(n_steps, obs_dim)

    def _flatten_observation(self, obs):
        """
        Handle observation which can be either numpy array or dict
        """
        try:
            if isinstance(obs, np.ndarray):
                return obs
            elif isinstance(obs, dict):
                flattened = []
                for node, node_data in obs.items():
                    # Add hardware metrics
                    for metric in ['cpu', 'ram', 'tx_bandwidth', 'rx_bandwidth', 
                                 'read_disks_bandwidth', 'write_disks_bandwidth', 'avg_latency']:
                        if metric not in node_data:
                            logger.warning(f"Missing metric {metric} for node {node}, using 0")
                            flattened.append(0.0)
                        else:
                            flattened.append(float(node_data[metric]))
                    
                    # Add deployment metrics
                    if 'deployments' not in node_data:
                        logger.warning(f"Missing deployments for node {node}, using 0")
                        for _ in self.deployments:
                            flattened.append(0.0)
                    else:
                        for deployment in self.deployments:
                            if deployment not in node_data['deployments']:
                                logger.warning(f"Missing deployment {deployment} for node {node}, using 0")
                                flattened.append(0.0)
                            else:
                                replicas = node_data['deployments'][deployment].get('replicas', 0)
                                flattened.append(float(replicas))
                
                return np.array(flattened, dtype=np.float32)
            else:
                raise ValueError(f"Unsupported observation type: {type(obs)}")
        except Exception as e:
            logger.error(f"Error flattening observation: {str(e)}")
            raise

    def select_action(self, state):
        """
        Выбираем действие (action) из политики в режиме training.
        state: np.array(obs_dim) — наблюдение.
        Возвращаем: action, log_prob, value
        """
        try:
            # Convert state to tensor
            state = self._flatten_observation(state)
            state_t = torch.tensor(state, dtype=torch.float32).to(self.device)
            
            # Validate observation dimensions
            if len(state_t) != self.obs_dim:
                raise ValueError(f"Observation dimension mismatch. Expected {self.obs_dim}, got {len(state_t)}")
            
            state_t = state_t.unsqueeze(0)

            with torch.no_grad():
                action, log_prob, dist, value = self.model.get_action_and_value(state_t)

            return (action.item(), log_prob.cpu().item(), value.cpu().item())
        except Exception as e:
            logger.error(f"Error in select_action: {str(e)}")
            raise

    def collect_trajectories(self, env):
        """
        Собираем траектории опыта.
        """
        self.buffer.reset()
        state = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0

        try:
            while not self.buffer.is_full():
                action, log_prob, value = self.select_action(state)
                next_state, reward, done, info = env.step(action)
                self.buffer.add(state, action, reward, value, log_prob, done)
                state = next_state
                episode_reward += reward
                episode_length += 1

                if done:
                    state = env.reset()
                    episode_reward = 0
                    episode_length = 0

            self.buffer.compute_advantages(self.gamma, self.lam)
            return episode_reward, episode_length
        except Exception as e:
            logger.error(f"Error in collect_trajectories: {str(e)}")
            raise

    def update(self):
        """
        Обновляем политику на собранных данных.
        """
        indices = np.arange(self.n_steps)
        np.random.shuffle(indices)

        for _ in range(self.n_epochs):
            for start in range(0, self.n_steps, self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]

                states = torch.tensor(self.buffer.states[batch_indices], dtype=torch.float32).to(self.device)
                actions = torch.tensor(self.buffer.actions[batch_indices], dtype=torch.int64).to(self.device)
                old_log_probs = torch.tensor(self.buffer.log_probs[batch_indices], dtype=torch.float32).to(self.device)
                advantages = torch.tensor(self.buffer.advantages[batch_indices], dtype=torch.float32).to(self.device)
                returns = torch.tensor(self.buffer.returns[batch_indices], dtype=torch.float32).to(self.device)

                action_probs, values = self.model(states)
                dist = torch.distributions.Categorical(action_probs)
                new_log_probs = dist.log_prob(actions)

                ratio = torch.exp(new_log_probs - old_log_probs)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = 0.5 * (returns - values.squeeze()).pow(2).mean()

                entropy = dist.entropy().mean()

                loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def train(self, env, total_timesteps):
        """
        Обучаем агента.
        """
        try:
            episode_rewards = []
            episode_lengths = []
            timesteps = 0

            while timesteps < total_timesteps:
                episode_reward, episode_length = self.collect_trajectories(env)
                self.update()
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
                timesteps += episode_length

                if len(episode_rewards) % 10 == 0:
                    avg_reward = np.mean(episode_rewards[-10:])
                    avg_length = np.mean(episode_lengths[-10:])
                    logger.info(f"Episode {len(episode_rewards)}, Average Reward: {avg_reward:.2f}, Average Length: {avg_length:.2f}")

            return {
                "episode_rewards": episode_rewards,
                "episode_lengths": episode_lengths
            }
        except Exception as e:
            logger.error(f"Error in train: {str(e)}")
            raise

    def save(self, path):
        """
        Сохраняем модель.
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'obs_dim': self.obs_dim,
            'act_dim': self.act_dim,
            'deployments': self.deployments
        }, path)

    def load(self, path):
        """
        Загружаем модель.
        """
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.obs_dim = checkpoint['obs_dim']
        self.act_dim = checkpoint['act_dim']
        self.deployments = checkpoint['deployments']


def main():
    # Создаем окружение
    env = LWMECPSEnv(
        num_nodes=3,
        node_name=["node1", "node2", "node3"],
        max_hardware={
            "cpu": 8,
            "ram": 16000,
            "tx_bandwidth": 1000,
            "rx_bandwidth": 1000,
            "read_disks_bandwidth": 500,
            "write_disks_bandwidth": 500,
            "avg_latency": 300,
        },
        pod_usage={
            "cpu": 2,
            "ram": 2000,
            "tx_bandwidth": 20,
            "rx_bandwidth": 20,
            "read_disks_bandwidth": 100,
            "write_disks_bandwidth": 100,
        },
        node_info={},
        deployment_name="mec-test-app",
        namespace="default",
        deployments=["mec-test-app"],
        max_pods=10000,
    )

    # Вычисляем размерность наблюдения
    obs_dim = 0
    for node in env.node_name:
        # Добавляем метрики оборудования
        obs_dim += 7  # cpu, ram, tx_bandwidth, rx_bandwidth, read_disks_bandwidth, write_disks_bandwidth, avg_latency
        # Добавляем метрики развертываний
        obs_dim += len(env.deployments)  # replicas для каждого развертывания

    act_dim = env.action_space.n  # 3 действия: scale down, no change, scale up

    # Создаём агент PPO
    ppo_agent = PPO(
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden_size=64,
        lr=3e-4,
        gamma=0.99,
        lam=0.95,
        clip_eps=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        device="cpu",
        deployments=env.deployments
    )

    # Запускаем обучение на 10 итераций
    print("Starting PPO training for 10 iterations...")
    ppo_agent.train(env, total_timesteps=20480)  # 2048 * 10 = 20480 timesteps

    # Тестируем обученную модель
    print("\nTesting trained model...")
    state = env.reset()
    done = False
    cum_reward = 0.0
    while not done:
        action, _, _ = ppo_agent.select_action(state)
        next_state, reward, done, info = env.step(action)
        cum_reward += reward
        state = next_state

    env.render()
    print(f"Final cumulative reward: {cum_reward:.2f}")
    print(f"Episode info: {info}")


if __name__ == "__main__":
    main()