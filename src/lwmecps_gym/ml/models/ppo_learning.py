import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
from torch.distributions.categorical import Categorical
import numpy as np
from gymnasium import spaces

# TODO: Добавить ENV LWMECPSEnv2


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_size=64):
        """
        obs_dim  : размерность вектора наблюдения
        act_dim  : размерность (число возможных действий)
        hidden_size: размер скрытых слоёв
        """
        super(ActorCritic, self).__init__()

        # Актер (Policy)
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, act_dim),  # logits на act_dim действий
        )

        # Критик (Value function)
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),  # ценность состояния (скаляр)
        )

    def forward(self, obs):
        """
        Он просто есть, потому что мы обязаны привести реализацию этого класса
        """
        raise NotImplementedError

    def get_action_and_value(self, obs):
        """
        Для заданного obs возвращаем:
          - action (выбор из act_dim)
          - log_prob(action)
          - value (V(obs))
          - распределение dist
        """
        logits = self.actor(obs)
        dist = Categorical(logits=logits)
        action = dist.sample()
        value = self.critic(obs).squeeze(-1)  # shape: [batch_size]
        return action, dist.log_prob(action), dist, value


class RolloutBuffer:
    def __init__(self, size, obs_dim):
        """
        size   : сколько шагов опыта мы собираем перед обновлением (n_steps)
        obs_dim: размер вектора наблюдения
        """
        self.size = size

        self.states = np.zeros((size, obs_dim), dtype=np.float32)
        self.actions = np.zeros(size, dtype=np.int64)
        self.rewards = np.zeros(size, dtype=np.float32)
        self.dones = np.zeros(size, dtype=np.bool_)

        # log(pi(a|s)) — логарифм вероятности выбранного действия
        self.log_probs = np.zeros(size, dtype=np.float32)
        # V(s)
        self.values = np.zeros(size, dtype=np.float32)

        # Для расчёта advantages
        self.advantages = np.zeros(size, dtype=np.float32)
        self.returns = np.zeros(size, dtype=np.float32)

        self.ptr = 0  # указатель на текущую позицию

    def store(self, state, action, reward, done, log_prob, value):
        idx = self.ptr
        self.states[idx] = state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.dones[idx] = done
        self.log_probs[idx] = log_prob
        self.values[idx] = value

        self.ptr += 1

    def ready_for_update(self):
        return self.ptr == self.size

    def compute_advantages(self, last_value, gamma=0.99, lam=0.95):
        """
        last_value: V(s_{t+1}) — ценность последнего состояния (для GAE)
        gamma     : discount factor
        lam       : lambda для GAE
        """
        # GAE лямбда (Generalized Advantage Estimation)
        # a_t = delta_t + gamma*lam*delta_{t+1} + ... и т.д.
        # delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)

        adv = 0.0
        for i in reversed(range(self.size)):
            if i == self.size - 1:
                next_non_terminal = 1.0 - float(self.dones[i])
                next_value = last_value
            else:
                next_non_terminal = 1.0 - float(self.dones[i])
                next_value = self.values[i + 1]

            delta = (
                self.rewards[i]
                + gamma * next_value * next_non_terminal
                - self.values[i]
            )
            adv = delta + gamma * lam * next_non_terminal * adv
            self.advantages[i] = adv

        self.returns = self.advantages + self.values

    def get(self, batch_size=None):
        """
        Возвращает батчи данных (state, action, log_prob, return, advantage и т.д.)
        Если batch_size=None, возвращаем всё сразу.
        """
        if batch_size is None:
            batch_size = self.size

        indices = np.arange(self.size)
        np.random.shuffle(indices)

        start = 0
        while start < self.size:
            end = start + batch_size
            yield (
                self.states[indices[start:end]],
                self.actions[indices[start:end]],
                self.log_probs[indices[start:end]],
                self.returns[indices[start:end]],
                self.advantages[indices[start:end]],
            )
            start = end

    def reset(self):
        self.ptr = 0


class PPO:
    def __init__(
        self,
        obs_dim,
        act_dim,
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
    ):
        """
        obs_dim   : размерность наблюдения
        act_dim   : размерность (количество дискретных действий)
        hidden_size: размер скрытых слоёв в сети
        lr        : learning rate
        gamma     : discount factor
        lam       : GAE-lambda
        clip_eps  : epsilon в PPO objective (clip range)
        ent_coef  : коэффициент при энтропии (необязательно)
        vf_coef   : коэффициент при value loss
        n_steps   : сколько шагов опыта собираем на каждом цикле обучения
        batch_size: размер мини-батча при оптимизации
        n_epochs  : число эпох обновления на каждом цикле
        device    : 'cpu' или 'cuda'
        """
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef

        self.n_steps = n_steps
        self.batch_size = batch_size
        self.n_epochs = n_epochs

        self.device = device

        # Модель
        self.model = ActorCritic(obs_dim, act_dim, hidden_size).to(device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        # Буфер
        self.buffer = RolloutBuffer(n_steps, obs_dim)

    def select_action(self, state):
        """
        Выбираем действие (action) из политики в режиме training.
        state: np.array(obs_dim) — одномерный вектор наблюдения.
        Возвращаем: action, log_prob, value
        """
        state_t = torch.tensor(state, dtype=torch.float32).to(self.device)
        # batch размером [1, obs_dim]
        state_t = state_t.unsqueeze(0)

        with torch.no_grad():
            action, log_prob, dist, value = self.model.get_action_and_value(state_t)

        return (action.item(), log_prob.cpu().item(), value.cpu().item())

    def collect_trajectories(self, env):
        """
        Собираем n_steps переходов в буфер (self.buffer).
        Затем считаем GAE.
        """
        self.buffer.reset()

        # Начинаем с нового эпизода
        state = env.reset()

        for t in range(self.n_steps):
            action, log_prob, value = self.select_action(state)
            next_state, reward, done, info = env.step(action)

            self.buffer.store(state, action, reward, done, log_prob, value)

            state = next_state

            if done:
                # начинаем новый эпизод, если эпизод закончился
                state = env.reset()

        # Получим V(s_{t+1}) для последнего состояния (или 0, если done)
        # Здесь берём последнее состояние из буфера (или текущее state)
        if not done:
            with torch.no_grad():
                next_state_t = torch.tensor(state, dtype=torch.float32).to(self.device)
                next_state_t = next_state_t.unsqueeze(0)
                _, _, _, last_value = self.model.get_action_and_value(next_state_t)
                last_value = last_value.cpu().item()
        else:
            last_value = 0.0

        # Считаем advantages
        self.buffer.compute_advantages(last_value, gamma=self.gamma, lam=self.lam)

    def update(self):
        """
        Обновляем параметры Actor-Critic на основе данных в self.buffer.
        Выполняем n_epochs проходов по батчам.
        """
        for _ in range(self.n_epochs):
            for (
                states_b,
                actions_b,
                old_log_probs_b,
                returns_b,
                advantages_b,
            ) in self.buffer.get(batch_size=self.batch_size):
                states_t = torch.tensor(states_b, dtype=torch.float32).to(self.device)
                actions_t = torch.tensor(actions_b, dtype=torch.long).to(self.device)
                old_log_probs_t = torch.tensor(old_log_probs_b, dtype=torch.float32).to(
                    self.device
                )
                returns_t = torch.tensor(returns_b, dtype=torch.float32).to(self.device)
                advantages_t = torch.tensor(advantages_b, dtype=torch.float32).to(
                    self.device
                )

                # Нормировка advantages (часто помогает)
                advantages_t = (advantages_t - advantages_t.mean()) / (
                    advantages_t.std() + 1e-8
                )

                # Получаем новое распределение, значения
                logits = self.model.actor(states_t)
                dist = Categorical(logits=logits)
                log_probs_t = dist.log_prob(actions_t)

                values_t = self.model.critic(states_t).squeeze(-1)

                # Вычислим ratio = exp(new_log_prob - old_log_prob)
                ratio = torch.exp(log_probs_t - old_log_probs_t)

                # L_clip = min(ratio * A, clip(ratio, 1-eps, 1+eps) * A)
                clip_adv = (
                    torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps)
                    * advantages_t
                )
                loss_policy = -torch.mean(torch.min(ratio * advantages_t, clip_adv))

                # Entropy (для улучшения исследования)
                entropy = dist.entropy().mean()

                # Value loss
                loss_value = nn.functional.mse_loss(values_t, returns_t)

                # Финальный лосс
                loss = loss_policy + self.vf_coef * loss_value - self.ent_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def train(self, env, total_timesteps=1_000_000, log_interval=1000):
        """
        Цикл обучения: собираем n_steps опыта -> update -> повторяем
        total_timesteps: общее кол-во шагов
        log_interval   : как часто печатать лог
        """
        timesteps_done = 0
        iteration = 0

        while timesteps_done < total_timesteps:
            self.collect_trajectories(env)
            self.update()

            timesteps_done += self.n_steps
            iteration += 1

            if iteration % (log_interval // self.n_steps) == 0:
                print(f"Iteration={iteration}, timesteps_done={timesteps_done}")

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))


def validate(env, ppo_agent, n_episodes=5):
    """
    Запускаем политику ppo_agent (без обучения)
    на n_episodes в среде env.
    Возвращаем среднюю награду и среднюю длину эпизода.
    """
    total_rewards = []
    for _ in range(n_episodes):
        state = env.reset()
        done = False
        episode_reward = 0.0
        steps = 0

        while not done:
            action, _, _ = ppo_agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            episode_reward += reward
            state = next_state
            steps += 1

        total_rewards.append(episode_reward)

    mean_reward = np.mean(total_rewards)
    return mean_reward


class NodeReallocationEnv(gym.Env):
    """
    - N нод (каждая имеет лимиты по CPU и памяти).
    - M пользователей (каждый требует CPU_i, MEM_i).
    - За шаг можно (пере)назначить ровно одного пользователя (user_idx) на одну ноду (node_idx).
    - Эпизод завершается, когда либо все пользователи назначены, либо достигли max_steps.
    """

    def __init__(
        self,
        num_nodes=3,
        num_users=5,
        max_steps=20,
        cpu_caps=None,
        mem_caps=None,
        user_cpu_req=None,
        user_mem_req=None,
        overload_penalty=100.0,
    ):
        super(NodeReallocationEnv, self).__init__()

        self.num_nodes = num_nodes
        self.num_users = num_users
        self.max_steps = max_steps

        if cpu_caps is None:
            cpu_caps = np.random.uniform(3.0, 5.0, size=num_nodes)
        if mem_caps is None:
            mem_caps = np.random.uniform(8.0, 10.0, size=num_nodes)
        self.cpu_caps = np.array(cpu_caps, dtype=np.float32)
        self.mem_caps = np.array(mem_caps, dtype=np.float32)

        if user_cpu_req is None:
            user_cpu_req = np.random.uniform(0.5, 1.5, size=num_users)
        if user_mem_req is None:
            user_mem_req = np.random.uniform(1.0, 2.5, size=num_users)
        self.user_cpu_req = np.array(user_cpu_req, dtype=np.float32)
        self.user_mem_req = np.array(user_mem_req, dtype=np.float32)

        # Назначение: для каждого пользователя сохраняем индекс ноды (или -1, если не назначен)
        self.assignment = np.full(shape=(num_users,), fill_value=-1, dtype=np.int32)

        # Штраф за превышение лимитов
        self.overload_penalty = overload_penalty

        # Количество сделанных шагов
        self.current_step = 0

        # ACTION SPACE
        # Одно действие = (пользователь, нода).
        # Кодируем это одним числом: action in [0..(num_users*num_nodes - 1)]
        # user_idx = action // num_nodes
        # node_idx = action % num_nodes
        self.action_space = spaces.Discrete(num_users * num_nodes)

        # OBSERVATION SPACE
        #  - [CPU usage ratio по всем нодам]
        #  - [MEM usage ratio по всем нодам]
        #  - [assignment каждого пользователя], нормируем, чтобы лежало ~ в 0..1
        #    (если -1, то кодируем отрицательным значением)
        obs_dim = num_nodes * 2 + num_users
        low = np.zeros(obs_dim, dtype=np.float32)
        high = np.ones(obs_dim, dtype=np.float32) * 2.0  # запас
        self.observation_space = spaces.Box(
            low=low, high=high, shape=(obs_dim,), dtype=np.float32
        )

        self.reset()

    def reset(self):
        """
        Сбрасываем окружение к начальному состоянию:
         - никто не назначен (assignment = -1)
         - обнуляем счётчик шагов.
        """
        self.assignment[:] = -1
        self.current_step = 0
        return self._get_obs()

    def step(self, action):
        """
        Выполняем 1 шаг:
          - Раскодируем (user_idx, node_idx)
          - Переназначаем пользователя
          - Считаем загруженность, задержку и штрафы
          - Проверяем, не завершился ли эпизод
        """
        self.current_step += 1

        user_idx = action // self.num_nodes
        node_idx = action % self.num_nodes

        # Переназначаем данного пользователя
        self.assignment[user_idx] = node_idx

        # Считаем текущее использование CPU/Memory
        cpu_used, mem_used = self._get_usage()

        # Считаем delay: суммарная загрузка (CPU ratio + MEM ratio) по всем нодам
        # и штраф за превышение
        over_capacity = False
        delay = 0.0
        for n in range(self.num_nodes):
            # Проверка превышения
            if cpu_used[n] > self.cpu_caps[n] or mem_used[n] > self.mem_caps[n]:
                over_capacity = True
            ratio_cpu = cpu_used[n] / (self.cpu_caps[n] + 1e-8)
            ratio_mem = mem_used[n] / (self.mem_caps[n] + 1e-8)
            delay += ratio_cpu + ratio_mem

        # reward = -delay (+ штраф, если overload)
        reward = -delay
        if over_capacity:
            reward -= self.overload_penalty

        # Условие завершения
        # 1) Если все пользователи назначены (нет -1)
        all_assigned = np.all(self.assignment != -1)

        # 2) Если достигли max_steps
        time_exceeded = self.current_step >= self.max_steps

        done = bool(all_assigned or time_exceeded)

        obs = self._get_obs()
        info = {
            "delay": delay,
            "over_capacity": over_capacity,
            "all_assigned": all_assigned,
        }

        return obs, reward, done, info

    def _get_usage(self):
        """
        Возвращает (cpu_used, mem_used) по каждой ноде.
        """
        cpu_used = np.zeros(self.num_nodes, dtype=np.float32)
        mem_used = np.zeros(self.num_nodes, dtype=np.float32)

        for u in range(self.num_users):
            n = self.assignment[u]
            if n >= 0:  # пользователь назначен
                cpu_used[n] += self.user_cpu_req[u]
                mem_used[n] += self.user_mem_req[u]

        return cpu_used, mem_used

    def _get_obs(self):
        """
        Формируем наблюдение.
        - cpu ratio для каждой ноды
        - mem ratio для каждой ноды
        - "normalized assignment" для каждого пользователя
          (если assignment[u] = -1 => -1/(num_nodes-1); иначе => node/(num_nodes-1))
        """
        cpu_used, mem_used = self._get_usage()
        cpu_ratio = cpu_used / (self.cpu_caps + 1e-8)
        mem_ratio = mem_used / (self.mem_caps + 1e-8)

        # Нормируем назначение
        assign_norm = []
        for a in self.assignment:
            if a < 0:
                val = -1.0 / max((self.num_nodes - 1), 1)
            else:
                val = a / max((self.num_nodes - 1), 1)
            assign_norm.append(val)

        obs = np.concatenate([cpu_ratio, mem_ratio, assign_norm], dtype=np.float32)
        return obs

    def render(self, mode="human"):
        cpu_used, mem_used = self._get_usage()
        print(f"Step={self.current_step}")
        for n in range(self.num_nodes):
            print(
                f" Node {n}: CPU used {cpu_used[n]:.2f}/{self.cpu_caps[n]:.2f},"
                f" Mem used {mem_used[n]:.2f}/{self.mem_caps[n]:.2f}"
            )
        print(" Assignment:", self.assignment)
        print("----------")

    def close(self):
        pass



# TODO: Переключить окружение на LWMECPSEnv2
def main():
    # env = TestBanditEnv(n_actions=3, best_action=1, rng_seed=123)
    env = NodeReallocationEnv(
        num_nodes=3,
        num_users=5,
        max_steps=20,
        cpu_caps=[5.0, 5.0, 4.0],
        mem_caps=[10.0, 9.0, 8.0],
        user_cpu_req=[1.0, 1.1, 2.8, 1.2, 0.7],
        user_mem_req=[2.0, 1.5, 2.2, 1.8, 1.0],
        overload_penalty=100.0,
    )

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

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
    )

    # Запускаем обучение
    ppo_agent.train(env, total_timesteps=10_000, log_interval=2048)

    state = env.reset()
    done = False
    cum_reward = 0.0
    while not done:
        action, _, _ = ppo_agent.select_action(state)
        next_state, reward, done, info = env.step(action)
        cum_reward += reward
        state = next_state

    env.render()
    print("Cumulative reward after test episode:", cum_reward)
    # # Сохраняем
    # ppo_agent.save("./ppo_lwme_model.pth")

    # # Валидация
    # avg_reward = validate(env, ppo_agent, n_episodes=5)
    # print("Validation reward:", avg_reward)


if __name__ == "__main__":
    main()