import gymnasium as gym


class LWMECPSEnv(gym.Env):

    def __init__(self, window_size, render_mode=None):
        self.render_mode = render_mode
        self.window_size = window_size

        # spaces https://gymnasium.farama.org/api/spaces/composite/
        self.action_space = gym.spaces.
        
        # spaces https://gymnasium.farama.org/api/spaces/composite/
        self.observation_space = gym.spaces.

    def reset(self, seed=None, options=None):
        return observation, info
    

    def step(self, action):
        return observation, reward, terminated, False, info