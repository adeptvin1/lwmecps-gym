from gymnasium.envs.registration import register

register(
    id='lwmecps-v0',
    entry_point='lwmecps_gym.envs:LWMECPSEnv',
    kwargs={
    }
)