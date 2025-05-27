from gymnasium.envs.registration import register

register(
    id='lwmecps-v0',
    entry_point='lwmecps_gym.envs:LWMECPSEnv',
    kwargs={
    }
)

register(
    id='lwmecps-v2',
    entry_point='lwmecps_gym.envs:LWMECPSEnv2',
    kwargs={
    }
)

register(
    id='lwmecps-v3',
    entry_point='lwmecps_gym.envs:LWMECPSEnv3',
    kwargs={
    }
)
