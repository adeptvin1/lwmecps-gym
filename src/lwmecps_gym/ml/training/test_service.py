from lwmecps_gym.envs.kubernetes_api import k8s
from gymnasium.envs.registration import register
import gymnasium as gym
import bitmath
import re
from time import time
from lwmecps_gym.ml.models.ppo_learning import PPO
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

register(
    id='lwmecps2-v0',
    entry_point='lwmecps_gym.envs:LWMECPSEnv2',
)

def init_kubernetes_client():
    """
    Initialize and verify Kubernetes client connection.
    Returns:
        k8s: Initialized Kubernetes client
    Raises:
        Exception: If client initialization fails
    """
    try:
        k8s_client = k8s()
        # Verify connection by listing namespaces
        k8s_client.core_api.list_namespace()
        logger.info("Successfully initialized Kubernetes client")
        return k8s_client
    except Exception as e:
        logger.error(f"Failed to initialize Kubernetes client: {str(e)}")
        raise

def main():
    try:
        # Initialize Kubernetes client
        k8s_client = init_kubernetes_client()
        node_name = []

        max_hardware = {
            'cpu': 8,
            'ram': 16000,
            'tx_bandwidth': 1000,
            'rx_bandwidth': 1000,
            'read_disks_bandwidth': 500,
            'write_disks_bandwidth': 500,
            'avg_latency': 300
        }

        pod_usage = {
            'cpu': 2,
            'ram': 2000,
            'tx_bandwidth': 20,
            'rx_bandwidth': 80,
            'read_disks_bandwidth': 100,
            'write_disks_bandwidth': 100
        }

        # Get cluster state
        state = k8s_client.k8s_state()
        if not state:
            raise Exception("Failed to get cluster state")

        for node in state:
            node_name.append(node)

        if not node_name:
            raise Exception("No nodes found in the cluster")

        avg_latency = 10
        node_info = {}
        max_pods = 10000

        for node in state:
            avg_latency = avg_latency + 10
            node_info[node] = {
                'cpu': int(state[node]['cpu']),
                'ram': round(bitmath.KiB(
                    int(re.findall(r'\d+', state[node]['memory'])[0])).to_MB().value),
                'tx_bandwidth': 100,
                'rx_bandwidth': 100,
                'read_disks_bandwidth': 300,
                'write_disks_bandwidth': 300,
                'avg_latency': avg_latency
            }
            # Calculate maximum number of pods
            max_pods = min([
                min([
                    val // pod_usage[key]
                    for key, (_, val) in zip(pod_usage.keys(), node_info[node].items())
                ]),
                max_pods,
            ])

        # Create environment
        env = gym.make('lwmecps2-v0',
                      num_nodes=len(node_name),
                      node_name=node_name,
                      max_hardware=max_hardware,
                      pod_usage=pod_usage,
                      node_info=node_info,
                      deployment_name='mec-test-app',
                      namespace='default',
                      deployments=['mec-test-app'],
                      max_pods=max_pods)
        start = time()
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.n

        # Create and train agent
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
        ppo_agent.train(env, total_timesteps=10_000, log_interval=2048)


        print(f"Training time: {(time() - start)}")
        ppo_agent.save_q_table("./ppo_table.pkl")
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

    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()
