import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
import wandb
from abc import ABC, abstractmethod
from pathlib import Path
import json

from lwmecps_gym.core.wandb_config import log_metrics
from lwmecps_gym.core.models import ModelType

logger = logging.getLogger(__name__)

class TransferType:
    """Types of transfer learning approaches"""
    FEATURE_EXTRACTION = "feature_extraction"  # Freeze all layers except last
    FINE_TUNING = "fine_tuning"               # Fine-tune all layers with low LR
    LAYER_WISE = "layer_wise"                 # Progressive unfreezing
    DOMAIN_ADAPTATION = "domain_adaptation"    # Domain adversarial training

class TransferLearningAgent(ABC):
    """
    Abstract base class for transfer learning agents.
    Provides common functionality for transferring knowledge between tasks.
    """
    
    def __init__(
        self,
        source_model_path: str,
        target_obs_dim: int,
        target_act_dim: int,
        transfer_type: str = TransferType.FINE_TUNING,
        frozen_layers: Optional[List[int]] = None,
        learning_rate: float = 1e-4,
        device: str = "cpu",
        max_replicas: int = 10,
        deployments: List[str] = None
    ):
        self.source_model_path = source_model_path
        self.target_obs_dim = target_obs_dim
        self.target_act_dim = target_act_dim
        self.transfer_type = transfer_type
        self.frozen_layers = frozen_layers or []
        self.learning_rate = learning_rate
        self.device = device
        self.max_replicas = max_replicas
        self.deployments = deployments or []
        
        # Load source model
        self.source_model = self._load_source_model()
        self.target_model = self._create_target_model()
        
        # Setup optimizer
        self.optimizer = self._setup_optimizer()
        
        # Metrics tracking
        self.metrics_collector = TransferMetricsCollector()
        
    @abstractmethod
    def _load_source_model(self):
        """Load the pre-trained source model"""
        pass
        
    @abstractmethod
    def _create_target_model(self):
        """Create target model architecture"""
        pass
        
    @abstractmethod
    def _setup_optimizer(self):
        """Setup optimizer with appropriate learning rates"""
        pass
        
    def freeze_layers(self, layer_indices: List[int]):
        """Freeze specified layers"""
        for idx in layer_indices:
            if idx < len(list(self.target_model.parameters())):
                for param in list(self.target_model.parameters())[idx]:
                    param.requires_grad = False
                    
    def unfreeze_layers(self, layer_indices: List[int]):
        """Unfreeze specified layers"""
        for idx in layer_indices:
            if idx < len(list(self.target_model.parameters())):
                for param in list(self.target_model.parameters())[idx]:
                    param.requires_grad = True
                    
    def progressive_unfreezing(self, epoch: int, total_epochs: int):
        """Progressively unfreeze layers during training"""
        if self.transfer_type == TransferType.LAYER_WISE:
            # Unfreeze layers progressively
            layers_to_unfreeze = int((epoch / total_epochs) * len(self.frozen_layers))
            if layers_to_unfreeze > 0:
                self.unfreeze_layers(self.frozen_layers[:layers_to_unfreeze])
                
    def select_action(self, obs: torch.Tensor) -> torch.Tensor:
        """Select action using target model"""
        with torch.no_grad():
            if hasattr(self.target_model, 'select_action'):
                return self.target_model.select_action(obs)
            else:
                # Default action selection for models without select_action method
                logits, _ = self.target_model(obs)
                if hasattr(self.target_model, 'act_dim'):
                    logits = logits.view(-1, self.target_model.act_dim, self.max_replicas + 1)
                    dist = torch.distributions.Categorical(logits=logits)
                    return dist.sample()
                else:
                    return torch.argmax(logits, dim=-1)
                    
    def train(
        self, 
        env, 
        total_episodes: int = 100,
        wandb_run_id: str = None,
        training_service=None,
        task_id: str = None,
        loop=None,
        db_connection=None
    ) -> Dict[str, List[float]]:
        """
        Train the transfer learning model
        
        Args:
            env: Environment for training
            total_episodes: Number of episodes to train
            wandb_run_id: Weights & Biases run ID
            training_service: Training service instance
            task_id: Task ID for progress tracking
            loop: Event loop for async operations
            db_connection: Database connection
            
        Returns:
            Dictionary of training metrics
        """
        logger.info(f"Starting transfer learning training for {total_episodes} episodes")
        logger.info(f"Transfer type: {self.transfer_type}")
        logger.info(f"Frozen layers: {self.frozen_layers}")
        
        # Initialize metrics
        episode_rewards = []
        episode_lengths = []
        losses = []
        transfer_metrics = []
        
        for episode in range(total_episodes):
            obs, info = env.reset()
            if info.get("group_completed"):
                logger.info("Experiment group finished (detected at reset). Stopping transfer learning.")
                break

            episode_reward = 0
            episode_length = 0
            episode_losses = []
            
            # Progressive unfreezing
            self.progressive_unfreezing(episode, total_episodes)
            
            done = False
            while not done:
                # Select action
                action = self.select_action(torch.FloatTensor(obs).unsqueeze(0))
                
                # Take step
                next_obs, reward, terminated, truncated, info = env.step(action.numpy())
                done = terminated or truncated
                
                if info.get("group_completed"):
                    logger.info("Experiment group finished. Stopping transfer learning.")
                    done = True

                # Store experience
                episode_reward += reward
                episode_length += 1
                
                # Update model (simplified for demonstration)
                loss = self._compute_loss(obs, action, reward, next_obs, done)
                if loss is not None:
                    episode_losses.append(loss.item())
                    
                    # Backward pass
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                
                obs = next_obs

                if info.get("group_completed"):
                    break
            
            if info.get("group_completed"):
                break
                
            # Store episode metrics
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            if episode_losses:
                losses.append(np.mean(episode_losses))
            
            # Calculate transfer learning specific metrics
            transfer_metric = self._calculate_transfer_metric()
            transfer_metrics.append(transfer_metric)
            
            # Log to wandb
            if wandb_run_id:
                log_metrics({
                    "transfer/episode_reward": episode_reward,
                    "transfer/episode_length": episode_length,
                    "transfer/avg_loss": np.mean(episode_losses) if episode_losses else 0,
                    "transfer/transfer_metric": transfer_metric,
                    "transfer/frozen_layers": len(self.frozen_layers)
                }, step=episode)
            
            # Update progress in database
            if training_service and task_id and loop and db_connection:
                progress = (episode / total_episodes) * 100
                metrics = {
                    "total_reward": episode_reward,
                    "steps": episode_length,
                    "loss": np.mean(episode_losses) if episode_losses else 0,
                    "transfer_metric": transfer_metric
                }
                
                future = asyncio.run_coroutine_threadsafe(
                    training_service.save_training_result(task_id, episode, metrics, db_connection),
                    loop
                )
                future.result()
            
            # Log progress
            if episode % 10 == 0:
                logger.info(f"Episode {episode}: Reward={episode_reward:.2f}, "
                           f"Length={episode_length}, Loss={np.mean(episode_losses) if episode_losses else 0:.4f}")
        
        # Store final metrics
        self.metrics_collector.update({
            "episode_rewards": episode_rewards,
            "episode_lengths": episode_lengths,
            "losses": losses,
            "transfer_metrics": transfer_metrics
        })
        
        logger.info("Transfer learning training completed")
        return {
            "episode_rewards": episode_rewards,
            "episode_lengths": episode_lengths,
            "losses": losses,
            "transfer_metrics": transfer_metrics
        }
    
    @abstractmethod
    def _compute_loss(self, obs, action, reward, next_obs, done):
        """Compute loss for the current experience"""
        pass
        
    def _calculate_transfer_metric(self) -> float:
        """Calculate transfer learning specific metric"""
        # Example: Measure how much the model has changed from source
        if hasattr(self, 'source_model') and hasattr(self, 'target_model'):
            # Calculate parameter distance
            source_params = torch.cat([p.flatten() for p in self.source_model.parameters()])
            target_params = torch.cat([p.flatten() for p in self.target_model.parameters()])
            distance = torch.norm(target_params - source_params).item()
            return distance
        return 0.0
        
    def save_model(self, path: str):
        """Save the trained model"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save model state and metadata
        torch.save({
            'model_state_dict': self.target_model.state_dict(),
            'obs_dim': self.target_obs_dim,
            'act_dim': self.target_act_dim,
            'transfer_type': self.transfer_type,
            'frozen_layers': self.frozen_layers,
            'deployments': self.deployments,
            'max_replicas': self.max_replicas,
            'model_type': 'transfer_learning'
        }, path)
        
        logger.info(f"Transfer learning model saved to {path}")
        
    def load_model(self, path: str):
        """Load a trained model"""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        self.target_model.load_state_dict(checkpoint['model_state_dict'])
        self.transfer_type = checkpoint.get('transfer_type', TransferType.FINE_TUNING)
        self.frozen_layers = checkpoint.get('frozen_layers', [])
        self.deployments = checkpoint.get('deployments', [])
        self.max_replicas = checkpoint.get('max_replicas', 10)
        
        logger.info(f"Transfer learning model loaded from {path}")

class TransferMetricsCollector:
    """Collects and validates transfer learning specific metrics"""
    
    def __init__(self):
        self.metrics = {}
        self.metric_validators = {
            "episode_rewards": lambda x: True,  # Can be any value
            "episode_lengths": lambda x: x >= 0,
            "losses": lambda x: x >= 0,
            "transfer_metrics": lambda x: x >= 0,
            "parameter_distance": lambda x: x >= 0,
            "adaptation_speed": lambda x: x >= 0
        }
    
    def update(self, metrics_dict: Dict[str, Any]):
        """Update metrics with new values"""
        for key, value in metrics_dict.items():
            if key not in self.metrics:
                self.metrics[key] = []
            
            if isinstance(value, list):
                self.metrics[key].extend(value)
            else:
                self.metrics[key].append(value)
    
    def get_average_metrics(self) -> Dict[str, float]:
        """Calculate average values for all metrics"""
        return {key: np.mean(values) for key, values in self.metrics.items() if values}
    
    def get_latest_metrics(self) -> Dict[str, float]:
        """Get the most recent values for all metrics"""
        return {key: values[-1] for key, values in self.metrics.items() if values}
    
    def reset(self):
        """Reset all metrics"""
        self.metrics.clear()

class PPOTransferAgent(TransferLearningAgent):
    """Transfer Learning implementation for PPO models"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def _load_source_model(self):
        """Load PPO source model"""
        from .ppo_learning import ActorCritic
        
        checkpoint = torch.load(self.source_model_path, map_location=self.device, weights_only=False)
        
        # Extract dimensions from checkpoint
        obs_dim = checkpoint.get('obs_dim', self.target_obs_dim)
        act_dim = checkpoint.get('act_dim', self.target_act_dim)
        hidden_size = checkpoint.get('hidden_size', 256)
        max_replicas = checkpoint.get('max_replicas', self.max_replicas)
        
        # Create source model
        source_model = ActorCritic(obs_dim, act_dim, hidden_size, max_replicas)
        source_model.load_state_dict(checkpoint['model_state_dict'])
        
        return source_model
        
    def _create_target_model(self):
        """Create target PPO model"""
        from .ppo_learning import ActorCritic
        
        # Create target model with new dimensions
        target_model = ActorCritic(
            self.target_obs_dim, 
            self.target_act_dim, 
            hidden_size=256, 
            max_replicas=self.max_replicas
        )
        
        # Initialize with source model weights where possible
        self._transfer_weights(self.source_model, target_model)
        
        return target_model
        
    def _transfer_weights(self, source_model, target_model):
        """Transfer weights from source to target model"""
        source_state = source_model.state_dict()
        target_state = target_model.state_dict()
        
        # Transfer compatible layers
        for name, param in source_state.items():
            if name in target_state:
                if param.shape == target_state[name].shape:
                    target_state[name] = param
                    logger.info(f"Transferred weights for layer: {name}")
                else:
                    logger.warning(f"Shape mismatch for layer {name}: "
                                 f"source {param.shape} vs target {target_state[name].shape}")
        
        target_model.load_state_dict(target_state)
        
    def _setup_optimizer(self):
        """Setup optimizer with different learning rates for frozen/unfrozen layers"""
        # Different learning rates for different parts
        frozen_params = []
        unfrozen_params = []
        
        for idx, param in enumerate(self.target_model.parameters()):
            if idx in self.frozen_layers:
                frozen_params.append(param)
            else:
                unfrozen_params.append(param)
        
        # Use different learning rates
        param_groups = []
        if unfrozen_params:
            param_groups.append({'params': unfrozen_params, 'lr': self.learning_rate})
        if frozen_params:
            param_groups.append({'params': frozen_params, 'lr': self.learning_rate * 0.1})
            
        return optim.Adam(param_groups)
        
    def _compute_loss(self, obs, action, reward, next_obs, done):
        """Compute PPO loss"""
        # Simplified PPO loss computation
        # In practice, you'd need to implement proper PPO loss with value function
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
        
        if hasattr(self.target_model, 'get_action_and_value'):
            _, log_prob, _, value = self.target_model.get_action_and_value(obs_tensor)
            # Simplified loss - in practice use proper PPO loss
            return -log_prob.mean()  # Policy gradient loss
        else:
            logits, value = self.target_model(obs_tensor)
            # Cross-entropy loss for discrete actions
            action_tensor = torch.LongTensor(action).unsqueeze(0)
            loss = nn.CrossEntropyLoss()(logits, action_tensor)
            return loss

class SACTransferAgent(TransferLearningAgent):
    """Transfer Learning implementation for SAC models"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def _load_source_model(self):
        """Load SAC source model"""
        from .sac_learning import SAC
        
        checkpoint = torch.load(self.source_model_path, map_location=self.device, weights_only=False)
        
        # Extract parameters
        obs_dim = checkpoint.get('obs_dim', self.target_obs_dim)
        act_dim = checkpoint.get('act_dim', self.target_act_dim)
        hidden_size = checkpoint.get('hidden_size', 256)
        max_replicas = checkpoint.get('max_replicas', self.max_replicas)
        
        # Create source SAC model
        source_model = SAC(
            obs_dim=obs_dim,
            act_dim=act_dim,
            hidden_size=hidden_size,
            max_replicas=max_replicas,
            device=self.device
        )
        
        # Load state dict
        source_model.load_state_dict(checkpoint['model_state_dict'])
        
        return source_model
        
    def _create_target_model(self):
        """Create target SAC model"""
        from .sac_learning import SAC
        
        # Create target SAC model
        target_model = SAC(
            obs_dim=self.target_obs_dim,
            act_dim=self.target_act_dim,
            hidden_size=256,
            max_replicas=self.max_replicas,
            device=self.device
        )
        
        # Transfer weights
        self._transfer_weights(self.source_model, target_model)
        
        return target_model
        
    def _transfer_weights(self, source_model, target_model):
        """Transfer weights from source to target SAC model"""
        # Transfer actor and critic networks separately
        self._transfer_network_weights(source_model.actor, target_model.actor)
        self._transfer_network_weights(source_model.critic1, target_model.critic1)
        self._transfer_network_weights(source_model.critic2, target_model.critic2)
        
    def _transfer_network_weights(self, source_net, target_net):
        """Transfer weights between two networks"""
        source_state = source_net.state_dict()
        target_state = target_net.state_dict()
        
        for name, param in source_state.items():
            if name in target_state and param.shape == target_state[name].shape:
                target_state[name] = param
                
        target_net.load_state_dict(target_state)
        
    def _setup_optimizer(self):
        """Setup SAC optimizer"""
        return optim.Adam(self.target_model.parameters(), lr=self.learning_rate)
        
    def _compute_loss(self, obs, action, reward, next_obs, done):
        """Compute SAC loss"""
        # Simplified SAC loss - in practice implement full SAC loss
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
        action_tensor = torch.FloatTensor(action).unsqueeze(0)
        
        # Actor loss
        if hasattr(self.target_model, 'actor'):
            actor_loss = -self.target_model.critic1(obs_tensor, action_tensor).mean()
            return actor_loss
        else:
            return torch.tensor(0.0, requires_grad=True)
