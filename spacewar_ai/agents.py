"""
Reinforcement Learning Agents for Spacewar!

Implements DQN and PPO with action masking support.
"""

import copy
import random
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from config import DQNConfig, PPOConfig


# =============================================================================
# Neural Network Architectures
# =============================================================================

class MLP(nn.Module):
    """Multi-layer perceptron with configurable hidden layers."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_sizes: Tuple[int, ...] = (256, 256),
        activation: str = "relu",
    ):
        super().__init__()

        layers = []
        prev_size = input_dim

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "tanh":
                layers.append(nn.Tanh())
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class DuelingQNetwork(nn.Module):
    """Dueling DQN architecture: separate value and advantage streams."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_sizes: Tuple[int, ...] = (256, 256),
    ):
        super().__init__()

        # Shared feature extractor
        self.feature = nn.Sequential(
            nn.Linear(input_dim, hidden_sizes[0]),
            nn.ReLU(),
        )

        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], 1),
        )

        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature(x)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)

        # Combine: Q = V + (A - mean(A))
        q_values = value + advantage - advantage.mean(dim=-1, keepdim=True)
        return q_values


class ActorCritic(nn.Module):
    """Actor-Critic network for PPO."""

    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        hidden_sizes: Tuple[int, ...] = (256, 256),
        shared_backbone: bool = False,
    ):
        super().__init__()

        self.shared_backbone = shared_backbone

        if shared_backbone:
            self.backbone = nn.Sequential(
                nn.Linear(input_dim, hidden_sizes[0]),
                nn.ReLU(),
                nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                nn.ReLU(),
            )
            self.actor_head = nn.Linear(hidden_sizes[1], action_dim)
            self.critic_head = nn.Linear(hidden_sizes[1], 1)
        else:
            self.actor = nn.Sequential(
                nn.Linear(input_dim, hidden_sizes[0]),
                nn.ReLU(),
                nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                nn.ReLU(),
                nn.Linear(hidden_sizes[1], action_dim),
            )
            self.critic = nn.Sequential(
                nn.Linear(input_dim, hidden_sizes[0]),
                nn.ReLU(),
                nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                nn.ReLU(),
                nn.Linear(hidden_sizes[1], 1),
            )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return action logits and value estimate."""
        if self.shared_backbone:
            features = self.backbone(x)
            logits = self.actor_head(features)
            value = self.critic_head(features)
        else:
            logits = self.actor(x)
            value = self.critic(x)

        return logits, value.squeeze(-1)

    def get_action_and_value(
        self,
        x: torch.Tensor,
        action: Optional[torch.Tensor] = None,
        action_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get action, log probability, entropy, and value.

        Args:
            x: Observation tensor
            action: If provided, compute log prob for this action
            action_mask: Boolean mask for valid actions (1 = valid, 0 = invalid)

        Returns:
            action, log_prob, entropy, value
        """
        logits, value = self.forward(x)

        # Apply action mask
        if action_mask is not None:
            # Set invalid action logits to large negative number
            mask = action_mask.bool() if action_mask.dtype != torch.bool else action_mask
            logits = logits.masked_fill(~mask, float("-inf"))

        dist = Categorical(logits=logits)

        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        return action, log_prob, entropy, value


# =============================================================================
# Experience Replay
# =============================================================================

@dataclass
class Transition:
    """Single transition for replay buffer."""
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    action_mask: Optional[np.ndarray] = None


class ReplayBuffer:
    """Experience replay buffer for DQN."""

    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, transition: Transition):
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> List[Transition]:
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def __len__(self) -> int:
        return len(self.buffer)


@dataclass
class RolloutStep:
    """Single step for PPO rollout buffer."""
    state: np.ndarray
    action: int
    reward: float
    value: float
    log_prob: float
    done: bool
    action_mask: Optional[np.ndarray] = None


class RolloutBuffer:
    """Rollout buffer for PPO."""

    def __init__(self):
        self.states: List[np.ndarray] = []
        self.actions: List[int] = []
        self.rewards: List[float] = []
        self.values: List[float] = []
        self.log_probs: List[float] = []
        self.dones: List[bool] = []
        self.action_masks: List[Optional[np.ndarray]] = []

    def add(self, step: RolloutStep):
        self.states.append(step.state)
        self.actions.append(step.action)
        self.rewards.append(step.reward)
        self.values.append(step.value)
        self.log_probs.append(step.log_prob)
        self.dones.append(step.done)
        self.action_masks.append(step.action_mask)

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()
        self.action_masks.clear()

    def __len__(self) -> int:
        return len(self.states)


# =============================================================================
# Base Agent
# =============================================================================

class BaseAgent(ABC):
    """Abstract base class for all agents."""

    def __init__(self, state_dim: int, action_dim: int, device: str = "cpu"):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = torch.device(device)
        self.training_steps = 0

    @abstractmethod
    def select_action(
        self,
        state: np.ndarray,
        action_mask: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> int:
        """Select action given state."""
        pass

    @abstractmethod
    def update(self, *args, **kwargs) -> Dict[str, float]:
        """Update agent parameters."""
        pass

    @abstractmethod
    def save(self, path: str):
        """Save agent to file."""
        pass

    @abstractmethod
    def load(self, path: str):
        """Load agent from file."""
        pass

    def get_action_with_confidence(
        self,
        state: np.ndarray,
        action_mask: Optional[np.ndarray] = None,
    ) -> Tuple[int, float]:
        """Get action and confidence (for hybrid agents)."""
        action = self.select_action(state, action_mask, deterministic=False)
        return action, 0.5  # Default confidence


# =============================================================================
# DQN Agent
# =============================================================================

class DQNAgent(BaseAgent):
    """
    Deep Q-Network agent with Double DQN and Dueling architecture.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        config: Optional[DQNConfig] = None,
        device: str = "cpu",
    ):
        super().__init__(state_dim, action_dim, device)

        self.config = config or DQNConfig()

        # Networks
        self.q_network = DuelingQNetwork(
            state_dim, action_dim, self.config.hidden_sizes
        ).to(self.device)

        self.target_network = DuelingQNetwork(
            state_dim, action_dim, self.config.hidden_sizes
        ).to(self.device)

        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        # Optimizer
        self.optimizer = optim.Adam(
            self.q_network.parameters(), lr=self.config.learning_rate
        )

        # Replay buffer
        self.replay_buffer = ReplayBuffer(self.config.buffer_size)

        # Exploration
        self.epsilon = self.config.epsilon_start

    def select_action(
        self,
        state: np.ndarray,
        action_mask: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> int:
        """Select action using epsilon-greedy policy with action masking."""

        # Epsilon-greedy exploration
        if not deterministic and random.random() < self.epsilon:
            # Random valid action
            if action_mask is not None:
                valid_actions = np.where(action_mask > 0.5)[0]
                if len(valid_actions) > 0:
                    return int(np.random.choice(valid_actions))
            return random.randint(0, self.action_dim - 1)

        # Greedy action
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor).squeeze(0)

            # Apply action mask
            if action_mask is not None:
                mask_tensor = torch.FloatTensor(action_mask).to(self.device)
                q_values = q_values.masked_fill(mask_tensor < 0.5, float("-inf"))

            return int(q_values.argmax().item())

    def store_transition(self, transition: Transition):
        """Store transition in replay buffer."""
        self.replay_buffer.push(transition)

    def update(self) -> Dict[str, float]:
        """Update Q-network using experience replay."""
        if len(self.replay_buffer) < self.config.batch_size:
            return {}

        # Sample batch
        batch = self.replay_buffer.sample(self.config.batch_size)

        states = torch.FloatTensor(np.array([t.state for t in batch])).to(self.device)
        actions = torch.LongTensor([t.action for t in batch]).to(self.device)
        rewards = torch.FloatTensor([t.reward for t in batch]).to(self.device)
        next_states = torch.FloatTensor(np.array([t.next_state for t in batch])).to(
            self.device
        )
        dones = torch.FloatTensor([t.done for t in batch]).to(self.device)

        # Current Q values
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target Q values
        with torch.no_grad():
            if self.config.use_double_dqn:
                # Double DQN: use online network to select action, target to evaluate
                next_actions = self.q_network(next_states).argmax(1)
                next_q = self.target_network(next_states).gather(
                    1, next_actions.unsqueeze(1)
                ).squeeze(1)
            else:
                next_q = self.target_network(next_states).max(1)[0]

            target_q = rewards + self.config.gamma * next_q * (1 - dones)

        # Compute loss
        loss = F.mse_loss(current_q, target_q)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 10.0)
        self.optimizer.step()

        # Update target network
        self.training_steps += 1
        if self.training_steps % self.config.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        # Decay epsilon
        self._decay_epsilon()

        return {"loss": loss.item(), "epsilon": self.epsilon}

    def _decay_epsilon(self):
        """Decay exploration rate."""
        decay_rate = (
            self.config.epsilon_start - self.config.epsilon_end
        ) / self.config.epsilon_decay_steps
        self.epsilon = max(self.config.epsilon_end, self.epsilon - decay_rate)

    def get_action_with_confidence(
        self,
        state: np.ndarray,
        action_mask: Optional[np.ndarray] = None,
    ) -> Tuple[int, float]:
        """Get action and confidence based on Q-value distribution."""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor).squeeze(0)

            if action_mask is not None:
                mask_tensor = torch.FloatTensor(action_mask).to(self.device)
                q_values = q_values.masked_fill(mask_tensor < 0.5, float("-inf"))

            # Softmax to get "confidence"
            probs = F.softmax(q_values, dim=0)
            action = int(q_values.argmax().item())
            confidence = float(probs[action].item())

        return action, confidence

    def save(self, path: str):
        """Save agent state."""
        torch.save(
            {
                "q_network": self.q_network.state_dict(),
                "target_network": self.target_network.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
                "training_steps": self.training_steps,
                "config": self.config,
            },
            path,
        )

    def load(self, path: str):
        """Load agent state."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.q_network.load_state_dict(checkpoint["q_network"])
        self.target_network.load_state_dict(checkpoint["target_network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon = checkpoint["epsilon"]
        self.training_steps = checkpoint["training_steps"]


# =============================================================================
# PPO Agent
# =============================================================================

class PPOAgent(BaseAgent):
    """
    Proximal Policy Optimization agent with GAE and action masking.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        config: Optional[PPOConfig] = None,
        device: str = "cpu",
    ):
        super().__init__(state_dim, action_dim, device)

        self.config = config or PPOConfig()

        # Actor-Critic network
        self.network = ActorCritic(
            state_dim, action_dim, self.config.hidden_sizes, shared_backbone=False
        ).to(self.device)

        # Optimizer
        self.optimizer = optim.Adam(
            self.network.parameters(), lr=self.config.learning_rate
        )

        # Rollout buffer
        self.rollout_buffer = RolloutBuffer()

        # Running stats for observation normalization
        self.obs_mean = np.zeros(state_dim, dtype=np.float32)
        self.obs_var = np.ones(state_dim, dtype=np.float32)
        self.obs_count = 1e-4

    def select_action(
        self,
        state: np.ndarray,
        action_mask: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> int:
        """Select action from policy."""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

            mask_tensor = None
            if action_mask is not None:
                mask_tensor = torch.FloatTensor(action_mask).unsqueeze(0).to(self.device)

            if deterministic:
                logits, _ = self.network(state_tensor)
                if mask_tensor is not None:
                    logits = logits.masked_fill(mask_tensor < 0.5, float("-inf"))
                return int(logits.argmax(dim=-1).item())

            action, _, _, _ = self.network.get_action_and_value(
                state_tensor, action_mask=mask_tensor
            )
            return int(action.item())

    def get_value(self, state: np.ndarray) -> float:
        """Get value estimate for state."""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            _, value = self.network(state_tensor)
            return float(value.item())

    def get_action_and_value_for_storage(
        self,
        state: np.ndarray,
        action_mask: Optional[np.ndarray] = None,
    ) -> Tuple[int, float, float]:
        """Get action, log_prob, and value for storing in rollout buffer."""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

            mask_tensor = None
            if action_mask is not None:
                mask_tensor = torch.FloatTensor(action_mask).unsqueeze(0).to(self.device)

            action, log_prob, _, value = self.network.get_action_and_value(
                state_tensor, action_mask=mask_tensor
            )

            return int(action.item()), float(log_prob.item()), float(value.item())

    def store_transition(self, step: RolloutStep):
        """Store step in rollout buffer."""
        self.rollout_buffer.add(step)

    def update(self, next_value: float = 0.0) -> Dict[str, float]:
        """Update policy using collected rollout."""
        if len(self.rollout_buffer) == 0:
            return {}

        # Compute returns and advantages using GAE
        returns, advantages = self._compute_gae(next_value)

        # Convert to tensors
        states = torch.FloatTensor(np.array(self.rollout_buffer.states)).to(self.device)
        actions = torch.LongTensor(self.rollout_buffer.actions).to(self.device)
        old_log_probs = torch.FloatTensor(self.rollout_buffer.log_probs).to(self.device)
        returns_tensor = torch.FloatTensor(returns).to(self.device)
        advantages_tensor = torch.FloatTensor(advantages).to(self.device)

        # Handle action masks
        action_masks = self.rollout_buffer.action_masks
        if action_masks[0] is not None:
            masks_tensor = torch.FloatTensor(np.array(action_masks)).to(self.device)
        else:
            masks_tensor = None

        # Normalize advantages
        if self.config.normalize_advantage:
            advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (
                advantages_tensor.std() + 1e-8
            )

        # PPO update epochs
        total_loss = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0

        n_samples = len(self.rollout_buffer)
        indices = np.arange(n_samples)

        for _ in range(self.config.n_epochs):
            np.random.shuffle(indices)

            for start in range(0, n_samples, self.config.batch_size):
                end = start + self.config.batch_size
                batch_indices = indices[start:end]

                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_returns = returns_tensor[batch_indices]
                batch_advantages = advantages_tensor[batch_indices]

                batch_masks = None
                if masks_tensor is not None:
                    batch_masks = masks_tensor[batch_indices]

                # Get current policy outputs
                _, new_log_probs, entropy, values = self.network.get_action_and_value(
                    batch_states, action=batch_actions, action_mask=batch_masks
                )

                # Policy loss (clipped surrogate)
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = (
                    torch.clamp(
                        ratio, 1 - self.config.clip_range, 1 + self.config.clip_range
                    )
                    * batch_advantages
                )
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = F.mse_loss(values, batch_returns)

                # Entropy bonus
                entropy_loss = -entropy.mean()

                # Total loss
                loss = (
                    policy_loss
                    + self.config.value_coef * value_loss
                    + self.config.entropy_coef * entropy_loss
                )

                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.network.parameters(), self.config.max_grad_norm
                )
                self.optimizer.step()

                total_loss += loss.item()
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()

        # Clear buffer
        n_updates = self.config.n_epochs * (n_samples // self.config.batch_size + 1)
        self.rollout_buffer.clear()
        self.training_steps += 1

        return {
            "loss": total_loss / n_updates,
            "policy_loss": total_policy_loss / n_updates,
            "value_loss": total_value_loss / n_updates,
            "entropy": total_entropy / n_updates,
        }

    def _compute_gae(self, next_value: float) -> Tuple[np.ndarray, np.ndarray]:
        """Compute returns and advantages using Generalized Advantage Estimation."""
        rewards = np.array(self.rollout_buffer.rewards)
        values = np.array(self.rollout_buffer.values)
        dones = np.array(self.rollout_buffer.dones)

        n = len(rewards)
        returns = np.zeros(n, dtype=np.float32)
        advantages = np.zeros(n, dtype=np.float32)

        last_gae = 0.0
        last_value = next_value

        for t in reversed(range(n)):
            if dones[t]:
                next_value = 0.0
                last_gae = 0.0
            else:
                next_value = last_value

            delta = rewards[t] + self.config.gamma * next_value - values[t]
            last_gae = delta + self.config.gamma * self.config.gae_lambda * last_gae

            advantages[t] = last_gae
            returns[t] = advantages[t] + values[t]

            last_value = values[t]

        return returns, advantages

    def get_action_with_confidence(
        self,
        state: np.ndarray,
        action_mask: Optional[np.ndarray] = None,
    ) -> Tuple[int, float]:
        """Get action and confidence based on policy entropy."""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            logits, _ = self.network(state_tensor)

            if action_mask is not None:
                mask_tensor = torch.FloatTensor(action_mask).to(self.device)
                logits = logits.masked_fill(mask_tensor < 0.5, float("-inf"))

            probs = F.softmax(logits.squeeze(0), dim=0)
            action = int(probs.argmax().item())
            confidence = float(probs[action].item())

        return action, confidence

    def save(self, path: str):
        """Save agent state."""
        torch.save(
            {
                "network": self.network.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "training_steps": self.training_steps,
                "obs_mean": self.obs_mean,
                "obs_var": self.obs_var,
                "obs_count": self.obs_count,
                "config": self.config,
            },
            path,
        )

    def load(self, path: str):
        """Load agent state."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.network.load_state_dict(checkpoint["network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.training_steps = checkpoint["training_steps"]
        self.obs_mean = checkpoint.get("obs_mean", self.obs_mean)
        self.obs_var = checkpoint.get("obs_var", self.obs_var)
        self.obs_count = checkpoint.get("obs_count", self.obs_count)


# =============================================================================
# Agent Factory
# =============================================================================

def create_agent(
    algorithm: str,
    state_dim: int,
    action_dim: int,
    config: Any = None,
    device: str = "cpu",
) -> BaseAgent:
    """Create agent by algorithm name."""
    if algorithm.lower() == "dqn":
        return DQNAgent(state_dim, action_dim, config, device)
    elif algorithm.lower() == "ppo":
        return PPOAgent(state_dim, action_dim, config, device)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
