"""
Spacewar! AI Configuration

All hyperparameters and settings in one place.
"""

from dataclasses import dataclass, field
from typing import Optional
import torch


@dataclass
class GameConfig:
    """Game physics and mechanics configuration."""

    # Arena
    width: int = 800
    height: int = 600

    # Physics
    gravity_constant: float = 5000.0
    star_mass: float = 100.0
    star_radius: float = 25.0
    ship_mass: float = 1.0
    ship_radius: float = 15.0
    torpedo_radius: float = 3.0
    torpedo_speed: float = 300.0
    torpedo_lifespan: float = 3.0  # seconds

    # Ship properties
    thrust_power: float = 150.0
    rotation_speed: float = 180.0  # degrees per second
    max_fuel: float = 100.0
    fuel_consumption: float = 10.0  # per second of thrust
    max_ammo: int = 15
    fire_cooldown: float = 0.5  # seconds

    # Episode
    max_episode_steps: int = 3000
    fps: int = 50

    # Rendering
    render_scale: float = 1.0


@dataclass
class DQNConfig:
    """DQN algorithm hyperparameters."""

    learning_rate: float = 1e-4
    gamma: float = 0.99
    buffer_size: int = 100_000
    batch_size: int = 64
    target_update_freq: int = 1000

    # Exploration
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 100_000

    # Network
    hidden_sizes: tuple = (256, 256)

    # Double DQN
    use_double_dqn: bool = True


@dataclass
class PPOConfig:
    """PPO algorithm hyperparameters."""

    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5

    # Training
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10

    # Network
    hidden_sizes: tuple = (256, 256)

    # Normalization
    normalize_advantage: bool = True


@dataclass
class RewardConfig:
    """Reward function weights."""

    # Terminal rewards
    kill_opponent: float = 100.0
    killed_by_opponent: float = -100.0
    star_collision: float = -50.0
    ship_collision: float = -30.0

    # Episode outcome
    win_bonus: float = 50.0
    lose_penalty: float = -50.0
    draw_penalty: float = -10.0

    # Efficiency bonuses (awarded to winner based on resources remaining)
    # These encourage winning efficiently rather than wastefully
    fuel_efficiency_bonus: float = 20.0  # max bonus for 100% fuel remaining
    ammo_efficiency_bonus: float = 15.0  # max bonus for 100% ammo remaining

    # Shaping (kept small to avoid reward hacking)
    time_penalty: float = -0.01  # per step
    fuel_empty_penalty: float = -5.0
    ammo_empty_penalty: float = -2.0

    # Distance-based (potential shaping)
    safe_star_distance: float = 100.0  # min safe distance from star


@dataclass
class SelfPlayConfig:
    """Self-play and league training configuration."""

    # Policy snapshots
    snapshot_interval: int = 100  # episodes between snapshots
    max_snapshots: int = 5

    # Opponent selection
    self_play_ratio: float = 0.3  # play against current self
    past_policy_ratio: float = 0.7  # play against past snapshots

    # Evaluation
    eval_interval: int = 50  # episodes between evaluation
    eval_episodes: int = 20


@dataclass
class LLMConfig:
    """LLM integration configuration."""

    # API settings
    keys_dir: str = "."
    preferred_providers: tuple = ("anthropic", "google", "openai", "groq", "together", "deepseek", "mistral")

    # Local LLM
    use_local: bool = False
    local_model: Optional[str] = None  # Auto-select if None
    local_context_size: int = 2048

    # Usage mode (training guidance only - not real-time)
    demonstration_episodes: int = 100  # Generate this many LLM demonstrations
    imitation_weight: float = 0.1  # Weight for imitation loss

    # Query settings
    temperature: float = 0.3
    max_tokens: int = 10


@dataclass
class TrainingConfig:
    """Overall training configuration."""

    algorithm: str = "ppo"  # "dqn" or "ppo"
    total_episodes: int = 10_000

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    checkpoint_interval: int = 100

    # Logging
    log_interval: int = 10
    use_tensorboard: bool = True

    # Reproducibility
    seed: int = 42

    # Hardware
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")

    # Rendering
    render_training: bool = False
    render_eval: bool = True

    # Early stopping
    early_stop_reward: Optional[float] = None  # Stop if avg reward exceeds this
    early_stop_window: int = 100  # Window for computing average

    # LLM guidance
    use_llm_guidance: bool = False


@dataclass
class Config:
    """Master configuration combining all sub-configs."""

    game: GameConfig = field(default_factory=GameConfig)
    dqn: DQNConfig = field(default_factory=DQNConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    self_play: SelfPlayConfig = field(default_factory=SelfPlayConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def __post_init__(self):
        """Validate configuration."""
        if self.training.algorithm not in ("dqn", "ppo"):
            raise ValueError(f"Unknown algorithm: {self.training.algorithm}")


def get_default_config() -> Config:
    """Return default configuration."""
    return Config()
