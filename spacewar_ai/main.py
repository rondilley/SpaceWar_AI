#!/usr/bin/env python3
"""
Spacewar! AI - Main Entry Point

A faithful recreation of the 1962 classic with modern RL agents.

Usage:
    # Train with PPO
    python main.py --algorithm ppo --episodes 10000 --render

    # Train with DQN
    python main.py --algorithm dqn --episodes 10000

    # With LLM-guided exploration
    python main.py --algorithm ppo --use-llm --llm-provider groq

    # Evaluate trained model
    python main.py --evaluate --checkpoint checkpoints/best_model.pt

    # Watch demo
    python main.py --demo
"""

import argparse
import logging
import os
import random
import sys
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import torch

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from config import Config, get_default_config
from environment import SpacewarEnv, Actions
from agents import DQNAgent, PPOAgent, BaseAgent, create_agent, Transition, RolloutStep
from llm_integration import (
    APIKeyManager,
    create_llm_client,
    _create_single_client,
    LLMExplorationGuide,
    DemonstrationGenerator,
    detect_hardware,
)
from tournament import (
    TournamentManager,
    LLMArena,
    AgentArena,
    ChampionTrainer,
    ELOSystem,
    Competitor,
    LLMCompetitor,
    AgentCompetitor,
    HeuristicCompetitor,
    create_llm_competitors,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Self-Play League Training
# =============================================================================

class PolicySnapshot:
    """Snapshot of a policy for league training."""

    def __init__(self, agent: BaseAgent, episode: int, win_rate: float):
        self.state_dict = {
            k: v.cpu().clone() for k, v in agent.network.state_dict().items()
        } if hasattr(agent, 'network') else {
            k: v.cpu().clone() for k, v in agent.q_network.state_dict().items()
        }
        self.episode = episode
        self.win_rate = win_rate

    def load_into(self, agent: BaseAgent):
        """Load snapshot into an agent."""
        if hasattr(agent, 'network'):
            agent.network.load_state_dict(self.state_dict)
        else:
            agent.q_network.load_state_dict(self.state_dict)


class SelfPlayManager:
    """Manages self-play training with league of past policies."""

    def __init__(self, config: Config, agent: BaseAgent):
        self.config = config.self_play
        self.agent = agent
        self.snapshots: List[PolicySnapshot] = []
        self.opponent_agent: Optional[BaseAgent] = None

        # Create opponent agent (same architecture)
        self._create_opponent()

        # Stats
        self.wins = 0
        self.losses = 0
        self.draws = 0

    def _create_opponent(self):
        """Create opponent agent with same architecture."""
        if isinstance(self.agent, PPOAgent):
            self.opponent_agent = PPOAgent(
                self.agent.state_dim,
                self.agent.action_dim,
                self.agent.config,
                str(self.agent.device),
            )
        else:
            self.opponent_agent = DQNAgent(
                self.agent.state_dim,
                self.agent.action_dim,
                self.agent.config,
                str(self.agent.device),
            )

        # Initialize with same weights
        if hasattr(self.agent, 'network'):
            self.opponent_agent.network.load_state_dict(
                self.agent.network.state_dict()
            )
        else:
            self.opponent_agent.q_network.load_state_dict(
                self.agent.q_network.state_dict()
            )

    def maybe_snapshot(self, episode: int, win_rate: float):
        """Take a policy snapshot if it's time."""
        if episode % self.config.snapshot_interval == 0:
            snapshot = PolicySnapshot(self.agent, episode, win_rate)
            self.snapshots.append(snapshot)

            # Keep only recent snapshots
            if len(self.snapshots) > self.config.max_snapshots:
                self.snapshots.pop(0)

            logger.info(f"Saved policy snapshot at episode {episode} (win rate: {win_rate:.2%})")

    def select_opponent(self) -> BaseAgent:
        """Select opponent for next episode."""
        if not self.snapshots or random.random() < self.config.self_play_ratio:
            # Play against current self
            if hasattr(self.agent, 'network'):
                self.opponent_agent.network.load_state_dict(
                    self.agent.network.state_dict()
                )
            else:
                self.opponent_agent.q_network.load_state_dict(
                    self.agent.q_network.state_dict()
                )
        else:
            # Play against random past snapshot
            snapshot = random.choice(self.snapshots)
            snapshot.load_into(self.opponent_agent)

        return self.opponent_agent

    def record_result(self, ship0_won: bool, ship1_won: bool):
        """Record episode result."""
        if ship0_won and not ship1_won:
            self.wins += 1
        elif ship1_won and not ship0_won:
            self.losses += 1
        else:
            self.draws += 1

    def get_win_rate(self) -> float:
        """Get current win rate."""
        total = self.wins + self.losses + self.draws
        if total == 0:
            return 0.5
        return self.wins / total


# =============================================================================
# Training Loop
# =============================================================================

class Trainer:
    """Main training loop for Spacewar! AI."""

    def __init__(self, config: Config, args: argparse.Namespace):
        self.config = config
        self.args = args

        # Set seeds
        self._set_seeds(config.training.seed)

        # Create environment
        render_mode = "human" if args.render else None
        self.env = SpacewarEnv(
            game_config=config.game,
            reward_config=config.reward,
            render_mode=render_mode,
        )

        # Get dimensions
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n

        # Create agent
        agent_config = config.dqn if args.algorithm == "dqn" else config.ppo
        self.agent = create_agent(
            args.algorithm,
            self.state_dim,
            self.action_dim,
            agent_config,
            config.training.device,
        )

        # Self-play manager
        self.self_play = SelfPlayManager(config, self.agent)

        # LLM integration
        self.llm_guide: Optional[LLMExplorationGuide] = None
        if args.use_llm:
            self._setup_llm()

        # Logging
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        self.best_reward = float("-inf")

        # PPO step accumulation (PPO updates after n_steps, not after each episode)
        self.ppo_steps_collected = 0

        # Checkpointing
        self.checkpoint_dir = Path(config.training.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # TensorBoard
        self.writer = None
        if config.training.use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                log_dir = f"runs/spacewar_{args.algorithm}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                self.writer = SummaryWriter(log_dir)
                logger.info(f"TensorBoard logging to: {log_dir}")
            except ImportError:
                logger.warning("TensorBoard not available")

    def _set_seeds(self, seed: int):
        """Set random seeds for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

    def _setup_llm(self):
        """Setup LLM-guided exploration."""
        # For local LLM mode, no API key needed
        if self.config.llm.use_local:
            logger.info("Initializing local LLM for guided exploration...")
            llm_client = create_llm_client(
                self.config.llm,
                api_manager=None,
                include_local=True,
            )
        else:
            # API-based LLM
            api_manager = APIKeyManager(self.config.llm.keys_dir)

            if not api_manager.has_any_key():
                logger.warning("No LLM API keys found, disabling LLM guidance")
                return

            llm_client = create_llm_client(
                self.config.llm,
                api_manager,
                provider=self.args.llm_provider,
                model=self.args.llm_model,
            )

        if llm_client:
            self.llm_guide = LLMExplorationGuide(
                llm_client,
                exploration_prob=0.1,  # Use LLM for 10% of exploration
            )
            logger.info("LLM-guided exploration enabled")

    def train(self):
        """Main training loop."""
        logger.info(f"Starting training: {self.args.algorithm.upper()}, {self.args.episodes} episodes")
        logger.info(f"Device: {self.config.training.device}")

        for episode in range(1, self.args.episodes + 1):
            # Select opponent
            opponent = self.self_play.select_opponent()

            # Run episode
            if self.args.algorithm == "dqn":
                episode_reward, episode_length, info = self._train_episode_dqn(opponent)
            else:
                episode_reward, episode_length, info = self._train_episode_ppo(opponent)

            # Record stats
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            # Only count real wins (torpedo kills), not opponent suicides
            real_winner = info.get("real_winner")
            self.self_play.record_result(
                real_winner == 0,  # ship0 won by killing ship1
                real_winner == 1,  # ship1 won by killing ship0
            )

            # Logging
            if episode % self.config.training.log_interval == 0:
                self._log_progress(episode)

            # Checkpointing
            if episode % self.config.training.checkpoint_interval == 0:
                self._save_checkpoint(episode)

            # Self-play snapshot
            self.self_play.maybe_snapshot(episode, self.self_play.get_win_rate())

            # Early stopping
            if self._check_early_stop():
                logger.info("Early stopping triggered")
                break

        # Final save
        self._save_checkpoint(self.args.episodes, final=True)
        logger.info("Training complete")

        if self.writer:
            self.writer.close()

    def _train_episode_dqn(
        self, opponent: BaseAgent
    ) -> Tuple[float, int, Dict[str, Any]]:
        """Train one episode with DQN."""
        obs, info = self.env.reset()
        episode_reward = 0.0
        step = 0

        while True:
            # Get action masks
            action_mask = self.env.get_action_mask(0)
            opp_mask = self.env.get_action_mask(1)

            # Select actions
            action = self._select_action_with_exploration(obs, action_mask)
            opp_action = opponent.select_action(
                self._get_opponent_obs(), opp_mask, deterministic=True
            )

            # Step environment
            next_obs, reward, terminated, truncated, info = self.env.step_both(action, opp_action)
            done = terminated or truncated

            # Store transition
            self.agent.store_transition(Transition(
                state=obs,
                action=action,
                reward=reward,
                next_state=next_obs,
                done=done,
                action_mask=action_mask,
            ))

            # Update
            if len(self.agent.replay_buffer) >= self.agent.config.batch_size:
                self.agent.update()

            episode_reward += reward
            obs = next_obs
            step += 1

            if done:
                break

        return episode_reward, step, info

    def _train_episode_ppo(
        self, opponent: BaseAgent
    ) -> Tuple[float, int, Dict[str, Any]]:
        """Train one episode with PPO."""
        obs, info = self.env.reset()
        episode_reward = 0.0
        step = 0

        while True:
            # Get action masks
            action_mask = self.env.get_action_mask(0)
            opp_mask = self.env.get_action_mask(1)

            # Get action with value for storage
            action, log_prob, value = self.agent.get_action_and_value_for_storage(
                obs, action_mask
            )

            # Maybe use LLM exploration instead
            if self.llm_guide and self.llm_guide.should_use_llm():
                state_dict = self._obs_to_llm_state(obs)
                action = self.llm_guide.get_exploration_action(state_dict)

            opp_action = opponent.select_action(
                self._get_opponent_obs(), opp_mask, deterministic=True
            )

            # Step environment
            next_obs, reward, terminated, truncated, info = self.env.step_both(action, opp_action)
            done = terminated or truncated

            # Store step
            self.agent.store_transition(RolloutStep(
                state=obs,
                action=action,
                reward=reward,
                value=value,
                log_prob=log_prob,
                done=done,
                action_mask=action_mask,
            ))

            episode_reward += reward
            obs = next_obs
            step += 1
            self.ppo_steps_collected += 1

            # PPO update when we've collected enough steps
            if self.ppo_steps_collected >= self.config.ppo.n_steps:
                next_value = 0.0 if done else self.agent.get_value(obs)
                self.agent.update(next_value)
                self.ppo_steps_collected = 0

            if done:
                break

        return episode_reward, step, info

    def _select_action_with_exploration(
        self, obs: np.ndarray, action_mask: np.ndarray
    ) -> int:
        """Select action with possible LLM-guided exploration (for DQN)."""
        if self.llm_guide and self.llm_guide.should_use_llm():
            state_dict = self._obs_to_llm_state(obs)
            return self.llm_guide.get_exploration_action(state_dict)

        return self.agent.select_action(obs, action_mask, deterministic=False)

    def _get_opponent_obs(self) -> np.ndarray:
        """Get observation from opponent's perspective."""
        return self.env._get_observation(1)

    def _obs_to_llm_state(self, obs: np.ndarray) -> Dict[str, Any]:
        """Convert observation to LLM state dict."""
        ship = self.env.ships[0]
        opponent = self.env.ships[1]

        return {
            "x": ship.position.x,
            "y": ship.position.y,
            "vx": ship.velocity.x,
            "vy": ship.velocity.y,
            "angle": ship.angle,
            "fuel": ship.fuel,
            "ammo": ship.ammo,
            "can_fire": ship.can_fire(),
            "star_dist": ship.position.distance_to(self.env.star.position),
            "opp_rel_x": opponent.position.x - ship.position.x,
            "opp_rel_y": opponent.position.y - ship.position.y,
            "opp_dist": ship.position.distance_to(opponent.position),
            "opp_angle": opponent.angle,
            "torpedo_count": len([t for t in self.env.torpedoes if t.owner_id != 0]),
        }

    def _log_progress(self, episode: int):
        """Log training progress."""
        avg_reward = np.mean(self.episode_rewards) if self.episode_rewards else 0
        avg_length = np.mean(self.episode_lengths) if self.episode_lengths else 0
        win_rate = self.self_play.get_win_rate()

        logger.info(
            f"Episode {episode} | "
            f"Avg Reward: {avg_reward:.1f} | "
            f"Avg Length: {avg_length:.0f} | "
            f"Win Rate: {win_rate:.1%}"
        )

        if self.writer:
            self.writer.add_scalar("reward/episode", self.episode_rewards[-1], episode)
            self.writer.add_scalar("reward/avg_100", avg_reward, episode)
            self.writer.add_scalar("episode/length", self.episode_lengths[-1], episode)
            self.writer.add_scalar("self_play/win_rate", win_rate, episode)

            if isinstance(self.agent, DQNAgent):
                self.writer.add_scalar("dqn/epsilon", self.agent.epsilon, episode)

    def _save_checkpoint(self, episode: int, final: bool = False):
        """Save model checkpoint with algorithm-specific naming."""
        algo = self.args.algorithm
        path = self.checkpoint_dir / f"{algo}_model_ep{episode}.pt"
        self.agent.save(str(path))

        avg_reward = np.mean(self.episode_rewards) if self.episode_rewards else 0
        if avg_reward > self.best_reward:
            self.best_reward = avg_reward
            best_path = self.checkpoint_dir / f"{algo}_best_model.pt"
            self.agent.save(str(best_path))
            logger.info(f"New best {algo.upper()} model saved (reward: {avg_reward:.1f})")

        if final:
            final_path = self.checkpoint_dir / f"{algo}_final_model.pt"
            self.agent.save(str(final_path))

    def _check_early_stop(self) -> bool:
        """Check if early stopping criteria met."""
        if self.config.training.early_stop_reward is None:
            return False

        if len(self.episode_rewards) < self.config.training.early_stop_window:
            return False

        avg = np.mean(list(self.episode_rewards)[-self.config.training.early_stop_window:])
        return avg >= self.config.training.early_stop_reward


# =============================================================================
# Evaluation
# =============================================================================

def find_best_checkpoint(checkpoint_dir: str, algorithm: Optional[str] = None) -> Optional[str]:
    """Find the best checkpoint in the directory, optionally filtered by algorithm."""
    checkpoint_path = Path(checkpoint_dir)
    if not checkpoint_path.exists():
        return None

    # If algorithm specified, filter to that algorithm's checkpoints
    if algorithm:
        patterns = [
            f"{algorithm}_best_model.pt",
            f"{algorithm}_best*.pt",
            f"{algorithm}_champion_best.pt",
            f"{algorithm}_final*.pt",
            f"{algorithm}_*.pt",
        ]
    else:
        # Look for best models (algorithm-specific first, then generic)
        patterns = ["*_best_model.pt", "best_model.pt", "*_final_model.pt", "final_model.pt"]

    for pattern in patterns:
        matches = list(checkpoint_path.glob(pattern))
        if matches:
            # If multiple, pick most recently modified
            matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            return str(matches[0])

    # Fall back to any .pt file if no algorithm specified, most recent first
    if not algorithm:
        all_checkpoints = list(checkpoint_path.glob("*.pt"))
        if all_checkpoints:
            all_checkpoints.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            return str(all_checkpoints[0])

    return None


def evaluate(args: argparse.Namespace, config: Config):
    """Evaluate trained models - both players use the best model."""

    # Auto-detect checkpoint if not specified or doesn't exist
    checkpoint_path = args.checkpoint
    if not checkpoint_path or not Path(checkpoint_path).exists():
        if checkpoint_path:
            logger.info(f"Checkpoint not found: {checkpoint_path}")
        logger.info("Searching for best checkpoint...")
        checkpoint_path = find_best_checkpoint(args.checkpoint_dir)
        if checkpoint_path is None:
            logger.error(f"No checkpoints found in {args.checkpoint_dir}. Train a model first.")
            return
        logger.info(f"Auto-detected best checkpoint: {checkpoint_path}")

    logger.info(f"Evaluating model: {checkpoint_path}")

    # Create environment
    env = SpacewarEnv(
        game_config=config.game,
        reward_config=config.reward,
        render_mode="human" if args.render else None,
    )

    # Load agent
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # Detect algorithm from checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    is_dqn = "q_network" in checkpoint

    if is_dqn:
        agent_0 = DQNAgent(state_dim, action_dim, config.dqn, config.training.device)
        agent_1 = DQNAgent(state_dim, action_dim, config.dqn, config.training.device)
    else:
        agent_0 = PPOAgent(state_dim, action_dim, config.ppo, config.training.device)
        agent_1 = PPOAgent(state_dim, action_dim, config.ppo, config.training.device)

    # Both agents use the same trained model
    agent_0.load(checkpoint_path)
    agent_1.load(checkpoint_path)

    logger.info("Both players using trained model")

    # Run evaluation episodes
    ship0_wins = 0
    ship1_wins = 0
    draws = 0
    total_rewards_0 = []
    total_rewards_1 = []

    for ep in range(args.eval_episodes):
        obs_0, info = env.reset()
        episode_reward_0 = 0
        episode_reward_1 = 0
        done = False

        while not done:
            # Both players use trained model
            action_mask_0 = env.get_action_mask(0)
            action_mask_1 = env.get_action_mask(1)

            obs_1 = env._get_observation(1)

            action_0 = agent_0.select_action(obs_0, action_mask_0, deterministic=True)
            action_1 = agent_1.select_action(obs_1, action_mask_1, deterministic=True)

            obs_0, reward_0, terminated, truncated, info = env.step_both(action_0, action_1)
            reward_1 = info.get("reward_1", 0)

            episode_reward_0 += reward_0
            episode_reward_1 += reward_1
            done = terminated or truncated

        total_rewards_0.append(episode_reward_0)
        total_rewards_1.append(episode_reward_1)

        ship0_alive = info.get("ship_0_alive", False)
        ship1_alive = info.get("ship_1_alive", False)

        if ship0_alive and not ship1_alive:
            ship0_wins += 1
            result = "P1 wins"
        elif ship1_alive and not ship0_alive:
            ship1_wins += 1
            result = "P2 wins"
        else:
            draws += 1
            result = "Draw"

        logger.info(f"Episode {ep+1}: {result} | P1 reward={episode_reward_0:.1f}, P2 reward={episode_reward_1:.1f}")

    logger.info("=" * 50)
    logger.info(f"Evaluation complete ({args.eval_episodes} episodes)")
    logger.info(f"P1 wins: {ship0_wins} ({ship0_wins/args.eval_episodes:.1%})")
    logger.info(f"P2 wins: {ship1_wins} ({ship1_wins/args.eval_episodes:.1%})")
    logger.info(f"Draws: {draws} ({draws/args.eval_episodes:.1%})")
    logger.info(f"P1 avg reward: {np.mean(total_rewards_0):.1f}")
    logger.info(f"P2 avg reward: {np.mean(total_rewards_1):.1f}")

    env.close()


# =============================================================================
# Human Play Mode
# =============================================================================

def human_play(config: Config, difficulty: str, checkpoint_dir: str):
    """
    Play against AI opponent with keyboard controls.

    Controls:
        Arrow Keys / WASD - Move and rotate
        Space            - Fire torpedo
        R                - Restart game
        ESC / Q          - Quit

    Difficulty levels:
        easy   - Heuristic AI (predictable)
        medium - Trained PPO with some randomness
        hard   - Best trained model
        insane - Best model, aggressive settings
    """
    import pygame

    logger.info(f"Starting human play mode - Difficulty: {difficulty}")
    logger.info("Controls: Arrow keys/WASD to move, SPACE to fire, R to restart, ESC to quit")

    # Find best checkpoint
    checkpoint_path = None
    if difficulty in ["medium", "hard", "insane"]:
        checkpoint_dir_path = Path(checkpoint_dir)
        if checkpoint_dir_path.exists():
            # Look for best model first, then any PPO checkpoint
            candidates = list(checkpoint_dir_path.glob("*best*.pt"))
            if not candidates:
                candidates = list(checkpoint_dir_path.glob("ppo_*.pt"))
            if not candidates:
                candidates = list(checkpoint_dir_path.glob("*.pt"))
            if candidates:
                # Get most recent
                checkpoint_path = max(candidates, key=lambda p: p.stat().st_mtime)
                logger.info(f"Loaded AI from: {checkpoint_path}")
            else:
                logger.warning("No checkpoints found, falling back to heuristic AI")

    # Create AI opponent
    ai_agent = None
    if checkpoint_path and difficulty in ["medium", "hard", "insane"]:
        try:
            ai_agent = PPOAgent(
                state_dim=36,
                action_dim=6,
                config=config.ppo,
                device="cpu",  # CPU for responsiveness
            )
            ai_agent.load(str(checkpoint_path))
            logger.info("AI agent loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}, using heuristic")
            ai_agent = None

    # Create environment
    env = SpacewarEnv(
        game_config=config.game,
        reward_config=config.reward,
        render_mode="human",
    )

    # Difficulty settings
    ai_deterministic = difficulty in ["hard", "insane"]
    ai_reaction_delay = {"easy": 3, "medium": 2, "hard": 1, "insane": 0}[difficulty]

    # Stats tracking
    human_wins = 0
    ai_wins = 0
    draws = 0
    games_played = 0

    running = True
    while running:
        obs, info = env.reset()
        done = False
        step = 0
        frame_counter = 0

        logger.info(f"Game {games_played + 1} started - You are the BLUE ship (left)")

        while not done and running:
            # Render first to process events
            env.render()

            # Handle pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_ESCAPE, pygame.K_q):
                        running = False
                    elif event.key == pygame.K_r:
                        done = True  # Restart

            if not running:
                break

            # Get human action
            human_action = env.get_human_action()

            # Get AI action
            if ai_agent is not None:
                # Get observation from AI's perspective (ship 1)
                ai_obs = env._get_observation(1)
                ai_mask = env._get_action_mask(1)

                # Add reaction delay for easier difficulties
                if frame_counter % (ai_reaction_delay + 1) == 0:
                    ai_action = ai_agent.select_action(
                        ai_obs,
                        ai_mask,
                        deterministic=ai_deterministic,
                    )
                else:
                    ai_action = 0  # No action during delay frames
            else:
                # Heuristic AI
                ai_action = env._heuristic_action(1)

            # Step environment
            obs, reward, terminated, truncated, info = env.step_both(human_action, ai_action)
            done = terminated or truncated
            step += 1
            frame_counter += 1

        if running and done:
            games_played += 1

            # Determine outcome
            real_winner = info.get("real_winner")
            if real_winner == 0:
                human_wins += 1
                result = "YOU WIN"
            elif real_winner == 1:
                ai_wins += 1
                result = "AI WINS"
            else:
                # Check for survival without kill
                if info.get("ship_0_alive") and not info.get("ship_1_alive"):
                    result = "OPPONENT DESTROYED ITSELF"
                elif info.get("ship_1_alive") and not info.get("ship_0_alive"):
                    result = "YOU CRASHED"
                else:
                    draws += 1
                    result = "DRAW"

            logger.info(
                f"Game Over: {result} | "
                f"Score: You {human_wins} - {ai_wins} AI ({draws} draws) | "
                f"Press R to restart, ESC to quit"
            )

            # Wait for input
            waiting = True
            while waiting and running:
                env.render()
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                        waiting = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key in (pygame.K_ESCAPE, pygame.K_q):
                            running = False
                            waiting = False
                        elif event.key == pygame.K_r:
                            waiting = False

    env.close()

    # Final stats
    logger.info("=" * 50)
    logger.info("FINAL SCORE")
    logger.info(f"  You: {human_wins}")
    logger.info(f"  AI:  {ai_wins}")
    logger.info(f"  Draws: {draws}")
    logger.info(f"  Games played: {games_played}")
    if games_played > 0:
        win_rate = human_wins / games_played * 100
        logger.info(f"  Your win rate: {win_rate:.1f}%")
    logger.info("=" * 50)


# =============================================================================
# Demo Mode
# =============================================================================

def demo(config: Config):
    """Run a demo with heuristic agents."""
    logger.info("Running demo with heuristic agents")

    env = SpacewarEnv(
        game_config=config.game,
        reward_config=config.reward,
        render_mode="human",
    )

    for episode in range(5):
        obs, info = env.reset()
        done = False
        step = 0

        while not done:
            # Both ships use heuristic
            action_0 = env._heuristic_action(0)
            action_1 = env._heuristic_action(1)

            obs, reward, terminated, truncated, info = env.step_both(action_0, action_1)
            done = terminated or truncated
            step += 1

        logger.info(f"Episode {episode+1} finished in {step} steps")

    env.close()


# =============================================================================
# Main
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Spacewar! AI - Train RL agents to play the classic 1962 game",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Train with PPO
    python main.py --algorithm ppo --episodes 10000 --render

    # Train with DQN
    python main.py --algorithm dqn --episodes 5000

    # With LLM-guided exploration
    python main.py --algorithm ppo --use-llm --llm-provider groq

    # Evaluate trained model
    python main.py --evaluate --checkpoint checkpoints/best_model.pt

    # Watch demo
    python main.py --demo
        """,
    )

    # Mode selection
    parser.add_argument("--demo", action="store_true", help="Run demo with heuristic agents")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate a trained model")
    parser.add_argument("--tournament", action="store_true", help="Run tournament between trained agents")
    parser.add_argument("--llm-arena", action="store_true", help="Run LLM vs LLM arena")
    parser.add_argument("--champion-train", action="store_true", help="Train against tournament champions")
    parser.add_argument("--human-play", action="store_true", help="Play against AI opponent")
    parser.add_argument(
        "--difficulty",
        choices=["easy", "medium", "hard", "insane"],
        default="medium",
        help="AI difficulty for human play mode (default: medium)",
    )

    # Training options
    parser.add_argument(
        "--algorithm",
        choices=["dqn", "ppo"],
        default="ppo",
        help="RL algorithm to use (default: ppo)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10000,
        help="Number of training episodes (default: 10000)",
    )
    parser.add_argument("--render", action="store_true", help="Render during training")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Checkpoint options
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help="Path to checkpoint for evaluation (auto-detects best if not specified)",
    )
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--eval-episodes", type=int, default=20, help="Episodes for evaluation")

    # LLM options
    parser.add_argument("--use-llm", action="store_true", help="Use LLM-guided exploration")
    parser.add_argument("--llm-provider", type=str, help="LLM provider (openai, anthropic, groq, etc.)")
    parser.add_argument("--llm-model", type=str, help="Specific LLM model to use")
    parser.add_argument("--keys-dir", type=str, default=".", help="Directory containing API key files")
    parser.add_argument("--llm-local", action="store_true", help="Use local LLM via llama-cpp-python")
    parser.add_argument("--llm-local-model", type=str, help="Path to local GGUF model")
    parser.add_argument("--gpu-vram", type=float, help="Override detected GPU VRAM (in GB)")

    # Tournament options
    parser.add_argument(
        "--tournament-mode",
        choices=["round_robin", "swiss", "elimination"],
        default="round_robin",
        help="Tournament format (default: round_robin)",
    )
    parser.add_argument(
        "--matches-per-pairing",
        type=int,
        default=5,
        help="Number of matches per pairing (default: 5)",
    )
    parser.add_argument(
        "--include-local-llm",
        action="store_true",
        help="Include local LLM in arena/tournament",
    )
    parser.add_argument(
        "--local-llm-only",
        action="store_true",
        help="Use ONLY local LLM (ignore cloud API keys)",
    )
    parser.add_argument(
        "--iterative",
        action="store_true",
        help="Use iterative strategy refinement (Eureka-style) - LLM improves strategy based on training feedback",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="Max strategy refinement iterations (default: 10)",
    )
    parser.add_argument(
        "--episodes-per-iteration",
        type=int,
        default=500,
        help="Episodes per iteration before refinement check (default: 500)",
    )

    return parser.parse_args()


# =============================================================================
# Tournament Mode
# =============================================================================

def run_agent_tournament(args: argparse.Namespace, config: Config):
    """Run tournament between trained agents."""
    logger.info("Starting Agent Tournament")

    arena = AgentArena(
        config=config,
        checkpoint_dir=args.checkpoint_dir,
        matches_per_pairing=args.matches_per_pairing,
        render=args.render,
    )

    if len(arena.competitors) < 2:
        logger.error("Need at least 2 trained models for tournament")
        logger.info("Train models first with: python main.py --algorithm ppo --episodes 1000")
        return

    result = arena.run(mode=args.tournament_mode)

    # Save ELO ratings
    ratings_path = Path(args.checkpoint_dir) / "tournament_ratings.json"
    arena.tournament.elo.save(str(ratings_path))
    logger.info(f"Saved ratings to: {ratings_path}")

    # Report champion
    champion = result.get("champion")
    if champion:
        cid, rating = champion
        competitor = arena.tournament.competitors.get(cid)
        logger.info(f"CHAMPION: {competitor.name if competitor else cid} (ELO: {rating.rating:.0f})")


def run_llm_arena(args: argparse.Namespace, config: Config):
    """Run LLM vs LLM arena."""
    logger.info("Starting LLM Arena")

    # Get all available LLM clients
    api_manager = APIKeyManager(config.llm.keys_dir)

    if not api_manager.has_any_key() and not args.include_local_llm:
        logger.error("No LLM API keys found. Add API keys or use --include-local-llm")
        return

    # Create individual clients for each provider
    llm_clients = {}

    for provider in api_manager.get_available_providers():
        api_key = api_manager.get_key(provider)
        if api_key:
            client, model = _create_single_client(config.llm, provider, api_key)
            if client:
                llm_clients[provider] = client
                logger.info(f"Added LLM: {provider}/{model}")

    # Add local LLM if requested
    if args.include_local_llm:
        from llm_integration import create_local_llm_client
        local_client = create_local_llm_client(config.llm)
        if local_client and local_client.llm is not None:
            llm_clients["local"] = local_client
            logger.info("Added local LLM")

    if len(llm_clients) < 1:
        logger.error("Need at least 1 LLM for arena (heuristic baseline is always included)")
        return

    arena = LLMArena(
        config=config,
        llm_clients=llm_clients,
        matches_per_pairing=args.matches_per_pairing,
        render=args.render,
    )

    result = arena.run(mode=args.tournament_mode)

    # Save ratings
    ratings_path = Path(args.checkpoint_dir) / "llm_arena_ratings.json"
    arena.tournament.elo.save(str(ratings_path))
    logger.info(f"Saved ratings to: {ratings_path}")

    # Report champion
    champion = result.get("champion")
    if champion:
        cid, rating = champion
        competitor = arena.tournament.competitors.get(cid)
        logger.info(f"CHAMPION: {competitor.name if competitor else cid} (ELO: {rating.rating:.0f})")


def run_champion_training(args: argparse.Namespace, config: Config):
    """Train against the best opponents from previous tournaments."""
    logger.info("Starting Champion Training")

    # Load previous tournament ratings if available
    elo = ELOSystem()
    ratings_path = Path(args.checkpoint_dir) / "tournament_ratings.json"
    if ratings_path.exists():
        elo.load(str(ratings_path))
        logger.info(f"Loaded previous ratings from {ratings_path}")

    # Create environment
    env = SpacewarEnv(
        game_config=config.game,
        reward_config=config.reward,
        render_mode="human" if args.render else None,
    )

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # Create training agent
    agent_config = config.dqn if args.algorithm == "dqn" else config.ppo
    training_agent = create_agent(
        args.algorithm,
        state_dim,
        action_dim,
        agent_config,
        config.training.device,
    )

    # Load best checkpoint if available (filter by algorithm to avoid mismatches)
    best_checkpoint = find_best_checkpoint(args.checkpoint_dir, algorithm=args.algorithm)
    if best_checkpoint:
        training_agent.load(best_checkpoint)
        logger.info(f"Loaded checkpoint: {best_checkpoint}")

    # Create opponents from checkpoints
    opponents = []

    # Load trained agents
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)  # Ensure dir exists
    if checkpoint_dir.exists():
        for checkpoint_path in checkpoint_dir.glob("*.pt"):
            try:
                checkpoint = torch.load(
                    checkpoint_path, map_location="cpu", weights_only=False
                )
                is_dqn = "q_network" in checkpoint

                if is_dqn:
                    opp_agent = DQNAgent(
                        state_dim, action_dim, config.dqn, config.training.device
                    )
                else:
                    opp_agent = PPOAgent(
                        state_dim, action_dim, config.ppo, config.training.device
                    )

                opp_agent.load(str(checkpoint_path))

                opponent = AgentCompetitor(
                    id=f"agent_{checkpoint_path.stem}",
                    name=checkpoint_path.stem,
                    agent=opp_agent,
                    checkpoint_path=str(checkpoint_path),
                    algorithm="dqn" if is_dqn else "ppo",
                )
                opponents.append(opponent)
            except Exception as e:
                logger.warning(f"Failed to load {checkpoint_path}: {e}")

    # Add LLM opponents if requested
    if args.use_llm or args.local_llm_only:
        if args.local_llm_only:
            # Use ONLY local LLM, skip cloud APIs
            from llm_integration import create_local_llm_client, LLMStrategyGenerator
            from tournament import StrategyCompetitor

            logger.info("Using local LLM only (ignoring cloud APIs)")
            local_client = create_local_llm_client(config.llm)
            if local_client and local_client.llm is not None:
                generator = LLMStrategyGenerator(local_client)
                strategy = generator.generate()
                if strategy:
                    model_name = local_client.model_spec.description if local_client.model_spec else 'local'
                    competitor = StrategyCompetitor(
                        id="strategy_local",
                        name=f"Local/{model_name}",
                        strategy_func=strategy.function,
                        provider="local",
                        model=model_name,
                        code=strategy.code,
                    )
                    opponents.append(competitor)
                    logger.info(f"Generated strategy from local LLM ({len(strategy.code)} chars)")
                else:
                    logger.warning("Failed to generate strategy from local LLM")
            else:
                logger.error("Failed to initialize local LLM")
        else:
            # Use cloud APIs (and optionally local)
            api_manager = APIKeyManager(config.llm.keys_dir)
            llm_competitors = create_llm_competitors(
                config, api_manager, include_local=args.include_local_llm
            )
            opponents.extend(llm_competitors)

    # Always include heuristic
    opponents.append(HeuristicCompetitor(id="heuristic", name="Heuristic"))

    if not opponents:
        logger.error("No opponents available for champion training")
        return

    logger.info(f"Training against {len(opponents)} opponents")

    # Create champion trainer
    trainer = ChampionTrainer(
        config=config,
        training_agent=training_agent,
        opponents=opponents,
        elo_system=elo,
    )

    # Training loop
    episode_rewards = []
    best_reward = float("-inf")

    for episode in range(1, args.episodes + 1):
        reward, info = trainer.train_episode()
        episode_rewards.append(reward)

        # Log progress
        if episode % config.training.log_interval == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            logger.info(
                f"Episode {episode} | "
                f"Avg Reward: {avg_reward:.1f} | "
                f"Last opponent: {info['opponent_name']}"
            )

        # Save checkpoint
        if episode % config.training.checkpoint_interval == 0:
            path = checkpoint_dir / f"{args.algorithm}_champion_ep{episode}.pt"
            training_agent.save(str(path))

            avg_reward = np.mean(episode_rewards[-100:]) if episode_rewards else 0
            if avg_reward > best_reward:
                best_reward = avg_reward
                best_path = checkpoint_dir / f"{args.algorithm}_champion_best.pt"
                training_agent.save(str(best_path))
                logger.info(f"New best champion model (reward: {avg_reward:.1f})")

    # Final save
    final_path = checkpoint_dir / f"{args.algorithm}_champion_final.pt"
    training_agent.save(str(final_path))
    logger.info(f"Champion training complete. Model saved to {final_path}")

    env.close()


def run_iterative_training(args: argparse.Namespace, config: Config):
    """
    Run iterative strategy training (Eureka-style).

    The LLM generates and refines strategies based on training feedback.
    """
    from llm_integration import (
        create_local_llm_client,
        IterativeStrategyTrainer,
        APIKeyManager,
        _create_single_client,
    )
    from tournament import StrategyCompetitor, HeuristicCompetitor

    logger.info("Starting Iterative Strategy Training (Eureka-style)")

    # Create environment
    env = SpacewarEnv(
        game_config=config.game,
        reward_config=config.reward,
        render_mode="human" if args.render else None,
    )

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # Create training agent
    agent_config = config.dqn if args.algorithm == "dqn" else config.ppo
    training_agent = create_agent(
        args.algorithm,
        state_dim,
        action_dim,
        agent_config,
        config.training.device,
    )

    # Load best checkpoint if available
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint = find_best_checkpoint(args.checkpoint_dir, algorithm=args.algorithm)
    if best_checkpoint:
        training_agent.load(best_checkpoint)
        logger.info(f"Loaded checkpoint: {best_checkpoint}")

    # Create LLM client for iterative training
    llm_client = None
    iterative_trainer = None
    strategy = None

    if args.local_llm_only:
        logger.info("Using local LLM for iterative training")
        llm_client = create_local_llm_client(config.llm)
        if llm_client is None or llm_client.llm is None:
            logger.error("Failed to initialize local LLM")
            return
        iterative_trainer = IterativeStrategyTrainer(
            llm_client=llm_client,
            episodes_per_iteration=args.episodes_per_iteration,
            max_iterations=args.iterations,
        )
        strategy = iterative_trainer.initialize()
    else:
        # Try each available provider until one works
        api_manager = APIKeyManager(config.llm.keys_dir)
        providers = api_manager.get_available_providers()

        for provider in providers:
            api_key = api_manager.get_key(provider)
            if not api_key:
                continue

            client, model = _create_single_client(config.llm, provider, api_key)
            if not client:
                continue

            logger.info(f"Trying {provider}/{model} for iterative training...")

            # Create trainer and try to generate initial strategy
            test_trainer = IterativeStrategyTrainer(
                llm_client=client,
                episodes_per_iteration=args.episodes_per_iteration,
                max_iterations=args.iterations,
            )

            # Try to generate - if API fails (billing, etc), try next provider
            test_strategy = test_trainer.generator.generate()
            if test_strategy:
                logger.info(f"Using {provider}/{model} for iterative training")
                llm_client = client
                iterative_trainer = test_trainer
                iterative_trainer.current_strategy = test_strategy
                iterative_trainer.iteration = 1
                strategy = test_strategy
                break
            else:
                logger.warning(f"Provider {provider} failed, trying next...")

    if iterative_trainer is None or strategy is None:
        logger.error("No working LLM available for iterative training.")
        logger.error("All providers failed or no API keys found.")
        logger.error("Use --local-llm-only or check API key billing/validity.")
        return

    # Create strategy competitor that will be updated
    strategy_competitor = StrategyCompetitor(
        id="strategy_iterative",
        name=f"Iterative/{iterative_trainer.provider_name}",
        strategy_func=strategy.function,
        provider=iterative_trainer.provider_name,
        model=iterative_trainer.model,
        code=strategy.code,
    )

    # Always include heuristic baseline
    heuristic = HeuristicCompetitor(id="heuristic", name="Heuristic")

    logger.info(f"Training with iterative strategy refinement")
    logger.info(f"  Episodes per iteration: {args.episodes_per_iteration}")
    logger.info(f"  Max iterations: {args.iterations}")
    logger.info(f"  Total max episodes: {args.episodes}")

    # Training loop with iterative refinement
    episode_rewards = []
    best_reward = float("-inf")
    total_episodes = 0
    ppo_steps_collected = 0  # Track steps for proper PPO batching

    while total_episodes < args.episodes:
        # Pick opponent (alternate between strategy and heuristic)
        if total_episodes % 2 == 0:
            opponent = strategy_competitor
        else:
            opponent = heuristic

        # Run one episode
        obs, info = env.reset()
        done = False
        truncated = False
        episode_reward = 0

        while not (done or truncated):
            # Get agent action
            action_mask = env.get_action_mask()

            # For PPO, we need action, log_prob, and value
            if args.algorithm == "ppo":
                action, log_prob, value = training_agent.get_action_and_value_for_storage(obs, action_mask)
            else:
                action = training_agent.select_action(obs, action_mask)

            # Get opponent action
            opp_action = opponent.get_action(obs, action_mask, env, 1)

            # Step environment
            next_obs, reward, done, truncated, info = env.step_both(action, opp_action)

            # Store transition for learning (different structure for PPO vs DQN)
            if args.algorithm == "ppo":
                step = RolloutStep(
                    state=obs,
                    action=action,
                    reward=reward,
                    value=value,
                    log_prob=log_prob,
                    done=done or truncated,
                    action_mask=action_mask,
                )
                training_agent.store_transition(step)
            else:
                transition = Transition(
                    state=obs,
                    action=action,
                    reward=reward,
                    next_state=next_obs,
                    done=done or truncated,
                    action_mask=action_mask,
                )
                training_agent.store_transition(transition)

            episode_reward += reward
            obs = next_obs

            # PPO: update when we've collected enough steps (not after every episode)
            if args.algorithm == "ppo":
                ppo_steps_collected += 1
                if ppo_steps_collected >= config.ppo.n_steps:
                    next_value = 0.0 if (done or truncated) else training_agent.get_value(obs)
                    training_agent.update(next_value)
                    ppo_steps_collected = 0

        # DQN updates after each transition (handled in store_transition or here)
        if args.algorithm == "dqn":
            training_agent.update()

        episode_rewards.append(episode_reward)
        total_episodes += 1

        # Record for iterative trainer (track wins/losses)
        # Only count real wins (torpedo kills), not opponent suicides
        real_winner = info.get('real_winner')
        won = real_winner == 0
        lost = real_winner == 1
        iterative_trainer.record_episode(episode_reward, won, lost, info)

        # Log progress
        if total_episodes % config.training.log_interval == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            perf = iterative_trainer.current_performance
            logger.info(
                f"Episode {total_episodes} | "
                f"Avg Reward: {avg_reward:.1f} | "
                f"Win Rate: {perf.win_rate:.1%} | "
                f"Iteration: {iterative_trainer.iteration}"
            )

        # Check if we should refine the strategy
        if iterative_trainer.should_refine():
            new_strategy = iterative_trainer.refine_strategy()
            if new_strategy:
                # Update the competitor with new strategy
                strategy_competitor.strategy_func = new_strategy.function
                strategy_competitor.code = new_strategy.code
                logger.info(f"Strategy updated to iteration {iterative_trainer.iteration}")

        # Save checkpoint
        if total_episodes % config.training.checkpoint_interval == 0:
            path = checkpoint_dir / f"{args.algorithm}_iterative_ep{total_episodes}.pt"
            training_agent.save(str(path))

            avg_reward = np.mean(episode_rewards[-100:]) if episode_rewards else 0
            if avg_reward > best_reward:
                best_reward = avg_reward
                best_path = checkpoint_dir / f"{args.algorithm}_iterative_best.pt"
                training_agent.save(str(best_path))
                logger.info(f"New best model (reward: {avg_reward:.1f})")

    # Save final model and strategy
    final_path = checkpoint_dir / f"{args.algorithm}_iterative_final.pt"
    training_agent.save(str(final_path))

    # Save best strategy code
    best_strategy = iterative_trainer.get_best_strategy()
    if best_strategy:
        strategy_path = checkpoint_dir / f"best_strategy_{iterative_trainer.provider_name}.py"
        with open(strategy_path, 'w') as f:
            f.write(f"# Best strategy from {iterative_trainer.provider_name}\n")
            f.write(f"# Win rate: {iterative_trainer.best_win_rate:.1%}\n")
            f.write(f"# Iterations: {iterative_trainer.iteration}\n\n")
            f.write(best_strategy.code)
        logger.info(f"Best strategy saved to {strategy_path}")

    # Print summary
    summary = iterative_trainer.get_summary()
    logger.info(f"Iterative training complete:")
    logger.info(f"  Total iterations: {summary['total_iterations']}")
    logger.info(f"  Best win rate: {summary['best_win_rate']:.1%}")
    logger.info(f"  Model saved to {final_path}")

    env.close()


def main():
    """Main entry point."""
    args = parse_args()

    # Load config
    config = get_default_config()
    config.training.seed = args.seed
    config.training.checkpoint_dir = args.checkpoint_dir
    config.llm.keys_dir = args.keys_dir

    if args.llm_local:
        config.llm.use_local = True
        config.llm.local_model = args.llm_local_model

    # Print hardware info
    hw = detect_hardware(vram_override=args.gpu_vram)
    logger.info(f"Hardware: {hw.cpu_cores} CPU cores, {hw.ram_gb:.1f}GB RAM")
    if hw.gpu_type != "cpu":
        vram_str = f"{hw.gpu_vram_gb:.1f}GB VRAM" if hw.gpu_vram_gb else "unknown VRAM"
        logger.info(f"GPU: {hw.gpu_name} ({vram_str}, {hw.gpu_backend})")

    # Run selected mode
    if args.demo:
        demo(config)
    elif args.human_play:
        human_play(config, args.difficulty, args.checkpoint_dir)
    elif args.evaluate:
        evaluate(args, config)
    elif args.tournament:
        run_agent_tournament(args, config)
    elif args.llm_arena:
        run_llm_arena(args, config)
    elif args.iterative:
        run_iterative_training(args, config)
    elif args.champion_train:
        run_champion_training(args, config)
    else:
        trainer = Trainer(config, args)
        trainer.train()


if __name__ == "__main__":
    main()
