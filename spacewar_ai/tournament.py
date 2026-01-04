"""
Tournament System for Spacewar! AI

Supports:
- Cross-algorithm competition (DQN vs PPO)
- LLM Arena (different LLMs compete)
- ELO rating system
- Champion training (best of the best)
"""

import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, List, Any, Tuple, Callable
from collections import defaultdict
import json

import numpy as np
import torch

from config import Config
from environment import SpacewarEnv
from agents import BaseAgent, DQNAgent, PPOAgent, create_agent

logger = logging.getLogger(__name__)


# =============================================================================
# ELO Rating System
# =============================================================================

@dataclass
class ELORating:
    """ELO rating for a competitor."""
    rating: float = 1500.0
    games_played: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0

    @property
    def win_rate(self) -> float:
        if self.games_played == 0:
            return 0.5
        return self.wins / self.games_played

    def expected_score(self, opponent_rating: float) -> float:
        """Calculate expected score against opponent."""
        return 1.0 / (1.0 + 10 ** ((opponent_rating - self.rating) / 400))

    def update(self, actual_score: float, expected_score: float, k: float = 32.0):
        """Update rating based on match result."""
        self.rating += k * (actual_score - expected_score)
        self.games_played += 1


class ELOSystem:
    """Manages ELO ratings for all competitors."""

    def __init__(self, k_factor: float = 32.0):
        self.k_factor = k_factor
        self.ratings: Dict[str, ELORating] = {}

    def get_rating(self, competitor_id: str) -> ELORating:
        """Get or create rating for competitor."""
        if competitor_id not in self.ratings:
            self.ratings[competitor_id] = ELORating()
        return self.ratings[competitor_id]

    def record_match(
        self,
        player1_id: str,
        player2_id: str,
        player1_won: bool,
        player2_won: bool,
    ):
        """Record match result and update ratings."""
        r1 = self.get_rating(player1_id)
        r2 = self.get_rating(player2_id)

        # Calculate expected scores
        e1 = r1.expected_score(r2.rating)
        e2 = r2.expected_score(r1.rating)

        # Determine actual scores
        if player1_won and not player2_won:
            s1, s2 = 1.0, 0.0
            r1.wins += 1
            r2.losses += 1
        elif player2_won and not player1_won:
            s1, s2 = 0.0, 1.0
            r1.losses += 1
            r2.wins += 1
        else:
            s1, s2 = 0.5, 0.5
            r1.draws += 1
            r2.draws += 1

        # Update ratings
        r1.update(s1, e1, self.k_factor)
        r2.update(s2, e2, self.k_factor)

    def get_rankings(self) -> List[Tuple[str, ELORating]]:
        """Get competitors sorted by rating."""
        return sorted(
            self.ratings.items(),
            key=lambda x: x[1].rating,
            reverse=True
        )

    def get_champion(self) -> Optional[Tuple[str, ELORating]]:
        """Get the highest rated competitor."""
        rankings = self.get_rankings()
        return rankings[0] if rankings else None

    def save(self, path: str):
        """Save ratings to file."""
        data = {
            cid: {
                "rating": r.rating,
                "games_played": r.games_played,
                "wins": r.wins,
                "losses": r.losses,
                "draws": r.draws,
            }
            for cid, r in self.ratings.items()
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, path: str):
        """Load ratings from file."""
        with open(path, "r") as f:
            data = json.load(f)

        for cid, rdata in data.items():
            self.ratings[cid] = ELORating(
                rating=rdata["rating"],
                games_played=rdata["games_played"],
                wins=rdata["wins"],
                losses=rdata["losses"],
                draws=rdata["draws"],
            )


# =============================================================================
# Competitor Types
# =============================================================================

@dataclass
class Competitor:
    """Base class for tournament competitors."""
    id: str
    name: str
    type: str = ""  # "agent", "llm", "heuristic" - set by subclass

    def get_action(
        self,
        obs: np.ndarray,
        action_mask: np.ndarray,
        env: SpacewarEnv,
        player_id: int,
    ) -> int:
        """Get action for current state."""
        raise NotImplementedError


@dataclass
class AgentCompetitor(Competitor):
    """Competitor backed by a trained RL agent."""
    agent: BaseAgent = None
    checkpoint_path: str = ""
    algorithm: str = ""

    def __post_init__(self):
        self.type = "agent"

    def get_action(
        self,
        obs: np.ndarray,
        action_mask: np.ndarray,
        env: SpacewarEnv,
        player_id: int,
    ) -> int:
        return self.agent.select_action(obs, action_mask, deterministic=True)


@dataclass
class StrategyCompetitor(Competitor):
    """
    Competitor using a generated strategy function.

    The strategy is generated ONCE by an LLM and executed locally.
    No API calls are made during gameplay - just fast local execution.
    """
    strategy_func: Callable = None
    provider: str = ""
    model: str = ""
    code: str = ""
    _error_count: int = field(default=0, repr=False)

    def __post_init__(self):
        self.type = "strategy"

    def get_action(
        self,
        obs: np.ndarray,
        action_mask: np.ndarray,
        env: SpacewarEnv,
        player_id: int,
    ) -> int:
        # Convert observation to state dict for strategy
        ship = env.ships[player_id]
        opponent = env.ships[1 - player_id]

        state_dict = {
            "x": ship.position.x,
            "y": ship.position.y,
            "vx": ship.velocity.x,
            "vy": ship.velocity.y,
            "angle": ship.angle,
            "fuel": ship.fuel,
            "ammo": ship.ammo,
            "can_fire": ship.can_fire(),
            "star_dist": ship.position.distance_to(env.star.position),
            "opp_rel_x": opponent.position.x - ship.position.x,
            "opp_rel_y": opponent.position.y - ship.position.y,
            "opp_dist": ship.position.distance_to(opponent.position),
            "opp_angle": opponent.angle,
            "torpedo_count": len([t for t in env.torpedoes if t.owner_id != player_id]),
        }

        try:
            action = int(self.strategy_func(state_dict))
            # Validate action
            if 0 <= action <= 5:
                return action
            return 0
        except Exception as e:
            self._error_count += 1
            if self._error_count <= 3:
                logger.debug(f"Strategy {self.name} execution error: {e}")
            # Fallback to simple heuristic on error
            if state_dict['star_dist'] < 120:
                return 3  # Thrust away from star
            return 0


# Legacy alias for backward compatibility
LLMCompetitor = StrategyCompetitor


@dataclass
class HeuristicCompetitor(Competitor):
    """Competitor using built-in heuristic."""

    def __post_init__(self):
        self.type = "heuristic"

    def get_action(
        self,
        obs: np.ndarray,
        action_mask: np.ndarray,
        env: SpacewarEnv,
        player_id: int,
    ) -> int:
        return env._heuristic_action(player_id)


# =============================================================================
# Tournament Manager
# =============================================================================

class TournamentManager:
    """
    Manages tournaments between multiple competitors.

    Supports:
    - Round-robin tournaments (everyone plays everyone)
    - Swiss-system tournaments (similar ratings play each other)
    - Elimination brackets
    """

    def __init__(
        self,
        config: Config,
        competitors: List[Competitor],
        matches_per_pairing: int = 5,
        render: bool = False,
    ):
        self.config = config
        self.competitors = {c.id: c for c in competitors}
        self.matches_per_pairing = matches_per_pairing
        self.render = render

        self.elo = ELOSystem()
        self.match_history: List[Dict[str, Any]] = []

        # Create environment
        self.env = SpacewarEnv(
            game_config=config.game,
            reward_config=config.reward,
            render_mode="human" if render else None,
        )

    def run_match(
        self,
        player1: Competitor,
        player2: Competitor,
    ) -> Dict[str, Any]:
        """Run a single match between two competitors."""
        obs, info = self.env.reset()
        done = False

        total_reward_1 = 0.0
        total_reward_2 = 0.0
        steps = 0

        while not done:
            # Get observations for each player
            obs_1 = self.env._get_observation(0)
            obs_2 = self.env._get_observation(1)

            mask_1 = self.env.get_action_mask(0)
            mask_2 = self.env.get_action_mask(1)

            # Get actions
            action_1 = player1.get_action(obs_1, mask_1, self.env, 0)
            action_2 = player2.get_action(obs_2, mask_2, self.env, 1)

            # Step environment
            obs, reward_1, terminated, truncated, info = self.env.step_both(
                action_1, action_2
            )
            reward_2 = info.get("reward_1", 0)

            total_reward_1 += reward_1
            total_reward_2 += reward_2
            steps += 1
            done = terminated or truncated

        # Determine winner
        p1_alive = info.get("ship_0_alive", False)
        p2_alive = info.get("ship_1_alive", False)

        result = {
            "player1_id": player1.id,
            "player2_id": player2.id,
            "player1_won": p1_alive and not p2_alive,
            "player2_won": p2_alive and not p1_alive,
            "draw": (p1_alive == p2_alive),
            "player1_reward": total_reward_1,
            "player2_reward": total_reward_2,
            "steps": steps,
        }

        return result

    def run_pairing(
        self,
        player1: Competitor,
        player2: Competitor,
    ) -> Dict[str, Any]:
        """Run multiple matches between a pairing."""
        results = []

        for match_num in range(self.matches_per_pairing):
            # Alternate who plays as player 0
            if match_num % 2 == 0:
                result = self.run_match(player1, player2)
            else:
                result = self.run_match(player2, player1)
                # Swap result perspectives
                result = {
                    "player1_id": player1.id,
                    "player2_id": player2.id,
                    "player1_won": result["player2_won"],
                    "player2_won": result["player1_won"],
                    "draw": result["draw"],
                    "player1_reward": result["player2_reward"],
                    "player2_reward": result["player1_reward"],
                    "steps": result["steps"],
                }

            results.append(result)

            # Update ELO after each match
            self.elo.record_match(
                player1.id,
                player2.id,
                result["player1_won"],
                result["player2_won"],
            )

        # Aggregate results
        p1_wins = sum(1 for r in results if r["player1_won"])
        p2_wins = sum(1 for r in results if r["player2_won"])
        draws = sum(1 for r in results if r["draw"])

        aggregate = {
            "player1_id": player1.id,
            "player2_id": player2.id,
            "player1_wins": p1_wins,
            "player2_wins": p2_wins,
            "draws": draws,
            "player1_avg_reward": np.mean([r["player1_reward"] for r in results]),
            "player2_avg_reward": np.mean([r["player2_reward"] for r in results]),
            "avg_steps": np.mean([r["steps"] for r in results]),
        }

        self.match_history.append(aggregate)
        return aggregate

    def run_round_robin(self) -> List[Dict[str, Any]]:
        """Run round-robin tournament (everyone plays everyone)."""
        logger.info(f"Starting round-robin tournament with {len(self.competitors)} competitors")

        competitor_list = list(self.competitors.values())
        all_results = []

        total_pairings = len(competitor_list) * (len(competitor_list) - 1) // 2
        pairing_num = 0

        for i, p1 in enumerate(competitor_list):
            for p2 in competitor_list[i+1:]:
                pairing_num += 1
                logger.info(f"Pairing {pairing_num}/{total_pairings}: {p1.name} vs {p2.name}")

                result = self.run_pairing(p1, p2)
                all_results.append(result)

                # Log pairing result
                logger.info(
                    f"  Result: {p1.name} {result['player1_wins']}-{result['player2_wins']}-{result['draws']} {p2.name}"
                )

        return all_results

    def run_swiss(self, rounds: int = 5) -> List[Dict[str, Any]]:
        """
        Run Swiss-system tournament.
        Players with similar ratings are paired.
        """
        logger.info(f"Starting Swiss tournament with {len(self.competitors)} competitors, {rounds} rounds")

        all_results = []
        paired_history: set = set()  # Track who has played whom

        for round_num in range(1, rounds + 1):
            logger.info(f"=== Round {round_num}/{rounds} ===")

            # Sort by rating
            rankings = self.elo.get_rankings()
            competitor_ids = [cid for cid, _ in rankings]

            # Add any unrated competitors
            for cid in self.competitors:
                if cid not in competitor_ids:
                    competitor_ids.append(cid)

            # Pair adjacent players (avoiding repeat pairings if possible)
            round_pairings = []
            used = set()

            for i, cid1 in enumerate(competitor_ids):
                if cid1 in used:
                    continue

                # Find best opponent (closest rating, not paired before)
                for cid2 in competitor_ids[i+1:]:
                    if cid2 in used:
                        continue

                    pair_key = tuple(sorted([cid1, cid2]))
                    if pair_key not in paired_history or round_num > len(self.competitors) // 2:
                        round_pairings.append((cid1, cid2))
                        paired_history.add(pair_key)
                        used.add(cid1)
                        used.add(cid2)
                        break

            # Run pairings
            for cid1, cid2 in round_pairings:
                p1 = self.competitors[cid1]
                p2 = self.competitors[cid2]

                logger.info(f"  {p1.name} vs {p2.name}")
                result = self.run_pairing(p1, p2)
                all_results.append(result)

                logger.info(
                    f"    Result: {result['player1_wins']}-{result['player2_wins']}-{result['draws']}"
                )

        return all_results

    def run_elimination(self) -> Tuple[Competitor, List[Dict[str, Any]]]:
        """
        Run single-elimination bracket.
        Returns the champion.
        """
        logger.info(f"Starting elimination tournament with {len(self.competitors)} competitors")

        all_results = []
        remaining = list(self.competitors.values())
        random.shuffle(remaining)

        round_num = 1
        while len(remaining) > 1:
            logger.info(f"=== Round {round_num} ({len(remaining)} competitors) ===")

            next_round = []

            for i in range(0, len(remaining) - 1, 2):
                p1 = remaining[i]
                p2 = remaining[i + 1]

                logger.info(f"  {p1.name} vs {p2.name}")
                result = self.run_pairing(p1, p2)
                all_results.append(result)

                # Winner advances
                if result["player1_wins"] > result["player2_wins"]:
                    winner = p1
                    logger.info(f"    Winner: {p1.name}")
                elif result["player2_wins"] > result["player1_wins"]:
                    winner = p2
                    logger.info(f"    Winner: {p2.name}")
                else:
                    # Tiebreaker: higher ELO
                    r1 = self.elo.get_rating(p1.id).rating
                    r2 = self.elo.get_rating(p2.id).rating
                    winner = p1 if r1 >= r2 else p2
                    logger.info(f"    Winner (tiebreak): {winner.name}")

                next_round.append(winner)

            # Handle odd competitor (bye to next round)
            if len(remaining) % 2 == 1:
                bye = remaining[-1]
                next_round.append(bye)
                logger.info(f"  {bye.name} gets a bye")

            remaining = next_round
            round_num += 1

        champion = remaining[0]
        logger.info(f"Champion: {champion.name}")

        return champion, all_results

    def get_final_rankings(self) -> List[Tuple[str, ELORating]]:
        """Get final rankings after tournament."""
        return self.elo.get_rankings()

    def print_standings(self):
        """Print current standings."""
        rankings = self.get_final_rankings()

        logger.info("=" * 60)
        logger.info("STANDINGS")
        logger.info("=" * 60)
        logger.info(f"{'Rank':<5} {'Name':<25} {'ELO':<8} {'W-L-D':<12} {'Win%':<8}")
        logger.info("-" * 60)

        for rank, (cid, rating) in enumerate(rankings, 1):
            competitor = self.competitors.get(cid)
            name = competitor.name if competitor else cid
            record = f"{rating.wins}-{rating.losses}-{rating.draws}"

            logger.info(
                f"{rank:<5} {name:<25} {rating.rating:<8.0f} {record:<12} {rating.win_rate:<8.1%}"
            )

        logger.info("=" * 60)


# =============================================================================
# LLM Arena
# =============================================================================

class LLMArena:
    """
    Arena for LLM strategy competition.

    Generates strategies from all LLMs ONCE, then has them compete.
    No API calls during gameplay - just fast local strategy execution.
    """

    def __init__(
        self,
        config: Config,
        llm_clients: Dict[str, Any],  # provider -> client
        matches_per_pairing: int = 5,
        render: bool = False,
    ):
        self.config = config
        self.llm_clients = llm_clients

        # Import strategy generator
        from llm_integration import LLMStrategyGenerator

        # Generate strategies from each LLM client ONCE
        logger.info("Generating strategies from LLMs (one-time API calls)...")
        competitors = []

        for provider, client in llm_clients.items():
            try:
                generator = LLMStrategyGenerator(client)
                strategy = generator.generate()

                if strategy:
                    competitor = StrategyCompetitor(
                        id=f"strategy_{provider}",
                        name=f"{provider}/{strategy.model}",
                        strategy_func=strategy.function,
                        provider=provider,
                        model=strategy.model,
                        code=strategy.code,
                    )
                    competitors.append(competitor)
                    logger.info(f"Generated strategy from {provider} ({len(strategy.code)} chars)")
                else:
                    # Use fallback if generation fails
                    fallback = LLMStrategyGenerator.create_fallback_strategy()
                    competitor = StrategyCompetitor(
                        id=f"strategy_{provider}",
                        name=f"{provider}/fallback",
                        strategy_func=fallback.function,
                        provider=provider,
                        model="fallback",
                        code=fallback.code,
                    )
                    competitors.append(competitor)
                    logger.warning(f"Using fallback strategy for {provider}")

            except Exception as e:
                logger.error(f"Failed to generate strategy for {provider}: {e}")

        # Add heuristic as baseline
        competitors.append(HeuristicCompetitor(
            id="heuristic",
            name="Heuristic (baseline)",
        ))

        logger.info(f"Created {len(competitors)} competitors (no more API calls needed)")

        self.tournament = TournamentManager(
            config=config,
            competitors=competitors,
            matches_per_pairing=matches_per_pairing,
            render=render,
        )

    def run(self, mode: str = "round_robin") -> Dict[str, Any]:
        """Run the LLM arena competition."""
        logger.info(f"Starting LLM Arena with {len(self.llm_clients)} strategies")

        if mode == "round_robin":
            results = self.tournament.run_round_robin()
        elif mode == "swiss":
            results = self.tournament.run_swiss()
        elif mode == "elimination":
            champion, results = self.tournament.run_elimination()
        else:
            raise ValueError(f"Unknown mode: {mode}")

        self.tournament.print_standings()

        return {
            "mode": mode,
            "results": results,
            "rankings": self.tournament.get_final_rankings(),
            "champion": self.tournament.elo.get_champion(),
        }


# =============================================================================
# Agent Arena (Cross-Algorithm Competition)
# =============================================================================

class AgentArena:
    """
    Arena for trained agents to compete.

    Loads checkpoints from different algorithms and has them compete.
    """

    def __init__(
        self,
        config: Config,
        checkpoint_dir: str = "checkpoints",
        matches_per_pairing: int = 10,
        render: bool = False,
    ):
        self.config = config
        self.checkpoint_dir = Path(checkpoint_dir)
        self.matches_per_pairing = matches_per_pairing
        self.render = render

        # Load all available checkpoints
        self.competitors = self._load_checkpoints()

        # Add heuristic baseline
        self.competitors.append(HeuristicCompetitor(
            id="heuristic",
            name="Heuristic (baseline)",
        ))

        self.tournament = TournamentManager(
            config=config,
            competitors=self.competitors,
            matches_per_pairing=matches_per_pairing,
            render=render,
        )

    def _load_checkpoints(self) -> List[AgentCompetitor]:
        """Load all available checkpoints as competitors."""
        competitors = []

        if not self.checkpoint_dir.exists():
            logger.warning(f"Checkpoint directory not found: {self.checkpoint_dir}")
            return competitors

        # Create temporary env for dimensions
        temp_env = SpacewarEnv(
            game_config=self.config.game,
            reward_config=self.config.reward,
        )
        state_dim = temp_env.observation_space.shape[0]
        action_dim = temp_env.action_space.n
        temp_env.close()

        # Find best checkpoints for each algorithm
        for pattern in ["*_best_model.pt", "*_final_model.pt"]:
            for checkpoint_path in self.checkpoint_dir.glob(pattern):
                try:
                    # Detect algorithm from filename
                    filename = checkpoint_path.stem
                    if filename.startswith("dqn"):
                        algorithm = "dqn"
                        agent_config = self.config.dqn
                        agent = DQNAgent(
                            state_dim, action_dim, agent_config,
                            self.config.training.device
                        )
                    elif filename.startswith("ppo"):
                        algorithm = "ppo"
                        agent_config = self.config.ppo
                        agent = PPOAgent(
                            state_dim, action_dim, agent_config,
                            self.config.training.device
                        )
                    else:
                        # Try to detect from checkpoint contents
                        checkpoint = torch.load(
                            checkpoint_path, map_location="cpu", weights_only=False
                        )
                        if "q_network" in checkpoint:
                            algorithm = "dqn"
                            agent = DQNAgent(
                                state_dim, action_dim, self.config.dqn,
                                self.config.training.device
                            )
                        else:
                            algorithm = "ppo"
                            agent = PPOAgent(
                                state_dim, action_dim, self.config.ppo,
                                self.config.training.device
                            )

                    agent.load(str(checkpoint_path))

                    competitor = AgentCompetitor(
                        id=f"agent_{filename}",
                        name=f"{algorithm.upper()}: {filename}",
                        agent=agent,
                        checkpoint_path=str(checkpoint_path),
                        algorithm=algorithm,
                    )
                    competitors.append(competitor)
                    logger.info(f"Loaded checkpoint: {checkpoint_path.name}")

                except Exception as e:
                    logger.warning(f"Failed to load {checkpoint_path}: {e}")

        return competitors

    def run(self, mode: str = "round_robin") -> Dict[str, Any]:
        """Run the agent arena competition."""
        logger.info(f"Starting Agent Arena with {len(self.competitors)} competitors")

        if len(self.competitors) < 2:
            logger.error("Need at least 2 competitors for arena")
            return {"error": "Insufficient competitors"}

        if mode == "round_robin":
            results = self.tournament.run_round_robin()
        elif mode == "swiss":
            results = self.tournament.run_swiss()
        elif mode == "elimination":
            champion, results = self.tournament.run_elimination()
        else:
            raise ValueError(f"Unknown mode: {mode}")

        self.tournament.print_standings()

        return {
            "mode": mode,
            "results": results,
            "rankings": self.tournament.get_final_rankings(),
            "champion": self.tournament.elo.get_champion(),
        }


# =============================================================================
# Champion Training
# =============================================================================

class ChampionTrainer:
    """
    Trains agents against the best available opponents.

    Uses tournament results to identify the strongest competitors
    and focuses training against them.
    """

    def __init__(
        self,
        config: Config,
        training_agent: BaseAgent,
        opponents: List[Competitor],
        elo_system: ELOSystem,
    ):
        self.config = config
        self.agent = training_agent
        self.opponents = {c.id: c for c in opponents}
        self.elo = elo_system

        self.env = SpacewarEnv(
            game_config=config.game,
            reward_config=config.reward,
        )

    def select_opponent(self) -> Competitor:
        """
        Select opponent weighted by strength.
        Stronger opponents are more likely to be selected.
        """
        rankings = self.elo.get_rankings()

        if not rankings:
            # No ratings yet, pick random
            return random.choice(list(self.opponents.values()))

        # Weight by rating (higher rating = more likely)
        competitor_ids = [cid for cid, _ in rankings if cid in self.opponents]
        if not competitor_ids:
            return random.choice(list(self.opponents.values()))

        ratings = [self.elo.get_rating(cid).rating for cid in competitor_ids]
        min_rating = min(ratings)
        weights = [r - min_rating + 100 for r in ratings]  # +100 to avoid zero weights
        total = sum(weights)
        probs = [w / total for w in weights]

        selected_id = random.choices(competitor_ids, weights=probs, k=1)[0]
        return self.opponents[selected_id]

    def train_episode(self) -> Tuple[float, Dict[str, Any]]:
        """Train one episode against a strong opponent."""
        opponent = self.select_opponent()

        obs, info = self.env.reset()
        episode_reward = 0.0
        done = False

        while not done:
            # Get action masks
            mask_0 = self.env.get_action_mask(0)
            mask_1 = self.env.get_action_mask(1)

            # Training agent's action
            action = self.agent.select_action(obs, mask_0, deterministic=False)

            # Opponent's action
            opp_obs = self.env._get_observation(1)
            opp_action = opponent.get_action(opp_obs, mask_1, self.env, 1)

            # Step
            next_obs, reward, terminated, truncated, info = self.env.step_both(
                action, opp_action
            )
            done = terminated or truncated

            episode_reward += reward
            obs = next_obs

        return episode_reward, {
            "opponent_id": opponent.id,
            "opponent_name": opponent.name,
            **info,
        }


# =============================================================================
# Utility Functions
# =============================================================================

def create_llm_competitors(
    config: Config,
    api_manager: Any,
    include_local: bool = False,
) -> List[StrategyCompetitor]:
    """
    Create strategy competitors from all available LLM providers.

    This generates strategy code from each LLM ONCE upfront.
    No API calls are made during gameplay - just fast local execution.
    """
    from llm_integration import (
        _create_single_client,
        LLMStrategyGenerator,
    )

    competitors = []

    logger.info("Generating strategies from LLM providers (one-time API calls)...")

    for provider in api_manager.get_available_providers():
        api_key = api_manager.get_key(provider)
        if api_key:
            try:
                client, model = _create_single_client(config.llm, provider, api_key)
                if client:
                    # Generate strategy ONCE
                    generator = LLMStrategyGenerator(client)
                    strategy = generator.generate()

                    if strategy:
                        competitor = StrategyCompetitor(
                            id=f"strategy_{provider}",
                            name=f"{provider}/{strategy.model}",
                            strategy_func=strategy.function,
                            provider=provider,
                            model=strategy.model,
                            code=strategy.code,
                        )
                        competitors.append(competitor)
                        logger.info(f"Generated strategy from {provider} ({len(strategy.code)} chars)")
                    else:
                        # Use fallback strategy
                        fallback = LLMStrategyGenerator.create_fallback_strategy()
                        competitor = StrategyCompetitor(
                            id=f"strategy_{provider}",
                            name=f"{provider}/fallback",
                            strategy_func=fallback.function,
                            provider=provider,
                            model="fallback",
                            code=fallback.code,
                        )
                        competitors.append(competitor)
                        logger.warning(f"Using fallback strategy for {provider}")

            except Exception as e:
                logger.error(f"Failed to generate strategy for {provider}: {e}")

    if include_local:
        from llm_integration import create_local_llm_client, LLMStrategyGenerator

        try:
            local_client = create_local_llm_client(config.llm)
            if local_client and local_client.llm is not None:
                # Generate strategy from local LLM
                generator = LLMStrategyGenerator(local_client)
                strategy = generator.generate()

                if strategy:
                    model_name = local_client.model_spec.description if local_client.model_spec else 'custom'
                    competitor = StrategyCompetitor(
                        id="strategy_local",
                        name=f"Local/{model_name}",
                        strategy_func=strategy.function,
                        provider="local",
                        model=model_name,
                        code=strategy.code,
                    )
                    competitors.append(competitor)
                    logger.info(f"Generated strategy from local LLM ({len(strategy.code)} chars)")
        except Exception as e:
            logger.error(f"Failed to create local LLM competitor: {e}")

    logger.info(f"Created {len(competitors)} strategy competitors (no more API calls needed)")

    return competitors
