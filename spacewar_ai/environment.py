"""
Spacewar! Game Environment

Implements the game physics, objects, and Gymnasium environment.
Faithful recreation of the 1962 classic with modern RL interface.
"""

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any
import numpy as np
import gymnasium as gym
from gymnasium import spaces

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False

from config import GameConfig, RewardConfig


# =============================================================================
# Physics Primitives
# =============================================================================

@dataclass
class Vector2D:
    """2D vector with physics operations."""

    x: float = 0.0
    y: float = 0.0

    def __add__(self, other: "Vector2D") -> "Vector2D":
        return Vector2D(self.x + other.x, self.y + other.y)

    def __sub__(self, other: "Vector2D") -> "Vector2D":
        return Vector2D(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar: float) -> "Vector2D":
        return Vector2D(self.x * scalar, self.y * scalar)

    def __rmul__(self, scalar: float) -> "Vector2D":
        return self.__mul__(scalar)

    def __truediv__(self, scalar: float) -> "Vector2D":
        return Vector2D(self.x / scalar, self.y / scalar)

    def magnitude(self) -> float:
        return math.sqrt(self.x * self.x + self.y * self.y)

    def normalized(self) -> "Vector2D":
        mag = self.magnitude()
        if mag < 1e-8:
            return Vector2D(0.0, 0.0)
        return self / mag

    def dot(self, other: "Vector2D") -> float:
        return self.x * other.x + self.y * other.y

    def distance_to(self, other: "Vector2D") -> float:
        return (self - other).magnitude()

    def angle_to(self, other: "Vector2D") -> float:
        """Angle from self to other in degrees."""
        diff = other - self
        return math.degrees(math.atan2(diff.y, diff.x))

    def as_tuple(self) -> Tuple[float, float]:
        return (self.x, self.y)

    def as_array(self) -> np.ndarray:
        return np.array([self.x, self.y], dtype=np.float32)

    @staticmethod
    def from_angle(angle_degrees: float, magnitude: float = 1.0) -> "Vector2D":
        """Create vector from angle (degrees) and magnitude."""
        rad = math.radians(angle_degrees)
        return Vector2D(math.cos(rad) * magnitude, math.sin(rad) * magnitude)


# =============================================================================
# Game Objects
# =============================================================================

@dataclass
class Star:
    """Central star with gravitational pull."""

    position: Vector2D
    mass: float
    radius: float

    def gravitational_force(self, obj_position: Vector2D, obj_mass: float, g_constant: float) -> Vector2D:
        """Calculate gravitational force on an object."""
        direction = self.position - obj_position
        distance = direction.magnitude()

        # Prevent division by zero and extreme forces near star
        distance = max(distance, self.radius * 1.5)

        # F = G * m1 * m2 / r^2
        force_magnitude = g_constant * self.mass * obj_mass / (distance * distance)

        return direction.normalized() * force_magnitude


class DeathCause:
    """How a ship died."""
    ALIVE = "alive"
    TORPEDO = "torpedo"      # Killed by opponent's torpedo
    STAR = "star"            # Flew into the star (suicide)
    COLLISION = "collision"  # Ship-ship collision (mutual)
    TIMEOUT = "timeout"      # Episode ended with both alive


@dataclass
class Ship:
    """Player spaceship."""

    position: Vector2D
    velocity: Vector2D = field(default_factory=Vector2D)
    angle: float = 0.0  # degrees, 0 = facing right
    fuel: float = 100.0
    ammo: int = 15
    fire_cooldown: float = 0.0
    alive: bool = True
    ship_id: int = 0  # 0 or 1
    death_cause: str = DeathCause.ALIVE  # Track how ship died

    # Physics
    mass: float = 1.0
    radius: float = 15.0

    def apply_force(self, force: Vector2D, dt: float):
        """Apply force to ship (F = ma -> a = F/m)."""
        acceleration = force / self.mass
        self.velocity = self.velocity + acceleration * dt

    def update(self, dt: float, width: int, height: int):
        """Update position with screen wrapping."""
        self.position = self.position + self.velocity * dt

        # Screen wrap (toroidal topology)
        self.position.x = self.position.x % width
        self.position.y = self.position.y % height

        # Update cooldown
        self.fire_cooldown = max(0.0, self.fire_cooldown - dt)

    def can_fire(self) -> bool:
        return self.alive and self.ammo > 0 and self.fire_cooldown <= 0

    def can_thrust(self) -> bool:
        return self.alive and self.fuel > 0

    def get_nose_position(self) -> Vector2D:
        """Get position of ship's nose (for torpedo spawn)."""
        return self.position + Vector2D.from_angle(self.angle, self.radius)


@dataclass
class Torpedo:
    """Projectile fired by ships."""

    position: Vector2D
    velocity: Vector2D
    owner_id: int  # Ship that fired this torpedo
    lifetime: float = 3.0
    radius: float = 3.0
    alive: bool = True

    def update(self, dt: float, width: int, height: int):
        """Update torpedo position."""
        self.position = self.position + self.velocity * dt
        self.lifetime -= dt

        # Screen wrap
        self.position.x = self.position.x % width
        self.position.y = self.position.y % height

        if self.lifetime <= 0:
            self.alive = False


# =============================================================================
# Collision Detection
# =============================================================================

def check_circle_collision(pos1: Vector2D, r1: float, pos2: Vector2D, r2: float) -> bool:
    """Check if two circles collide."""
    distance = pos1.distance_to(pos2)
    return distance < (r1 + r2)


def check_collision_with_wrap(
    pos1: Vector2D, r1: float,
    pos2: Vector2D, r2: float,
    width: int, height: int
) -> bool:
    """Check collision accounting for screen wrap."""
    # Check direct collision
    if check_circle_collision(pos1, r1, pos2, r2):
        return True

    # Check wrap-around collisions (check 8 wrapped positions)
    for dx in [-width, 0, width]:
        for dy in [-height, 0, height]:
            if dx == 0 and dy == 0:
                continue
            wrapped_pos = Vector2D(pos2.x + dx, pos2.y + dy)
            if check_circle_collision(pos1, r1, wrapped_pos, r2):
                return True

    return False


# =============================================================================
# Actions
# =============================================================================

class Actions:
    """Discrete action space for ships."""

    NONE = 0
    ROTATE_LEFT = 1
    ROTATE_RIGHT = 2
    THRUST = 3
    FIRE = 4
    THRUST_FIRE = 5  # Combined action

    NUM_ACTIONS = 6

    @staticmethod
    def to_string(action: int) -> str:
        names = ["NONE", "ROTATE_LEFT", "ROTATE_RIGHT", "THRUST", "FIRE", "THRUST_FIRE"]
        return names[action] if 0 <= action < len(names) else "UNKNOWN"


# =============================================================================
# Gymnasium Environment
# =============================================================================

class SpacewarEnv(gym.Env):
    """
    Spacewar! Gymnasium Environment.

    Two ships compete around a central star with gravity.
    Supports self-play training with symmetric observations.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(
        self,
        game_config: Optional[GameConfig] = None,
        reward_config: Optional[RewardConfig] = None,
        render_mode: Optional[str] = None,
    ):
        super().__init__()

        self.game_config = game_config or GameConfig()
        self.reward_config = reward_config or RewardConfig()
        self.render_mode = render_mode

        self.cfg = self.game_config
        self.dt = 1.0 / self.cfg.fps

        # Action space: discrete actions for each ship
        # When training single policy, we step one ship at a time
        self.action_space = spaces.Discrete(Actions.NUM_ACTIONS)

        # Observation space (normalized to [-1, 1] or [0, 1])
        # Own ship: x, y, vx, vy, angle_sin, angle_cos, fuel, ammo, cooldown
        # Opponent: rel_x, rel_y, rel_vx, rel_vy, angle_sin, angle_cos
        # Star: rel_x, rel_y, distance
        # Nearest 3 enemy torpedoes: rel_x, rel_y, rel_vx, rel_vy (each)
        # Action mask: 6 values (can take action or not)
        obs_dim = 9 + 6 + 3 + 12 + 6  # 36 total
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )

        # Rendering
        self.screen = None
        self.clock = None
        self.font = None

        # Game state
        self.star: Optional[Star] = None
        self.ships: List[Ship] = []
        self.torpedoes: List[Torpedo] = []
        self.step_count = 0
        self.episode_rewards = [0.0, 0.0]

        # Running observation stats for normalization
        self._obs_mean = np.zeros(obs_dim, dtype=np.float32)
        self._obs_var = np.ones(obs_dim, dtype=np.float32)
        self._obs_count = 1e-4

    def _create_star(self) -> Star:
        """Create central star."""
        return Star(
            position=Vector2D(self.cfg.width / 2, self.cfg.height / 2),
            mass=self.cfg.star_mass,
            radius=self.cfg.star_radius,
        )

    def _create_ship(self, ship_id: int) -> Ship:
        """Create a ship at starting position."""
        # Ships start on opposite sides of the arena
        if ship_id == 0:
            x = self.cfg.width * 0.2
            y = self.cfg.height / 2
            angle = 0.0  # Facing right
        else:
            x = self.cfg.width * 0.8
            y = self.cfg.height / 2
            angle = 180.0  # Facing left

        return Ship(
            position=Vector2D(x, y),
            velocity=Vector2D(0.0, 0.0),
            angle=angle,
            fuel=self.cfg.max_fuel,
            ammo=self.cfg.max_ammo,
            fire_cooldown=0.0,
            alive=True,
            ship_id=ship_id,
            death_cause=DeathCause.ALIVE,
            mass=self.cfg.ship_mass,
            radius=self.cfg.ship_radius,
        )

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state."""
        super().reset(seed=seed)

        self.star = self._create_star()
        self.ships = [self._create_ship(0), self._create_ship(1)]
        self.torpedoes = []
        self.step_count = 0
        self.episode_rewards = [0.0, 0.0]

        # Add small random velocity to make starts interesting
        if self.np_random is not None:
            for ship in self.ships:
                ship.velocity = Vector2D(
                    self.np_random.uniform(-20, 20),
                    self.np_random.uniform(-20, 20),
                )

        obs = self._get_observation(0)
        info = self._get_info()

        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one timestep.

        For self-play, this steps ship 0. Use step_both() for simultaneous actions.
        """
        return self.step_both(action, self._get_opponent_action())

    def step_both(
        self, action_0: int, action_1: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute actions for both ships simultaneously."""
        self.step_count += 1

        # Apply actions
        self._apply_action(0, action_0)
        self._apply_action(1, action_1)

        # Physics update
        self._physics_step()

        # Check collisions
        rewards = self._check_collisions()

        # Add time penalty
        rewards[0] += self.reward_config.time_penalty
        rewards[1] += self.reward_config.time_penalty

        self.episode_rewards[0] += rewards[0]
        self.episode_rewards[1] += rewards[1]

        # Check termination
        terminated = self._check_termination()
        truncated = self.step_count >= self.cfg.max_episode_steps

        # Add episode outcome rewards
        if terminated or truncated:
            outcome_rewards = self._get_outcome_rewards()
            rewards[0] += outcome_rewards[0]
            rewards[1] += outcome_rewards[1]
            self.episode_rewards[0] += outcome_rewards[0]
            self.episode_rewards[1] += outcome_rewards[1]

        obs = self._get_observation(0)
        info = self._get_info()
        info["reward_1"] = rewards[1]
        info["episode_reward_0"] = self.episode_rewards[0]
        info["episode_reward_1"] = self.episode_rewards[1]

        if self.render_mode == "human":
            self.render()

        return obs, rewards[0], terminated, truncated, info

    def _apply_action(self, ship_id: int, action: int):
        """Apply action to a ship with action masking."""
        ship = self.ships[ship_id]
        if not ship.alive:
            return

        # Rotation
        if action == Actions.ROTATE_LEFT:
            ship.angle -= self.cfg.rotation_speed * self.dt
        elif action == Actions.ROTATE_RIGHT:
            ship.angle += self.cfg.rotation_speed * self.dt

        # Thrust (with fuel check)
        if action in (Actions.THRUST, Actions.THRUST_FIRE):
            if ship.can_thrust():
                thrust_vec = Vector2D.from_angle(ship.angle, self.cfg.thrust_power)
                ship.apply_force(thrust_vec, self.dt)
                ship.fuel -= self.cfg.fuel_consumption * self.dt
                ship.fuel = max(0.0, ship.fuel)

        # Fire (with ammo and cooldown check)
        if action in (Actions.FIRE, Actions.THRUST_FIRE):
            if ship.can_fire():
                self._fire_torpedo(ship)

        # Keep angle in [0, 360)
        ship.angle = ship.angle % 360

    def _fire_torpedo(self, ship: Ship):
        """Fire a torpedo from ship."""
        ship.ammo -= 1
        ship.fire_cooldown = self.cfg.fire_cooldown

        # Torpedo spawns at ship nose, inherits ship velocity + additional speed
        spawn_pos = ship.get_nose_position()
        torpedo_vel = ship.velocity + Vector2D.from_angle(ship.angle, self.cfg.torpedo_speed)

        torpedo = Torpedo(
            position=spawn_pos,
            velocity=torpedo_vel,
            owner_id=ship.ship_id,
            lifetime=self.cfg.torpedo_lifespan,
            radius=self.cfg.torpedo_radius,
        )
        self.torpedoes.append(torpedo)

    def _physics_step(self):
        """Update physics for all objects."""
        # Update ships
        for ship in self.ships:
            if not ship.alive:
                continue

            # Apply gravity
            gravity = self.star.gravitational_force(
                ship.position, ship.mass, self.cfg.gravity_constant
            )
            ship.apply_force(gravity, self.dt)

            # Update position
            ship.update(self.dt, self.cfg.width, self.cfg.height)

        # Update torpedoes (no gravity for simplicity - makes aiming more intuitive)
        for torpedo in self.torpedoes:
            if torpedo.alive:
                torpedo.update(self.dt, self.cfg.width, self.cfg.height)

        # Remove dead torpedoes
        self.torpedoes = [t for t in self.torpedoes if t.alive]

    def _check_collisions(self) -> List[float]:
        """Check and resolve collisions, return rewards."""
        rewards = [0.0, 0.0]

        # Ship-Star collisions (suicide - no credit to opponent)
        for ship in self.ships:
            if not ship.alive:
                continue
            if check_collision_with_wrap(
                ship.position, ship.radius,
                self.star.position, self.star.radius,
                self.cfg.width, self.cfg.height
            ):
                ship.alive = False
                ship.death_cause = DeathCause.STAR
                rewards[ship.ship_id] += self.reward_config.star_collision

        # Ship-Ship collision (mutual destruction)
        if self.ships[0].alive and self.ships[1].alive:
            if check_collision_with_wrap(
                self.ships[0].position, self.ships[0].radius,
                self.ships[1].position, self.ships[1].radius,
                self.cfg.width, self.cfg.height
            ):
                self.ships[0].alive = False
                self.ships[1].alive = False
                self.ships[0].death_cause = DeathCause.COLLISION
                self.ships[1].death_cause = DeathCause.COLLISION
                rewards[0] += self.reward_config.ship_collision
                rewards[1] += self.reward_config.ship_collision

        # Torpedo-Ship collisions (real kills - credit to shooter)
        for torpedo in self.torpedoes:
            if not torpedo.alive:
                continue

            for ship in self.ships:
                if not ship.alive:
                    continue
                if ship.ship_id == torpedo.owner_id:
                    continue  # Can't hit yourself

                if check_collision_with_wrap(
                    torpedo.position, torpedo.radius,
                    ship.position, ship.radius,
                    self.cfg.width, self.cfg.height
                ):
                    ship.alive = False
                    ship.death_cause = DeathCause.TORPEDO
                    torpedo.alive = False
                    # Reward shooter, penalize victim
                    rewards[torpedo.owner_id] += self.reward_config.kill_opponent
                    rewards[ship.ship_id] += self.reward_config.killed_by_opponent

        # Torpedo-Star collisions
        for torpedo in self.torpedoes:
            if not torpedo.alive:
                continue
            if check_collision_with_wrap(
                torpedo.position, torpedo.radius,
                self.star.position, self.star.radius,
                self.cfg.width, self.cfg.height
            ):
                torpedo.alive = False

        return rewards

    def _check_termination(self) -> bool:
        """Check if episode should terminate."""
        alive_count = sum(1 for s in self.ships if s.alive)
        return alive_count < 2  # End when at least one ship is destroyed

    def _get_outcome_rewards(self) -> List[float]:
        """Get end-of-episode outcome rewards including efficiency bonuses.

        Win bonus is only awarded for real kills (torpedo hits).
        Opponent suicides (star collision) don't count as wins.
        """
        rewards = [0.0, 0.0]

        ship0 = self.ships[0]
        ship1 = self.ships[1]

        if ship0.alive and not ship1.alive:
            # Ship 0 survived, Ship 1 died
            if ship1.death_cause == DeathCause.TORPEDO:
                # Real kill - ship 0 shot ship 1
                rewards[0] += self.reward_config.win_bonus
                rewards[1] += self.reward_config.lose_penalty
                rewards[0] += self._calculate_efficiency_bonus(ship0)
            else:
                # Opponent suicided (star collision) - no win credit
                # Survivor gets nothing extra, suicide already penalized in collision
                pass
        elif ship1.alive and not ship0.alive:
            # Ship 1 survived, Ship 0 died
            if ship0.death_cause == DeathCause.TORPEDO:
                # Real kill - ship 1 shot ship 0
                rewards[1] += self.reward_config.win_bonus
                rewards[0] += self.reward_config.lose_penalty
                rewards[1] += self._calculate_efficiency_bonus(ship1)
            else:
                # Opponent suicided (star collision) - no win credit
                pass
        elif not ship0.alive and not ship1.alive:
            # Both dead - draw (collision or mutual destruction)
            rewards[0] += self.reward_config.draw_penalty
            rewards[1] += self.reward_config.draw_penalty
        else:
            # Both alive (timeout) - slight penalty
            rewards[0] += self.reward_config.draw_penalty
            rewards[1] += self.reward_config.draw_penalty

        return rewards

    def _calculate_efficiency_bonus(self, ship) -> float:
        """Calculate efficiency bonus based on remaining fuel and ammo."""
        # Fuel efficiency: remaining / max (0 to 1)
        fuel_ratio = ship.fuel / self.cfg.max_fuel if self.cfg.max_fuel > 0 else 0
        fuel_bonus = fuel_ratio * self.reward_config.fuel_efficiency_bonus

        # Ammo efficiency: remaining / max (0 to 1)
        ammo_ratio = ship.ammo / self.cfg.max_ammo if self.cfg.max_ammo > 0 else 0
        ammo_bonus = ammo_ratio * self.reward_config.ammo_efficiency_bonus

        return fuel_bonus + ammo_bonus

    def _get_observation(self, ship_id: int) -> np.ndarray:
        """Get normalized observation for a ship."""
        ship = self.ships[ship_id]
        opponent = self.ships[1 - ship_id]

        obs = []

        # === Own ship state (9 values) ===
        # Position (normalized to [0, 1] then shifted to [-1, 1])
        obs.append(ship.position.x / self.cfg.width * 2 - 1)
        obs.append(ship.position.y / self.cfg.height * 2 - 1)

        # Velocity (normalized by typical max speed ~200)
        max_speed = 200.0
        obs.append(np.clip(ship.velocity.x / max_speed, -1, 1))
        obs.append(np.clip(ship.velocity.y / max_speed, -1, 1))

        # Angle as sin/cos (for continuity)
        angle_rad = math.radians(ship.angle)
        obs.append(math.sin(angle_rad))
        obs.append(math.cos(angle_rad))

        # Resources (normalized to [0, 1])
        obs.append(ship.fuel / self.cfg.max_fuel)
        obs.append(ship.ammo / self.cfg.max_ammo)
        obs.append(1.0 - min(ship.fire_cooldown / self.cfg.fire_cooldown, 1.0))

        # === Opponent state (6 values) ===
        # Relative position
        rel_pos = opponent.position - ship.position
        obs.append(np.clip(rel_pos.x / (self.cfg.width / 2), -1, 1))
        obs.append(np.clip(rel_pos.y / (self.cfg.height / 2), -1, 1))

        # Relative velocity
        rel_vel = opponent.velocity - ship.velocity
        obs.append(np.clip(rel_vel.x / max_speed, -1, 1))
        obs.append(np.clip(rel_vel.y / max_speed, -1, 1))

        # Opponent angle
        opp_angle_rad = math.radians(opponent.angle)
        obs.append(math.sin(opp_angle_rad))
        obs.append(math.cos(opp_angle_rad))

        # === Star state (3 values) ===
        rel_star = self.star.position - ship.position
        obs.append(np.clip(rel_star.x / (self.cfg.width / 2), -1, 1))
        obs.append(np.clip(rel_star.y / (self.cfg.height / 2), -1, 1))

        # Distance to star (normalized, 0 = far, 1 = close/dangerous)
        star_dist = rel_star.magnitude()
        max_dist = math.sqrt(self.cfg.width**2 + self.cfg.height**2) / 2
        obs.append(1.0 - min(star_dist / max_dist, 1.0))

        # === Nearest 3 enemy torpedoes (12 values) ===
        enemy_torpedoes = [t for t in self.torpedoes if t.alive and t.owner_id != ship_id]
        enemy_torpedoes.sort(key=lambda t: t.position.distance_to(ship.position))

        for i in range(3):
            if i < len(enemy_torpedoes):
                torp = enemy_torpedoes[i]
                rel_t = torp.position - ship.position
                rel_tv = torp.velocity - ship.velocity
                obs.append(np.clip(rel_t.x / (self.cfg.width / 2), -1, 1))
                obs.append(np.clip(rel_t.y / (self.cfg.height / 2), -1, 1))
                obs.append(np.clip(rel_tv.x / max_speed, -1, 1))
                obs.append(np.clip(rel_tv.y / max_speed, -1, 1))
            else:
                obs.extend([0.0, 0.0, 0.0, 0.0])  # No torpedo

        # === Action mask (6 values) ===
        obs.extend(self._get_action_mask(ship_id))

        return np.array(obs, dtype=np.float32)

    def _get_action_mask(self, ship_id: int) -> List[float]:
        """Get action availability mask (1.0 = available, 0.0 = unavailable)."""
        ship = self.ships[ship_id]

        mask = [1.0] * Actions.NUM_ACTIONS

        if not ship.alive:
            return [0.0] * Actions.NUM_ACTIONS

        # Can always do nothing or rotate
        # Thrust requires fuel
        if not ship.can_thrust():
            mask[Actions.THRUST] = 0.0
            mask[Actions.THRUST_FIRE] = 0.0

        # Fire requires ammo and cooldown
        if not ship.can_fire():
            mask[Actions.FIRE] = 0.0
            if not ship.can_thrust():
                mask[Actions.THRUST_FIRE] = 0.0
            else:
                # Can thrust but not fire - THRUST_FIRE becomes just THRUST
                pass

        return mask

    def get_action_mask(self, ship_id: int = 0) -> np.ndarray:
        """Public method to get action mask as numpy array."""
        return np.array(self._get_action_mask(ship_id), dtype=np.float32)

    def _get_opponent_action(self) -> int:
        """Get action for opponent (used when stepping single agent)."""
        # Simple heuristic opponent for basic training
        # In self-play, this should be replaced with learned policy
        return self._heuristic_action(1)

    def _heuristic_action(self, ship_id: int) -> int:
        """Simple rule-based action for a ship."""
        ship = self.ships[ship_id]
        opponent = self.ships[1 - ship_id]

        if not ship.alive:
            return Actions.NONE

        # Calculate angle to opponent
        to_opponent = opponent.position - ship.position
        target_angle = math.degrees(math.atan2(to_opponent.y, to_opponent.x))
        angle_diff = (target_angle - ship.angle + 180) % 360 - 180

        # Calculate distance to star
        to_star = self.star.position - ship.position
        star_dist = to_star.magnitude()

        # Priority 1: Avoid star
        if star_dist < self.reward_config.safe_star_distance:
            # Thrust away from star
            if ship.can_thrust():
                return Actions.THRUST

        # Priority 2: Aim at opponent
        if abs(angle_diff) > 15:
            if angle_diff > 0:
                return Actions.ROTATE_RIGHT
            else:
                return Actions.ROTATE_LEFT

        # Priority 3: Fire if aimed and have ammo
        if abs(angle_diff) < 10 and ship.can_fire():
            if ship.can_thrust():
                return Actions.THRUST_FIRE
            return Actions.FIRE

        # Priority 4: Thrust toward opponent
        if ship.can_thrust():
            return Actions.THRUST

        return Actions.NONE

    def _get_info(self) -> Dict[str, Any]:
        """Get info dictionary."""
        ship0 = self.ships[0]
        ship1 = self.ships[1]

        # Determine real winner (only torpedo kills count)
        real_winner = None
        if ship0.alive and not ship1.alive and ship1.death_cause == DeathCause.TORPEDO:
            real_winner = 0
        elif ship1.alive and not ship0.alive and ship0.death_cause == DeathCause.TORPEDO:
            real_winner = 1

        return {
            "step": self.step_count,
            "ship_0_alive": ship0.alive,
            "ship_1_alive": ship1.alive,
            "ship_0_death_cause": ship0.death_cause,
            "ship_1_death_cause": ship1.death_cause,
            "ship_0_fuel": ship0.fuel,
            "ship_1_fuel": ship1.fuel,
            "ship_0_ammo": ship0.ammo,
            "ship_1_ammo": ship1.ammo,
            "torpedoes": len(self.torpedoes),
            "real_winner": real_winner,  # Only set if winner killed opponent
        }

    # =========================================================================
    # Human Input
    # =========================================================================

    def get_human_action(self) -> int:
        """
        Get action from keyboard input.

        Controls:
            Left Arrow / A  - Rotate left
            Right Arrow / D - Rotate right
            Up Arrow / W    - Thrust
            Space           - Fire
            Up + Space      - Thrust + Fire

        Returns action integer 0-5.
        """
        if not PYGAME_AVAILABLE:
            return Actions.NONE

        keys = pygame.key.get_pressed()

        # Check for rotation
        rotate_left = keys[pygame.K_LEFT] or keys[pygame.K_a]
        rotate_right = keys[pygame.K_RIGHT] or keys[pygame.K_d]

        # Check for thrust
        thrust = keys[pygame.K_UP] or keys[pygame.K_w]

        # Check for fire
        fire = keys[pygame.K_SPACE]

        # Determine action based on key combination
        if thrust and fire:
            return Actions.THRUST_FIRE
        elif fire:
            return Actions.FIRE
        elif thrust:
            return Actions.THRUST
        elif rotate_left:
            return Actions.ROTATE_LEFT
        elif rotate_right:
            return Actions.ROTATE_RIGHT
        else:
            return Actions.NONE

    # =========================================================================
    # Rendering
    # =========================================================================

    def render(self) -> Optional[np.ndarray]:
        """Render the game."""
        if not PYGAME_AVAILABLE:
            return None

        if self.screen is None:
            pygame.init()
            pygame.display.init()

            if self.render_mode == "human":
                self.screen = pygame.display.set_mode(
                    (self.cfg.width, self.cfg.height)
                )
                pygame.display.set_caption("Spacewar! AI")
            else:
                self.screen = pygame.Surface((self.cfg.width, self.cfg.height))

            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 24)

        # Handle events
        if self.render_mode == "human":
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
                    return None

        # Clear screen with starfield
        self.screen.fill((0, 0, 10))
        self._draw_starfield()

        # Draw star
        self._draw_star()

        # Draw ships
        for ship in self.ships:
            self._draw_ship(ship)

        # Draw torpedoes
        for torpedo in self.torpedoes:
            self._draw_torpedo(torpedo)

        # Draw HUD
        self._draw_hud()

        if self.render_mode == "human":
            pygame.display.flip()
            self.clock.tick(self.cfg.fps)
        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

        return None

    def _draw_starfield(self):
        """Draw background stars."""
        # Fixed seed for consistent starfield
        rng = np.random.default_rng(42)
        for _ in range(100):
            x = int(rng.uniform(0, self.cfg.width))
            y = int(rng.uniform(0, self.cfg.height))
            brightness = int(rng.uniform(50, 200))
            self.screen.set_at((x, y), (brightness, brightness, brightness))

    def _draw_star(self):
        """Draw central star with glow effect."""
        center = (int(self.star.position.x), int(self.star.position.y))

        # Glow layers
        for i in range(5, 0, -1):
            radius = int(self.star.radius + i * 5)
            alpha = 50 - i * 10
            color = (255, 200, 50, max(0, alpha))
            pygame.draw.circle(self.screen, (255, 200, 50), center, radius)

        # Core
        pygame.draw.circle(self.screen, (255, 255, 200), center, int(self.star.radius))

    def _draw_ship(self, ship: Ship):
        """Draw a ship as a triangle."""
        if not ship.alive:
            return

        # Ship colors
        colors = [(100, 200, 255), (255, 100, 100)]  # Blue, Red
        color = colors[ship.ship_id]

        # Triangle points (nose, left wing, right wing)
        angle_rad = math.radians(ship.angle)
        nose = (
            ship.position.x + math.cos(angle_rad) * ship.radius,
            ship.position.y + math.sin(angle_rad) * ship.radius,
        )
        left = (
            ship.position.x + math.cos(angle_rad + 2.5) * ship.radius * 0.7,
            ship.position.y + math.sin(angle_rad + 2.5) * ship.radius * 0.7,
        )
        right = (
            ship.position.x + math.cos(angle_rad - 2.5) * ship.radius * 0.7,
            ship.position.y + math.sin(angle_rad - 2.5) * ship.radius * 0.7,
        )

        pygame.draw.polygon(self.screen, color, [nose, left, right])

    def _draw_torpedo(self, torpedo: Torpedo):
        """Draw a torpedo."""
        if not torpedo.alive:
            return

        colors = [(150, 220, 255), (255, 150, 150)]
        color = colors[torpedo.owner_id]

        pos = (int(torpedo.position.x), int(torpedo.position.y))
        pygame.draw.circle(self.screen, color, pos, int(torpedo.radius))

    def _draw_hud(self):
        """Draw heads-up display."""
        # Ship 0 stats (left side)
        ship0 = self.ships[0]
        text = f"P1 Fuel:{ship0.fuel:.0f} Ammo:{ship0.ammo}"
        if not ship0.alive:
            text = "P1 DESTROYED"
        surface = self.font.render(text, True, (100, 200, 255))
        self.screen.blit(surface, (10, 10))

        # Ship 1 stats (right side)
        ship1 = self.ships[1]
        text = f"P2 Fuel:{ship1.fuel:.0f} Ammo:{ship1.ammo}"
        if not ship1.alive:
            text = "P2 DESTROYED"
        surface = self.font.render(text, True, (255, 100, 100))
        self.screen.blit(surface, (self.cfg.width - 150, 10))

        # Step count
        text = f"Step: {self.step_count}"
        surface = self.font.render(text, True, (200, 200, 200))
        self.screen.blit(surface, (self.cfg.width // 2 - 40, 10))

    def close(self):
        """Clean up resources."""
        if self.screen is not None and PYGAME_AVAILABLE:
            pygame.display.quit()
            pygame.quit()
            self.screen = None
            self.clock = None
