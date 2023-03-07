import abc
from typing import Callable

from environment.memory import MemoryBuffer
from environment.types import TimestepBuilder


class RewardFunction(abc.ABC):

    NO_REWARD = 0.0

    @abc.abstractmethod
    def calculate(
        self,
        memory_buffer: MemoryBuffer,
        this_timestep_builder: TimestepBuilder,
    ) -> dict[str, float]:
        pass

    @property
    @abc.abstractmethod
    def required_memory_size(self) -> int:
        pass


class RewardChain(RewardFunction):

    def __init__(
        self,
        reward_functions: list[RewardFunction],
    ):
        self.reward_functions = reward_functions
    
    def calculate(
        self,
        memory_buffer: MemoryBuffer,
        this_timestep_builder: TimestepBuilder,
    ) -> dict[str, float]:
        assert len(memory_buffer) >= self.required_memory_size
        aggregate_rewards = {}
        for reward_function in self.reward_functions:
            rewards = reward_function.calculate(
                memory_buffer=memory_buffer,
                this_timestep_builder=this_timestep_builder,
            )
            for snake_id, reward in rewards.items():
                aggregate_rewards[snake_id] = aggregate_rewards.get(snake_id, self.NO_REWARD) + reward
        return aggregate_rewards
    
    @property
    def required_memory_size(self) -> int:
        return max(
            reward_function.required_memory_size
            for reward_function in self.reward_functions
        )


class RewardWinLoseDraw(RewardFunction):

    _SURVIVAL_REWARD = RewardFunction.NO_REWARD

    def __init__(
        self,
        win_reward: float = 1.0,
        lose_reward: float = -1.0,
        draw_reward: float = -1.0,
    ):
        self.win_reward = win_reward
        self.lose_reward = lose_reward
        self.draw_reward = draw_reward
    
    def calculate(
        self,
        memory_buffer: MemoryBuffer,
        this_timestep_builder: TimestepBuilder,
    ) -> dict[str, float]:
        assert this_timestep_builder.terminations
        n_snakes = len(this_timestep_builder.terminations)
        n_dead_snakes = sum(
            is_terminated
            for is_terminated in this_timestep_builder.terminations.values()
        )
        n_alive_snakes = n_snakes - n_dead_snakes
        if n_snakes > 1:
            death_reward = self.lose_reward if n_alive_snakes > 0 else self.draw_reward
            survival_reward = self.win_reward if n_alive_snakes == 1 else self._SURVIVAL_REWARD
        else:
            # If there is only one snake (solo mode), then the snake can only eventually lose
            death_reward = self.lose_reward
            survival_reward = self._SURVIVAL_REWARD
        rewards = {
            snake: death_reward if is_terminated else survival_reward
            for snake, is_terminated in this_timestep_builder.terminations.items()
        }
        return rewards
    
    @property
    def required_memory_size(self) -> int:
        return 0


class RewardSurvival(RewardFunction):

    def __init__(
        self,
        reward: float = 0.001,
    ):
        self.reward = reward
    
    def calculate(
        self,
        memory_buffer: MemoryBuffer,
        this_timestep_builder: TimestepBuilder,
    ) -> dict[str, float]:
        assert this_timestep_builder.terminations
        rewards = {
            snake: self.NO_REWARD if is_terminated else self.reward
            for snake, is_terminated in this_timestep_builder.terminations.items()
        }
        return rewards
    
    @property
    def required_memory_size(self) -> int:
        return 0


class RewardOpponentDeath(RewardFunction):

    def __init__(
        self,
        outlive_reward: float = 0.01,
        collision_reward: float = 0.05,
        consume_reward: float = 0.05,
    ):
        self.outlive_reward = outlive_reward
        self.collision_reward = collision_reward
        self.consume_reward = consume_reward
    
    def calculate(
        self,
        memory_buffer: MemoryBuffer,
        this_timestep_builder: TimestepBuilder,
    ) -> dict[str, float]:
        assert this_timestep_builder.terminations
        assert this_timestep_builder.raw_observations

        snakes = list(this_timestep_builder.terminations.keys())
        dead_snakes = [snake for snake in snakes if this_timestep_builder.terminations[snake]]
        alive_snakes = [snake for snake in snakes if not this_timestep_builder.terminations[snake]]

        if len(dead_snakes) == 0:
            return {snake: self.NO_REWARD for snake in snakes}
    
        head_position_by_snake = {
            snake: observation["you"]["head"]
            for snake, observation in this_timestep_builder.raw_observations.items()
        }
        body_positions_by_snake = {
            snake: observation["you"]["body"]
            for snake, observation in this_timestep_builder.raw_observations.items()
        }

        rewards = {snake: self.NO_REWARD if snake in dead_snakes else self.outlive_reward for snake in snakes}
        for dead_snake in dead_snakes:
            dead_snake_head_position = head_position_by_snake[dead_snake]
            reward = None
            for alive_snake in alive_snakes:
                if dead_snake_head_position == head_position_by_snake[alive_snake]:
                    reward = self.consume_reward
                elif dead_snake_head_position in body_positions_by_snake[alive_snake]:
                    reward = self.collision_reward
                if reward is not None:
                    rewards[alive_snake] += reward
                    break

        return rewards

    @property
    def required_memory_size(self) -> int:
        return 0


class RewardFoodConsumption(RewardFunction):

    def __init__(
        self,
        reward_schedule: Callable[[int], float] = lambda health: 0.01 * (0.98 ** health),
    ):
        self.reward_schedule = reward_schedule
    
    def calculate(
        self,
        memory_buffer: MemoryBuffer,
        this_timestep_builder: TimestepBuilder,
    ) -> dict[str, float]:
        assert this_timestep_builder.raw_observations
        assert len(memory_buffer) >= self.required_memory_size

        previous_raw_observations = memory_buffer[-1].raw_observations
        current_raw_observations = this_timestep_builder.raw_observations
        current_snakes = set(current_raw_observations.keys())
        previous_health_by_snake = {
            snake: previous_raw_observations[snake]["you"]["health"]
            for snake in current_snakes
        }
        current_health_by_snake = {
            snake: current_raw_observations[snake]["you"]["health"]
            for snake in current_snakes
        }

        rewards = {
            snake: self.reward_schedule(previous_health_by_snake[snake])
            if current_health_by_snake[snake] >= previous_health_by_snake[snake]
            else self.NO_REWARD
            for snake in current_snakes
        }

        return rewards

    @property
    def required_memory_size(self) -> int:
        return 1
