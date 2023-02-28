import abc
import copy

from environment.memory import MemoryBuffer
from environment.types import TimestepBuilder


class RewardFunction(abc.ABC):

    @abc.abstractmethod
    def calculate(
        self,
        memory_buffer: MemoryBuffer,
        this_timestep_builder: TimestepBuilder,
    ) -> TimestepBuilder:
        pass


class RewardWinLoseDraw(RewardFunction):

    NO_REWARD = 0.0

    def __init__(
        self,
        win_reward: float,
        lose_reward: float,
        draw_reward: float,
    ):
        self.win_reward = win_reward
        self.lose_reward = lose_reward
        self.draw_reward = draw_reward
    
    def calculate(
        self,
        memory_buffer: MemoryBuffer,
        this_timestep_builder: TimestepBuilder,
    ) -> TimestepBuilder:
        assert this_timestep_builder.terminations
        this_timestep_builder = copy.deepcopy(this_timestep_builder)
        n_snakes = len(this_timestep_builder.terminations)
        n_dead_snakes = sum(
            is_terminated
            for is_terminated in this_timestep_builder.terminations.values()
        )
        n_alive_snakes = n_snakes - n_dead_snakes
        if n_snakes > 1:
            death_reward = self.lose_reward if n_alive_snakes > 0 else self.draw_reward
            survival_reward = self.win_reward if n_alive_snakes == 1 else self.NO_REWARD
        else:
            # If there is only one snake (solo mode), then the snake can only eventually lose
            death_reward = self.lose_reward
            survival_reward = self.NO_REWARD
        rewards = {
            snake: death_reward if is_terminated else survival_reward
            for snake, is_terminated in this_timestep_builder.terminations.items()
        }
        this_timestep_builder.with_rewards(rewards)
        return this_timestep_builder
