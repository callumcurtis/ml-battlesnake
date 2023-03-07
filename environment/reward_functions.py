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


class RewardChain(RewardFunction):

    def __init__(self, reward_functions: list[RewardFunction]):
        self.reward_functions = reward_functions
    
    def calculate(
        self,
        memory_buffer: MemoryBuffer,
        this_timestep_builder: TimestepBuilder,
    ) -> TimestepBuilder:
        for reward_function in self.reward_functions:
            this_timestep_builder = reward_function.calculate(
                memory_buffer=memory_buffer,
                this_timestep_builder=this_timestep_builder,
            )
        return this_timestep_builder


class RewardWinLoseDrawSurvival(RewardFunction):

    def __init__(
        self,
        win_reward: float = 1.0,
        lose_reward: float = -1.0,
        draw_reward: float = -1.0,
        survival_reward: float = 0.001,
    ):
        self.win_reward = win_reward
        self.lose_reward = lose_reward
        self.draw_reward = draw_reward
        self.survival_reward = survival_reward
    
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
            survival_reward = self.win_reward if n_alive_snakes == 1 else self.survival_reward
        else:
            # If there is only one snake (solo mode), then the snake can only eventually lose
            death_reward = self.lose_reward
            survival_reward = self.survival_reward
        rewards = {
            snake: death_reward if is_terminated else survival_reward
            for snake, is_terminated in this_timestep_builder.terminations.items()
        }
        this_timestep_builder.with_rewards(rewards)
        return this_timestep_builder
