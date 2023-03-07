import abc

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
        aggregate_rewards = {}
        for reward_function in self.reward_functions:
            rewards = reward_function.calculate(
                memory_buffer=memory_buffer,
                this_timestep_builder=this_timestep_builder,
            )
            for snake_id, reward in rewards.items():
                aggregate_rewards[snake_id] += reward
        return aggregate_rewards


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
