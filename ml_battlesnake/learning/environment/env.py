"""Battlesnake environments in PettingZoo and Gymnasium formats."""

import functools

import pettingzoo
import gymnasium
import supersuit

from .types import BattlesnakeEnvironmentConfiguration, InitialStateBuilder, TimestepBuilder
from .adapters import BattlesnakeEngineForParallelEnv, Movement
from .observation_transformers import ObservationTransformer
from .reward_functions import RewardFunction
from .memory import MemoryBuffer


def make_parallel_pettingzoo_env(
    engine_adapter: BattlesnakeEngineForParallelEnv,
    observation_transformer: ObservationTransformer,
    reward_function: RewardFunction,
    memory_buffer: MemoryBuffer,
    configuration: BattlesnakeEnvironmentConfiguration,
) -> pettingzoo.ParallelEnv:
    """Creates a parallel PettingZoo environment for Battlesnake."""
    env = BattlesnakeEnvironment(
        engine_adapter=engine_adapter,
        observation_transformer=observation_transformer,
        reward_function=reward_function,
        memory_buffer=memory_buffer,
        configuration=configuration,
    )
    # Adds support for agent removal during episodes
    env = supersuit.black_death_v3(env)
    return env


def make_gymnasium_vec_env(
    engine_adapter: BattlesnakeEngineForParallelEnv,
    observation_transformer: ObservationTransformer,
    reward_function: RewardFunction,
    memory_buffer: MemoryBuffer,
    configuration: BattlesnakeEnvironmentConfiguration,
) -> gymnasium.vector.VectorEnv:
    """Creates a vectorized Gymnasium environment for Battlesnake.
    
    Allows client algorithms to treat the environment as a vector of single-agent
    environments, one for each Battlesnake. The size of the vector is equal to the
    maximum number of agents in the environment.
    """
    env = make_parallel_pettingzoo_env(
        engine_adapter=engine_adapter,
        observation_transformer=observation_transformer,
        reward_function=reward_function,
        memory_buffer=memory_buffer,
        configuration=configuration,
    )
    env = supersuit.pettingzoo_env_to_vec_env_v1(env)
    return env


class BattlesnakeEnvironment(pettingzoo.ParallelEnv):
    """Battlesnake environment in parallel PettingZoo format.
    
    Refer to the official documentation for the parallel PettingZoo environment API.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        engine_adapter: BattlesnakeEngineForParallelEnv,
        observation_transformer: ObservationTransformer,
        reward_function: RewardFunction,
        memory_buffer: MemoryBuffer,
        configuration: BattlesnakeEnvironmentConfiguration,
    ):
        self.engine_adapter = engine_adapter
        self.observation_transformer = observation_transformer
        self.reward_function = reward_function
        self.memory_buffer = memory_buffer
        self.configuration = configuration

    @property
    def possible_agents(self) -> list[str]:
        return self.configuration.possible_agents

    @functools.cache
    def observation_space(self, agent) -> gymnasium.spaces.Space:
        return self.observation_transformer.space

    @functools.cache
    def action_space(self, agent) -> gymnasium.spaces.Space:
        return gymnasium.spaces.Discrete(len(Movement))

    def render(self) -> None:
        assert self.configuration.render_mode == "human", "Only human render mode is supported"
        self.engine_adapter.render()
    
    def reset(self, seed=None, return_info=False, options=None):
        self.agents = self.possible_agents.copy()

        if seed:
            self.configuration.seed = seed

        initial_state_builder = InitialStateBuilder().with_configuration(self.configuration)

        initial_state_builder = self.engine_adapter.reset(initial_state_builder)
        
        initial_state = initial_state_builder.build()

        self.memory_buffer.reset(initial_state)

        transformed_observations = self.observation_transformer.transform_all(initial_state.observations)
        return (
            (transformed_observations, initial_state.infos)
            if return_info
            else transformed_observations
        )
    
    def step(self, actions):
        timestep_builder = TimestepBuilder().with_actions(actions)
        timestep_builder = self.engine_adapter.step(timestep_builder)

        timestep_builder.with_rewards(self.reward_function.calculate(self.memory_buffer, timestep_builder))

        if self.engine_adapter.is_game_over():
            timestep_builder.with_terminations(dict.fromkeys(timestep_builder.terminations, True))

        timestep_builder.with_truncations(timestep_builder.terminations.copy())

        self.agents = [
            agent for agent in self.agents
            if not timestep_builder.terminations[agent]
            and not timestep_builder.truncations[agent]
        ]

        timestep = timestep_builder.build()

        self.memory_buffer.add(timestep)

        transformed_observations = self.observation_transformer.transform_all(timestep.observations)
        return (
            transformed_observations,
            timestep.rewards,
            timestep.terminations,
            timestep.truncations,
            timestep.infos,
        )
