import functools

import pettingzoo
import gymnasium
import supersuit

from environment.types import BattlesnakeEnvironmentConfiguration, InitialStateBuilder, TimestepBuilder
from environment.adapters import BattlesnakeEngineForParallelEnv, Movement
from environment.observation_transformers import ObservationTransformer
from environment.reward_functions import RewardFunction
from environment.memory import MemoryBuffer


def make_parallel_pettingzoo_env(
    engine_adapter: BattlesnakeEngineForParallelEnv,
    observation_transformer: ObservationTransformer,
    reward_function: RewardFunction,
    memory_buffer: MemoryBuffer,
    configuration: BattlesnakeEnvironmentConfiguration,
):
    env = BattlesnakeEnvironment(
        engine_adapter=engine_adapter,
        observation_transformer=observation_transformer,
        reward_function=reward_function,
        memory_buffer=memory_buffer,
        configuration=configuration,
    )
    env = supersuit.black_death_v3(env)
    return env


def make_gymnasium_vec_env(
    engine_adapter: BattlesnakeEngineForParallelEnv,
    observation_transformer: ObservationTransformer,
    reward_function: RewardFunction,
    memory_buffer: MemoryBuffer,
    configuration: BattlesnakeEnvironmentConfiguration,
):
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
    def possible_agents(self):
        return self.configuration.possible_agents

    @functools.cache
    def observation_space(self, agent) -> gymnasium.spaces.Space:
        return self.observation_transformer.space

    @functools.cache
    def action_space(self, agent) -> gymnasium.spaces.Space:
        return gymnasium.spaces.Discrete(len(Movement))

    def render(self):
        assert self.configuration.render_mode == "human", "Only human render mode is supported"
        self.engine_adapter.render()
    
    def reset(self, seed=None, return_info=False, options=None):
        self.agents = self.possible_agents.copy()

        if seed:
            self.configuration.seed = seed

        initial_state_builder = InitialStateBuilder().with_configuration(self.configuration)

        initial_state_builder = self.engine_adapter.reset(initial_state_builder)

        initial_state_builder.with_observations({
            agent: self.observation_transformer.transform(obs)
            for agent, obs in initial_state_builder.observations.items()
        })
        
        initial_state = initial_state_builder.build()

        self.memory_buffer.reset(initial_state)

        return (
            (initial_state.observations, initial_state.infos)
            if return_info
            else initial_state.observations
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

        timestep_builder.with_observations({
            agent: self.observation_transformer.transform(obs)
            if agent in self.agents
            else self.observation_transformer.empty_observation()
            for agent, obs in timestep_builder.observations.items()
        })

        timestep = timestep_builder.build()

        self.memory_buffer.add(timestep)

        return (
            timestep.observations,
            timestep.rewards,
            timestep.terminations,
            timestep.truncations,
            timestep.infos,
        )
