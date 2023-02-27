import functools

import pettingzoo
import gymnasium

from environment.configuration import BattlesnakeEnvironmentConfiguration
from environment.adapters import BattlesnakeEngineForParallelEnv, Movement
from environment.observation_transformers import ObservationTransformer


def make_env(
    engine_adapter: BattlesnakeEngineForParallelEnv,
    observation_transformer: ObservationTransformer,
    configuration: BattlesnakeEnvironmentConfiguration,
):
    env = BattlesnakeEnvironment(
        engine_adapter=engine_adapter,
        observation_transformer=observation_transformer,
        configuration=configuration,
    )
    return env


class BattlesnakeEnvironment(pettingzoo.ParallelEnv):

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        engine_adapter: BattlesnakeEngineForParallelEnv,
        observation_transformer: ObservationTransformer,
        configuration: BattlesnakeEnvironmentConfiguration,
    ):
        self.engine_adapter = engine_adapter
        self.observation_transformer = observation_transformer
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

        observations, infos = self.engine_adapter.reset(self.configuration)
        observations = {
            agent: self.observation_transformer.transform(obs)
            for agent, obs in observations.items()
        }
        return (observations, infos) if return_info else observations
    
    def step(self, action):
        observations, rewards, terminations, infos = self.engine_adapter.step(action)

        if self.engine_adapter.is_game_over():
            terminations = dict.fromkeys(terminations, True)

        truncations = {agent: False for agent in self.agents}

        self.agents = [agent for agent in self.agents if not terminations[agent] and not truncations[agent]]

        observations = {
            agent: self.observation_transformer.transform(obs)
            if agent in self.agents
            else self.observation_transformer.empty_observation()
            for agent, obs in observations.items()
        }

        return observations, rewards, terminations, truncations, infos
