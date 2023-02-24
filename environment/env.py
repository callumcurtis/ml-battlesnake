import functools

import pettingzoo
import gymnasium

from environment.configuration import BattlesnakeEnvironmentConfiguration
from environment.adapters import BattlesnakeEngineForParallelEnv


def make_env(
    engine: BattlesnakeEngineForParallelEnv,
    configuration: BattlesnakeEnvironmentConfiguration,
):
    env = BattlesnakeEnvironment(engine, configuration)
    return env


class BattlesnakeEnvironment(pettingzoo.ParallelEnv):

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        engine: BattlesnakeEngineForParallelEnv,
        configuration: BattlesnakeEnvironmentConfiguration,
    ):
        self.engine = engine
        self.configuration = configuration

    @property
    def possible_agents(self):
        return self.configuration.possible_agents

    @functools.cache
    def observation_space(self, agent) -> gymnasium.spaces.Space:
        return self.engine.observation_space

    @functools.cache
    def action_space(self, agent) -> gymnasium.spaces.Space:
        return self.engine.action_space
    
    def render(self):
        assert self.configuration.render_mode == "human", "Only human render mode is supported"
        self.engine.render()
    
    def reset(self, seed=None, return_info=False, options=None):
        self.agents = self.possible_agents.copy()

        if seed:
            self.configuration.seed = seed

        observations, infos = self.engine.reset(self.configuration)
        return (observations, infos) if return_info else observations
    
    def step(self, action):
        observations, rewards, terminations, infos = self.engine.step(action)
        truncations = {agent: False for agent in self.agents}

        self.agents = [agent for agent in self.agents if not terminations[agent] and not truncations[agent]]

        return observations, rewards, terminations, truncations, infos
