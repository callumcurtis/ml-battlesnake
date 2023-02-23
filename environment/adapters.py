import abc

import gymnasium

from environment.engines import BattlesnakeEngine, BattlesnakeDllEngine
from environment.configuration import BattlesnakeEnvironmentConfiguration
from environment.observation_transformers import ObservationTransformer


class BattlesnakeEngineForParallelEnv(abc.ABC):

    @property
    @abc.abstractmethod
    def engine(self) -> BattlesnakeEngine:
        pass

    @property
    @abc.abstractmethod
    def observation_transformer(self) -> ObservationTransformer:
        pass

    @abc.abstractmethod
    def render(self):
        pass

    @abc.abstractmethod
    def is_game_over(self) -> bool:
        pass

    @abc.abstractmethod
    def _reset(self, env_config: BattlesnakeEnvironmentConfiguration) -> tuple[dict, dict]:
        pass

    @abc.abstractmethod
    def _step(self, actions) -> tuple[dict, dict, dict, dict]:
        pass

    def reset(self, env_config: BattlesnakeEnvironmentConfiguration) -> tuple[dict, dict]:
        observations, infos = self._reset(env_config)
        observations = {agent: self.observation_transformer.transform(obs) for agent, obs in observations.items()}
        return observations, infos
    
    def step(self, actions) -> tuple[dict, dict, dict, dict]:
        observations, rewards, terminations, infos = self._step(actions)
        observations = {agent: self.observation_transformer.transform(obs) for agent, obs in observations.items()}
        return observations, rewards, terminations, infos

    @property
    def action_space(self):
        return gymnasium.spaces.Discrete(len(self.engine.Movement))

    @property
    def observation_space(self):
        return self.observation_transformer.space


_ENGINE_TO_PARALLEL_ENV_ADAPTERS = {}


def _parallel_env_adapter_for(engine_type):
    def decorator(cls):
        _ENGINE_TO_PARALLEL_ENV_ADAPTERS[engine_type] = cls
        return cls
    return decorator


@_parallel_env_adapter_for(BattlesnakeDllEngine)
class BattlesnakeDllEngineForParallelEnv(BattlesnakeEngineForParallelEnv):
    
        def __init__(
            self,
            engine: BattlesnakeDllEngine,
            observation_transformer: ObservationTransformer,
        ):
            self._engine = engine
            self._observation_transformer = observation_transformer
        
        @property
        def engine(self) -> BattlesnakeEngine:
            return self._engine
        
        @property
        def observation_transformer(self) -> ObservationTransformer:
            return self._observation_transformer

        def render(self):
            self._engine.render()

        def _reset(self, env_config: BattlesnakeEnvironmentConfiguration) -> tuple[dict, dict]:
            engine_config = {
                "width": env_config.width,
                "height": env_config.height,
                "map": env_config.game_map,
                "game_type": env_config.game_type,
                "seed": env_config.seed,
                "names": env_config.possible_agents,
                "colors": env_config.colors,
            }
            response = self._engine.reset(engine_config)
            observations = {agent: response[agent]["observation"] for agent in env_config.possible_agents}
            infos = {agent: info if (info := response[agent]["info"]) else {} for agent in env_config.possible_agents}
            return observations, infos
        
        def _step(self, actions) -> tuple[dict, dict, dict, dict]:
            agents = list(actions.keys())
            response = self._engine.step(actions)
            observations = {agent: response[agent]["observation"] for agent in agents}
            rewards = {agent: response[agent]["reward"] for agent in agents}
            terminations = {agent: response[agent]["done"] for agent in agents}
            infos = {agent: info if (info := response[agent]["info"]) else {} for agent in agents}
            return observations, rewards, terminations, infos
        
        def is_game_over(self) -> bool:
            return self._engine.done()


def wrap_engine_for_parallel_env(
    engine: BattlesnakeEngine,
    observation_transformer: ObservationTransformer,
) -> BattlesnakeEngineForParallelEnv:
    return _ENGINE_TO_PARALLEL_ENV_ADAPTERS[type(engine)](engine, observation_transformer)
