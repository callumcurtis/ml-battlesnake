import abc

from environment.engines import BattlesnakeEngine, BattlesnakeDllEngine, Movement
from environment.configuration import BattlesnakeEnvironmentConfiguration


class BattlesnakeEngineForParallelEnv(abc.ABC):

    @abc.abstractmethod
    def render(self):
        pass

    @abc.abstractmethod
    def is_game_over(self) -> bool:
        pass

    @abc.abstractmethod
    def reset(self, env_config: BattlesnakeEnvironmentConfiguration) -> tuple[dict, dict]:
        pass

    @abc.abstractmethod
    def step(self, actions) -> tuple[dict, dict, dict, dict]:
        pass


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
        ):
            self._engine = engine

        def render(self):
            self._engine.render()

        def reset(self, env_config: BattlesnakeEnvironmentConfiguration) -> tuple[dict, dict]:
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
        
        def step(self, actions) -> tuple[dict, dict, dict, dict]:
            previously_alive_agents = self._engine.active_snakes()
            actions = {agent: Movement(action) for agent, action in actions.items()}
            response = self._engine.step(actions)
            observations = {agent: response[agent]["observation"] for agent in previously_alive_agents}
            rewards = {agent: response[agent]["reward"] for agent in previously_alive_agents}
            terminations = {agent: response[agent]["done"] for agent in previously_alive_agents}
            infos = {agent: info if (info := response[agent]["info"]) else {} for agent in previously_alive_agents}
            return observations, rewards, terminations, infos

        def is_game_over(self) -> bool:
            return self._engine.done()


def adapt_engine_for_parallel_env(
    engine: BattlesnakeEngine,
) -> BattlesnakeEngineForParallelEnv:
    return _ENGINE_TO_PARALLEL_ENV_ADAPTERS[type(engine)](engine)
