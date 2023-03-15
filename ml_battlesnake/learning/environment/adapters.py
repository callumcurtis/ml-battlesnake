import abc
import copy

from .engines import BattlesnakeEngine, BattlesnakeDllEngine
from .types import Movement, InitialStateBuilder, TimestepBuilder


class BattlesnakeEngineForParallelEnv(abc.ABC):

    @abc.abstractmethod
    def render(self):
        pass

    @abc.abstractmethod
    def is_game_over(self) -> bool:
        pass

    @abc.abstractmethod
    def reset(
        self,
        initial_state_builder: InitialStateBuilder,
    ) -> InitialStateBuilder:
        pass

    @abc.abstractmethod
    def step(
        self,
        timestep_builder: TimestepBuilder,
    ) -> TimestepBuilder:
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

        def reset(
            self,
            initial_state_builder: InitialStateBuilder,
        ) -> InitialStateBuilder:
            assert initial_state_builder.configuration
            initial_state_builder = copy.deepcopy(initial_state_builder)
            env_config = initial_state_builder.configuration
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
            return initial_state_builder.with_observations(observations).with_infos(infos)

        def step(
            self,
            timestep_builder: TimestepBuilder,
        ) -> TimestepBuilder:
            assert timestep_builder.actions
            timestep_builder = copy.deepcopy(timestep_builder)
            previously_alive_agents = self._engine.active_snakes()
            action_ids = {agent: Movement(action) for agent, action in timestep_builder.actions.items()}
            response = self._engine.step(action_ids)
            observations = {agent: response[agent]["observation"] for agent in previously_alive_agents}
            rewards = {agent: response[agent]["reward"] for agent in previously_alive_agents}
            terminations = {agent: response[agent]["done"] for agent in previously_alive_agents}
            infos = {agent: info if (info := response[agent]["info"]) else {} for agent in previously_alive_agents}
            return timestep_builder \
                        .with_observations(observations) \
                        .with_rewards(rewards) \
                        .with_terminations(terminations) \
                        .with_infos(infos)

        def is_game_over(self) -> bool:
            return self._engine.done()


def adapt_engine_for_parallel_env(
    engine: BattlesnakeEngine,
) -> BattlesnakeEngineForParallelEnv:
    return _ENGINE_TO_PARALLEL_ENV_ADAPTERS[type(engine)](engine)
