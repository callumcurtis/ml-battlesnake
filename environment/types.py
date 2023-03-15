import enum


class BattlesnakeEnvironmentConfiguration:

    DEFAULT_COLORS = [
        "#00FF00",
        "#0000FF",
        "#FF00FF",
        "#FFFF00",
    ]

    MAX_HEALTH = 100

    def __init__(
        self,
        possible_agents: list[str],
        render_mode = "human",
        width = 11,
        height = 11,
        colors = DEFAULT_COLORS,
        game_map = "standard",
        game_type = "standard",
        seed: int = None,
    ) -> None:
        self.possible_agents = possible_agents
        self.render_mode = render_mode
        self.width = width
        self.height = height
        self.colors = colors
        self.game_map = game_map
        self.game_type = game_type
        self.seed = seed


class Movement(enum.IntEnum):
    UP = 0
    DOWN = enum.auto()
    LEFT = enum.auto()
    RIGHT = enum.auto()


class InitialState:

    def __init__(
        self,
        configuration,
        raw_observations,
        infos = None,
    ):
        self.configuration = configuration
        self.raw_observations = raw_observations
        self.infos = infos


class InitialStateBuilder:

    def __init__(self):
        self.configuration = None
        self.raw_observations = {}
        self.infos = {}
    
    def with_configuration(self, configuration) -> 'InitialStateBuilder':
        self.configuration = configuration
        return self

    def with_raw_observations(self, raw_observations) -> 'InitialStateBuilder':
        self.raw_observations = raw_observations
        return self

    def with_infos(self, infos) -> 'InitialStateBuilder':
        self.infos = infos
        return self

    def build(self) -> InitialState:
        assert (self.raw_observations.keys() == self.infos.keys()) and self.configuration
        return InitialState(
            configuration=self.configuration,
            raw_observations=self.raw_observations,
            infos=self.infos,
        )


class Timestep:

    def __init__(
        self,
        actions,
        raw_observations,
        rewards,
        terminations,
        truncations,
        infos,
    ):
        self.actions = actions
        self.raw_observations = raw_observations
        self.rewards = rewards
        self.terminations = terminations
        self.truncations = truncations
        self.infos = infos


class TimestepBuilder:

    def __init__(self):
        self.actions = {}
        self.raw_observations = {}
        self.rewards = {}
        self.terminations = {}
        self.truncations = {}
        self.infos = {}

    def with_actions(self, actions) -> 'TimestepBuilder':
        self.actions = actions
        return self

    def with_raw_observations(self, raw_observations) -> 'TimestepBuilder':
        self.raw_observations = raw_observations
        return self

    def with_rewards(self, rewards) -> 'TimestepBuilder':
        self.rewards = rewards
        return self

    def with_terminations(self, terminations) -> 'TimestepBuilder':
        self.terminations = terminations
        return self

    def with_truncations(self, truncations) -> 'TimestepBuilder':
        self.truncations = truncations
        return self

    def with_infos(self, infos) -> 'TimestepBuilder':
        self.infos = infos
        return self

    def build(self) -> Timestep:
        assert (
            self.actions.keys()
            == self.raw_observations.keys()
            == self.rewards.keys()
            == self.terminations.keys()
            == self.truncations.keys()
            == self.infos.keys()
        )
        return Timestep(
            actions=self.actions,
            raw_observations=self.raw_observations,
            rewards=self.rewards,
            terminations=self.terminations,
            truncations=self.truncations,
            infos=self.infos,
        )
