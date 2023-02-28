class BattlesnakeEnvironmentConfiguration:

    DEFAULT_COLORS = [
        "#00FF00",
        "#0000FF",
        "#FF00FF",
        "#FFFF00",
    ]

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


class InitialState:

    def __init__(
        self,
        configuration,
        observations,
        infos = None,
    ):
        self.configuration = configuration
        self.observations = observations
        self.infos = infos


class InitialStateBuilder:

    def __init__(self):
        self.configuration = None
        self.observations = {}
        self.infos = {}
    
    def with_configuration(self, configuration) -> 'InitialStateBuilder':
        self.configuration = configuration
        return self

    def with_observations(self, observations) -> 'InitialStateBuilder':
        self.observations = observations
        return self

    def with_infos(self, infos) -> 'InitialStateBuilder':
        self.infos = infos
        return self

    def build(self) -> InitialState:
        assert (self.observations.keys() == self.infos.keys()) and self.configuration
        return InitialState(
            configuration=self.configuration,
            observations=self.observations,
            infos=self.infos,
        )


class Timestep:

    def __init__(
        self,
        actions,
        observations,
        rewards,
        terminations,
        truncations,
        infos,
    ):
        self.actions = actions
        self.observations = observations
        self.rewards = rewards
        self.terminations = terminations
        self.truncations = truncations
        self.infos = infos


class TimestepBuilder:

    def __init__(self):
        self.actions = {}
        self.observations = {}
        self.rewards = {}
        self.terminations = {}
        self.truncations = {}
        self.infos = {}

    def with_actions(self, actions) -> 'TimestepBuilder':
        self.actions = actions
        return self

    def with_observations(self, observations) -> 'TimestepBuilder':
        self.observations = observations
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
            == self.observations.keys()
            == self.rewards.keys()
            == self.terminations.keys()
            == self.truncations.keys()
            == self.infos.keys()
        )
        return Timestep(
            actions=self.actions,
            observations=self.observations,
            rewards=self.rewards,
            terminations=self.terminations,
            truncations=self.truncations,
            infos=self.infos,
        )
