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


class Coordinates:
    
        def __init__(
            self,
            x: int,
            y: int,
        ):
            self.x = x
            self.y = y
        
        @classmethod
        def from_dict(cls, coordinates: dict) -> 'Coordinates':
            return cls(
                x=coordinates["x"],
                y=coordinates["y"],
            )

        def __eq__(self, other):
            return self.x == other.x and self.y == other.y


class SnakeObservation:

    def __init__(
        self,
        id: str,
        name: str,
        health: int,
        body: list[Coordinates],
        head: Coordinates,
    ):
        self.id = id
        self.name = name
        self.health = health
        self.body = body
        self.head = head
    
    @property
    def length(self):
        return len(self.body)
    
    @classmethod
    def from_raw_observation(cls, raw_observation: dict) -> list['SnakeObservation']:
        return [
            cls.from_raw_snake_observation(raw_snake_observation)
            for raw_snake_observation in raw_observation["board"]["snakes"]
        ]
    
    @classmethod
    def from_raw_snake_observation(cls, raw_snake_observation: dict) -> 'SnakeObservation':
        id = raw_snake_observation["id"]
        name = raw_snake_observation["name"]
        health = raw_snake_observation["health"]
        body = [Coordinates.from_dict(body) for body in raw_snake_observation["body"]]
        head = Coordinates.from_dict(raw_snake_observation["head"])
        return cls(
            id=id,
            name=name,
            health=health,
            body=body,
            head=head,
        )


class Observation:

    def __init__(
        self,
        turn: int,
        snakes: list[SnakeObservation],
        food: list[Coordinates],
        you: SnakeObservation,
    ):
        self.turn = turn
        self.snakes = snakes
        self.food = food
        self.you = you
    
    @classmethod
    def from_raw_observation(cls, raw_observation: dict) -> 'Observation':
        turn = raw_observation["turn"]
        snakes = SnakeObservation.from_raw_observation(raw_observation)
        food = [Coordinates.from_dict(food) for food in raw_observation["board"]["food"]]
        you = next((snake for snake in snakes if snake.id == raw_observation["you"]["id"]), None)
        if not you:
            you = SnakeObservation.from_raw_snake_observation(raw_observation["you"])
        return cls(
            turn=turn,
            snakes=snakes,
            food=food,
            you=you,
        )


class InitialState:

    def __init__(
        self,
        configuration,
        raw_observations,
        observations,
        infos = None,
    ):
        self.configuration = configuration
        self.raw_observations = raw_observations
        self.observations = observations
        self.infos = infos


class InitialStateBuilder:

    def __init__(self):
        self.configuration = None
        self.raw_observations = {}
        self.observations = {}
        self.infos = {}
    
    def with_configuration(self, configuration) -> 'InitialStateBuilder':
        self.configuration = configuration
        return self

    def with_raw_observations(self, raw_observations) -> 'InitialStateBuilder':
        self.raw_observations = raw_observations
        self.observations = {
            agent: Observation.from_raw_observation(raw_observation)
            for agent, raw_observation in self.raw_observations.items()
        }
        return self

    def with_infos(self, infos) -> 'InitialStateBuilder':
        self.infos = infos
        return self

    def build(self) -> InitialState:
        assert (
            (
                self.raw_observations.keys()
                == self.observations.keys()
                == self.infos.keys()
            ) and self.configuration)
        return InitialState(
            configuration=self.configuration,
            raw_observations=self.raw_observations,
            observations=self.observations,
            infos=self.infos,
        )


class Timestep:

    def __init__(
        self,
        actions,
        raw_observations,
        observations,
        rewards,
        terminations,
        truncations,
        infos,
    ):
        self.actions = actions
        self.raw_observations = raw_observations
        self.observations = observations
        self.rewards = rewards
        self.terminations = terminations
        self.truncations = truncations
        self.infos = infos


class TimestepBuilder:

    def __init__(self):
        self.actions = {}
        self.raw_observations = {}
        self.observations = {}
        self.rewards = {}
        self.terminations = {}
        self.truncations = {}
        self.infos = {}

    def with_actions(self, actions) -> 'TimestepBuilder':
        self.actions = actions
        return self

    def with_raw_observations(self, raw_observations) -> 'TimestepBuilder':
        self.raw_observations = raw_observations
        self.observations = {
            agent: Observation.from_raw_observation(raw_observation)
            for agent, raw_observation in self.raw_observations.items()
        }
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
            == self.observations.keys()
            == self.rewards.keys()
            == self.terminations.keys()
            == self.truncations.keys()
            == self.infos.keys()
        )
        return Timestep(
            actions=self.actions,
            raw_observations=self.raw_observations,
            observations=self.observations,
            rewards=self.rewards,
            terminations=self.terminations,
            truncations=self.truncations,
            infos=self.infos,
        )
