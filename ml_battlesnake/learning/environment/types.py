import enum
from typing import Union


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
    def length(self) -> int:
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
        configuration: BattlesnakeEnvironmentConfiguration,
        observations: dict[str, Observation],
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
    
    def with_configuration(self, configuration: BattlesnakeEnvironmentConfiguration) -> 'InitialStateBuilder':
        self.configuration = configuration
        return self

    def with_observations(self, observations: Union[dict, Observation]) -> 'InitialStateBuilder':
        if isinstance(observations, dict):
            observations = {
                agent: Observation.from_raw_observation(raw_observation)
                for agent, raw_observation in observations.items()
            }
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

    def with_actions(self, actions: dict[str, int]) -> 'TimestepBuilder':
        self.actions = actions
        return self

    def with_observations(self, observations: Union[dict, Observation]) -> 'TimestepBuilder':
        if isinstance(observations, dict):
            observations = {
                agent: Observation.from_raw_observation(raw_observation)
                for agent, raw_observation in observations.items()
            }
        self.observations = observations
        return self

    def with_rewards(self, rewards: dict[str, float]) -> 'TimestepBuilder':
        self.rewards = rewards
        return self

    def with_terminations(self, terminations: dict[str, bool]) -> 'TimestepBuilder':
        self.terminations = terminations
        return self

    def with_truncations(self, truncations: dict[str, bool]) -> 'TimestepBuilder':
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
