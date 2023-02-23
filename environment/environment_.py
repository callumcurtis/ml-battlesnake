import functools
import abc

from environment.rules import Rules
from environment.constants import DEFAULT_COLORS
from common.model import Snake

import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv
from pettingzoo.test import parallel_api_test
from pettingzoo.utils import parallel_to_aec, wrappers


def env(render_mode=None):
    internal_render_mode = render_mode if render_mode != "ansi" else "human"
    env = raw_env(render_mode=internal_render_mode)
    if render_mode == "ansi":
        env = wrappers.CaptureStdoutWrapper(env)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env


def raw_env(render_mode=None):
    # To support the AEC API, the raw_env() function just uses the from_parallel
    # function to convert from a ParallelEnv to an AEC env
    env = BattlesnakeEnvironment(
        TrainingEngine(Rules("/workspaces/rl-battlesnake/bin/rules.dll")),
        BattlesnakeEnvironmentConfiguration([Snake("agent_0", "foobar"), Snake("agent_1", "foobar")]))
    env = parallel_to_aec(env)
    return env


class BattlesnakeEnvironmentConfiguration:
    # TODO move to own file
    pass
    # agents

    def __init__(
        self,
        snakes: list[Snake],
        map: str = "standard",
        game_type: str = "standard",
        seed: int = None,
        width: int = 11,
        height: int = 11,
        colors: list[str] = DEFAULT_COLORS,
        max_turn: int = 2000,
    ) -> None:
        self.snakes = snakes
        self.map = map
        self.game_type = game_type
        self.seed = seed
        self.width = width
        self.height = height
        self.colors = colors
        self.max_turn = max_turn


class Engine(abc.ABC):

    @abc.abstractproperty
    def gamestate(self) -> dict:
        pass

    @abc.abstractmethod
    def moves(self) -> list[str]:
        pass

    @abc.abstractmethod
    def render(self) -> None:
        pass

    @abc.abstractmethod
    def reset(self, configuration: BattlesnakeEnvironmentConfiguration) -> None:
        pass

    @abc.abstractmethod
    def step(self, moves: dict[str, str]) -> None:
        pass


class TrainingEngine(Engine):

    def __init__(self, dll: Rules) -> None:
        self._dll = dll
        self._gamestate = None

    @property
    def gamestate(self) -> dict:
        """Returns the gamestate given by the most recent update"""
        assert self._gamestate is not None
        return self._gamestate

    def moves(self) -> list[str]:
        return self._dll.moves()
    
    def render(self) -> None:
        self._dll.render()

    def reset(self, configuration: BattlesnakeEnvironmentConfiguration) -> None:
        if not self._dll.isloaded():
            self._dll.load()
        options = {
            "width": configuration.width,
            "height": configuration.height,
            "map": configuration.map,
            "game_type": configuration.game_type,
            "seed": configuration.seed,
            "names": [snake.name for snake in configuration.snakes],
            "colors": configuration.colors,
        }
        self._gamestate = self._dll.reset(options)

    def step(self, moves: dict[str, str]) -> None:
        self._gamestate = self._dll.step(moves)


MAX_TEMPORARY_OBSTACLE_LIFETIME = 255
MAX_LENGTH = MAX_TEMPORARY_OBSTACLE_LIFETIME
MAX_HEALTH = 100


class BattlesnakeEnvironment(ParallelEnv):

    metadata = {"render_modes": ["human"], "name": "battlesnake"}

    def __init__(
        self,
        engine: Engine,
        configuration: BattlesnakeEnvironmentConfiguration,
        render_mode: str = None,
    ):
        self.engine = engine
        self.configuration = configuration
        self.render_mode = render_mode
        self.turn = None
        self.possible_agents = [snake.name for snake in self.configuration.snakes]
        self._action_to_move = {i: move for i, move in enumerate(self.engine.moves())}

    @functools.cache
    def observation_space(self, agent):
        dtype = np.uint8
        num_cells = self.configuration.width * self.configuration.height
        def grid(cell_value_interval):
            low, high = cell_value_interval
            assert np.iinfo(dtype).min <= low <= high <= np.iinfo(dtype).max
            return np.array([[low] * num_cells, [high] * num_cells], dtype=dtype)
        def point():
            low = 0
            high = num_cells - 1
            assert np.iinfo(dtype).min <= low <= high <= np.iinfo(dtype).max
            return np.array([[low], [high]], dtype=dtype)
        def interval(low, high):
            assert np.iinfo(dtype).min <= low <= high <= np.iinfo(dtype).max
            return np.array([[low], [high]], dtype=dtype)
        turn = interval(0, np.iinfo(dtype).max)
        food = grid([0, 1])
        your_head = point()
        your_health = interval(0, MAX_HEALTH)
        your_length = interval(0, MAX_LENGTH)
        opponents_health = grid([0, MAX_HEALTH])
        opponents_length = grid([0, MAX_LENGTH])
        temporary_obstacles = grid([0, MAX_TEMPORARY_OBSTACLE_LIFETIME])
        permanent_obstacles = grid([0, 1])
        sections = [
            turn,                   # [0]
            food,                   # [0, 121]
            your_head,              # [122]
            your_health,            # [123]
            your_length,            # [124]
            opponents_health,       # [125, 245]
            opponents_length,       # [246, 366]
            temporary_obstacles,    # [367, 487]
            permanent_obstacles,    # [488, 608]
        ]
        low, high = np.hstack(sections)
        return spaces.Box(
            low=low,
            high=high,
            dtype=dtype,
        )

    @functools.cache
    def action_space(self, agent):
        return spaces.Discrete(len(self._action_to_move))

    def render(self):
        """
        Renders the environment. In human mode, it can print to terminal, open
        up a graphical window, or open up some other display that a human can see
        and understand.
        """
        self.engine.render()

    def reset(self, seed=None, return_info=False, options=None):
        """
        Reset needs to initialize the `agents` attribute and must set up the
        environment so that render(), and step() can be called without issues.

        Returns the observations for each agent
        """
        self.agents = self.possible_agents.copy()
        self.turn = 0

        self.engine.reset(self.configuration)

        observations = self._get_observations()

        if not return_info:
            return observations
        else:
            infos = self._get_infos()
            return observations, infos

    def step(self, actions):
        """
        step(action) takes in an action for each agent and should return the
        - observations
        - rewards
        - terminations
        - truncations
        - infos
        dicts where each dict looks like {agent_1: item_1, agent_2: item_2}
        """
        # If a user passes in actions with no agents, then just return empty observations, etc.
        if not actions:
            self.agents = []
            return {}, {}, {}, {}, {}

        self.engine.step({agent: self._action_to_move[action] for agent, action in actions.items()})

        rewards = self._get_rewards()

        terminations = self._get_terminations()

        self.turn += 1

        truncate = self.turn > self.configuration.max_turn
        truncations = {agent: truncate for agent in self.agents}

        observations = self._get_observations()

        infos = self._get_infos()

        if truncate:
            self.agents = []

        self.agents = [agent for agent in self.agents if not terminations[agent]]

        if self.render_mode == "human":
            self.render()

        return observations, rewards, terminations, truncations, infos

    def _get_observations(self) -> dict:
        raw_observations = {agent: self.engine.gamestate[agent]["observation"] for agent in self.agents}
        dtype = np.uint8
        turn_scaling_factor = 1 / (self.configuration.max_turn / np.iinfo(dtype).max + 1)

        def coord_to_index(coord):
            x = coord["x"]
            y = coord["y"]
            return x + (y * self.configuration.width)

        def set_values_by_idx(grid, value_by_idx):
            if not value_by_idx:
                return
            idx, value = map(list, zip(*value_by_idx.items()))
            grid[idx] = value
        
        def snake_body_segment_lifetime_by_idx(observation):
            lifetime_by_idx = {}
            for snake in observation["board"]["snakes"]:
                snake_length = snake["length"]
                for position_in_snake, coord in enumerate(snake["body"][1:], start=1):
                    idx = coord_to_index(coord)
                    if idx in lifetime_by_idx:
                        continue
                    this_segments_lifetime = snake_length - position_in_snake - 1
                    lifetime_by_idx[idx] = this_segments_lifetime
            return lifetime_by_idx

        def resize_turn(turn):
            return turn * turn_scaling_factor

        def transform(observation):
            your_id = observation["you"]["id"]
            num_cells = self.configuration.width * self.configuration.height

            turn = np.array([resize_turn(observation["turn"])], dtype=dtype)

            food = np.zeros((num_cells,), dtype=dtype)
            food_idxs = [coord_to_index(coord) for coord in observation["board"]["food"]]
            food[food_idxs] = 1

            your_head = np.array([coord_to_index(observation["you"]["head"])], dtype=dtype)

            your_health = np.array([observation["you"]["health"]], dtype=dtype)

            your_length = np.array([observation["you"]["length"]], dtype=dtype)

            opponents_health = np.zeros((num_cells,), dtype=dtype)
            opponent_health_by_idx = {
                coord_to_index(snake["head"]): snake["health"]
                for snake in observation["board"]["snakes"]
                if snake["id"] != your_id
            }
            set_values_by_idx(opponents_health, opponent_health_by_idx)

            opponents_length = np.zeros((num_cells,), dtype=dtype)
            opponent_length_by_idx = {
                coord_to_index(snake["head"]): snake["length"]
                for snake in observation["board"]["snakes"]
                if snake["id"] != your_id
            }
            set_values_by_idx(opponents_length, opponent_length_by_idx)

            temporary_obstacles = np.zeros((num_cells,), dtype=dtype)
            temporary_obstacle_lifetime_by_idx = snake_body_segment_lifetime_by_idx(observation)
            set_values_by_idx(temporary_obstacles, temporary_obstacle_lifetime_by_idx)

            # TODO: centered board with walls
            permanent_obstacles = np.zeros((num_cells,), dtype=dtype)

            sections = [
                turn,
                food,
                your_head,
                your_health,
                your_length,
                opponents_health,
                opponents_length,
                temporary_obstacles,
                permanent_obstacles,
            ]

            return np.hstack(sections)

        return {agent: transform(obs) for agent, obs in raw_observations.items()}

    def _get_infos(self) -> dict:
        return {agent: info if (info := self.engine.gamestate[agent]["info"]) else {} for agent in self.agents}

    def _get_rewards(self) -> dict:
        # TODO: custom reward shaping
        return {agent: self.engine.gamestate[agent]["reward"] for agent in self.agents}
    
    def _get_terminations(self) -> dict:
        terminated = {agent: self.engine.gamestate[agent]["done"] for agent in self.agents}
        if sum(terminated.values()) == 1:
            terminated = {agent: True for agent in self.agents}
        return terminated


parallel_api_test(
    BattlesnakeEnvironment(
        TrainingEngine(Rules("/workspaces/rl-battlesnake/bin/rules.dll")),
        BattlesnakeEnvironmentConfiguration([Snake("agent_0", "foobar"), Snake("agent_1", "foobar")])
    ),
    num_cycles=1000)
