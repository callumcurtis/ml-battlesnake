"""Transformers of observations into formats suitable for neural networks."""

import abc
import functools
import enum

import numpy as np
import gymnasium

from .types import BattlesnakeEnvironmentConfiguration, Observation, SnakeObservation


class BoardEntity(enum.Enum):
    EMPTY = enum.auto()
    FOOD = enum.auto()
    NEXT_SNAKE_PART_IS_ON_TOP = enum.auto()
    NEXT_SNAKE_PART_IS_UP = enum.auto()
    NEXT_SNAKE_PART_IS_DOWN = enum.auto()
    NEXT_SNAKE_PART_IS_LEFT = enum.auto()
    NEXT_SNAKE_PART_IS_RIGHT = enum.auto()
    ENEMY_HEAD = enum.auto()
    YOUR_HEAD = enum.auto()
    WALL = enum.auto()


class BoardOrientationTransformer:
    """Transforms the battlesnake coordinate system to the natural one and vice versa.

    The battlesnake coordinate system is such that the origin is at the bottom-left corner.
    The natural coordinate system is such that the origin is at the top-left corner.
    """

    def to_natural_orientation(self, battlesnake_array: np.ndarray) -> np.ndarray:
        """Transforms the battlesnake coordinate system to the natural one."""
        return np.rot90(battlesnake_array, k=-1, axes=(1, 2))

    def to_battlesnake_orientation(self, natural_array: np.ndarray) -> np.ndarray:
        """Transforms the natural coordinate system to the battlesnake one."""
        return np.rot90(natural_array, k=1, axes=(1, 2))


class ObservationTransformer(abc.ABC):

    @abc.abstractmethod
    def transform(self, observation: Observation):
        pass

    @abc.abstractmethod
    def transform_all(self, observations: dict[str, Observation]):
        pass

    @property
    @abc.abstractmethod
    def space(self) -> gymnasium.spaces.Space:
        pass

    @abc.abstractmethod
    def empty_observation(self):
        pass


class TransformAllMixin:

    def transform_all(self, observations: dict[str, Observation]):
        return {
            agent: self.transform(obs)
            for agent, obs in observations.items()
        }


class ObservationToImage(TransformAllMixin, ObservationTransformer):

    DTYPE = np.uint8
    NUM_CHANNELS = 1

    def __init__(
        self,
        env_config: BattlesnakeEnvironmentConfiguration,
        egocentric: bool = False,
        board_orientation_transformer: BoardOrientationTransformer = BoardOrientationTransformer(),
    ):
        self._env_config = env_config
        self._egocentric = egocentric
        self._board_orientation_transformer = board_orientation_transformer

        board_shape = (self.NUM_CHANNELS, self._env_config.height, self._env_config.width)
        view_shape = (self.NUM_CHANNELS, board_shape[1] * 2 - 1, board_shape[2] * 2 - 1) if egocentric else board_shape
        self._board_shape = board_shape
        self._view_shape = view_shape

    @functools.cached_property
    def space(self) -> gymnasium.spaces.Box:
        return gymnasium.spaces.Box(
            low=np.iinfo(self.DTYPE).min,
            high=np.iinfo(self.DTYPE).max,
            shape=self._view_shape,
            dtype=self.DTYPE,
        )

    @functools.cached_property
    def possible_directions_to_next_snake_part(self) -> list[BoardEntity]:
        return [
            BoardEntity.NEXT_SNAKE_PART_IS_ON_TOP,
            BoardEntity.NEXT_SNAKE_PART_IS_UP,
            BoardEntity.NEXT_SNAKE_PART_IS_DOWN,
            BoardEntity.NEXT_SNAKE_PART_IS_LEFT,
            BoardEntity.NEXT_SNAKE_PART_IS_RIGHT,
        ]

    @functools.cached_property
    def value_by_entity(self) -> dict[BoardEntity, int]:
        value_spacing = self.space.high.item(0) // (len(BoardEntity) - 1)
        assert value_spacing > 0, "Not enough space to encode all game entities"
        return {
            entity: self.space.low.item(0) + (i * value_spacing)
            for i, entity in enumerate(BoardEntity)
        }

    def transform(self, observation: Observation):
        """Transforms the observation into an image.

        The image is a 2D array of integers, where each integer
        represents a different entity on the board. The integers
        representing game entities are spread evenly across the range
        of the dtype.

        Snakes are encoded using a linked-list representation where
        the integer value of a snake part encodes the direction to
        the next snake part (ordered from tail to head).
        """
        # TODO: instead, delegate to the binary matrices transformer and squash the results

        coords_by_entity = {
            BoardEntity.FOOD: list((food.x, food.y) for food in observation.food),
        }

        def get_coords_by_direction_to_next_snake_part(snakes: list[SnakeObservation]):
            coords_by_direction_to_next_snake_part = {d: [] for d in self.possible_directions_to_next_snake_part}
            for snake in snakes:
                for snake_part_coord, next_snake_part_coord in zip(snake.body[1:], snake.body):
                    snake_part_coord = (snake_part_coord.x, snake_part_coord.y)
                    next_snake_part_coord = (next_snake_part_coord.x, next_snake_part_coord.y)
                    coord_delta = (
                        next_snake_part_coord[0] - snake_part_coord[0],
                        next_snake_part_coord[1] - snake_part_coord[1],
                    )
                    if coord_delta == (0, 1):
                        direction_to_next_snake_part = BoardEntity.NEXT_SNAKE_PART_IS_UP
                    elif coord_delta == (0, -1):
                        direction_to_next_snake_part = BoardEntity.NEXT_SNAKE_PART_IS_DOWN
                    elif coord_delta == (1, 0):
                        direction_to_next_snake_part = BoardEntity.NEXT_SNAKE_PART_IS_RIGHT
                    elif coord_delta == (-1, 0):
                        direction_to_next_snake_part = BoardEntity.NEXT_SNAKE_PART_IS_LEFT
                    elif coord_delta == (0, 0):
                        direction_to_next_snake_part = BoardEntity.NEXT_SNAKE_PART_IS_ON_TOP
                    else:
                        raise ValueError(f"Unexpected coord delta: {coord_delta}")
                    coords_by_direction_to_next_snake_part[direction_to_next_snake_part].append(snake_part_coord)
            return coords_by_direction_to_next_snake_part

        for direction_to_next_snake_part, coords in get_coords_by_direction_to_next_snake_part(observation.snakes).items():
            coords_by_entity[direction_to_next_snake_part] = tuple(coords)

        # remove the "on top" game entities as otherwise they will overwrite the more important direction entities
        coords_by_entity.pop(BoardEntity.NEXT_SNAKE_PART_IS_ON_TOP, None)

        for snake in observation.snakes:
            head_class = BoardEntity.YOUR_HEAD if snake.id == observation.you.id else BoardEntity.ENEMY_HEAD
            coords_by_entity.setdefault(head_class, []).append((snake.head.x, snake.head.y))

        board_array = np.zeros(self._board_shape, dtype=self.DTYPE)
        assert board_array.shape[0] == 1, "Only one channel is currently supported"
        for entity, coords in coords_by_entity.items():
            if coords:
                board_array[0][tuple(zip(*coords))] = self.value_by_entity[entity]

        if self._egocentric:
            # place the board array within a larger array representing the egocentric view
            view_array = np.full(self._view_shape, self.value_by_entity[BoardEntity.WALL], dtype=self.DTYPE)
            x0 = self._view_shape[1] - self._board_shape[1] - observation.you.head.x
            x1 = x0 + self._board_shape[1]
            y0 = self._view_shape[2] - self._board_shape[2] - observation.you.head.y
            y1 = y0 + self._board_shape[2]
            view_array[0][x0:x1, y0:y1] = board_array[0]
        else:
            view_array = board_array

        view_array = self._board_orientation_transformer.to_battlesnake_orientation(view_array)

        return view_array

    def empty_observation(self):
        return np.zeros(self.space.shape, dtype=self.DTYPE)


class ObservationToFlattenedArray(TransformAllMixin, ObservationTransformer):

    DTYPE = np.uint8

    def __init__(
        self,
        env_config: BattlesnakeEnvironmentConfiguration,
        egocentric: bool = True,
        include_your_health: bool = True,
        include_enemy_health: bool = True,
    ):
        assert (
            not (include_your_health or include_enemy_health)
            or env_config.MAX_HEALTH <= np.iinfo(self.DTYPE).max
        ), "Not a large enough dtype to encode health"
        assert (
            env_config.width * env_config.height < np.iinfo(self.DTYPE).max
        ), "Not a large enough dtype to encode scalar coordinates for board"
        assert (
            not include_enemy_health
            or env_config.width * env_config.height < np.iinfo(self.DTYPE).max - 1
        ), "Not a large enough dtype to encode scalar coordinates for enemies, with one extra value for 'no enemy'"

        self._env_config = env_config
        self._egocentric = egocentric
        self._include_your_health = include_your_health
        self._include_enemy_health = include_enemy_health
        size = env_config.width * env_config.height
        if include_your_health:
            size += 1
        if include_enemy_health:
            size += (len(env_config.possible_agents) - 1) * 2
        self._shape = (size,)
        self._to_image = ObservationToImage(env_config, egocentric=False)
    
    @functools.cached_property
    def space(self) -> gymnasium.spaces.Box:
        return gymnasium.spaces.Box(
            low=np.iinfo(self.DTYPE).min,
            high=np.iinfo(self.DTYPE).max,
            shape=self._shape,
            dtype=self.DTYPE,
        )
    
    def transform(self, observation: Observation):
        """Transforms the observation into a flattened image array."""
        image = self._to_image.transform(observation)
        board = image.reshape(self._env_config.height, self._env_config.width)
        coord_to_scalar = lambda coord: (abs(coord.y) * self._env_config.width) + abs(coord.x)
        if self._egocentric:
            # shift the board so that your head is at the battlesnake api origin
            shift = (observation.you.head.y, -observation.you.head.x)
            shift_scalar = coord_to_scalar(observation.you.head)
            board = np.roll(board, shift, axis=(0, 1))
            # encode the shift in your head position as it is fixed for each observation
            board[-1, 0] = shift_scalar
        flat_array = board.flatten()
        health_info = []
        if self._include_your_health:
            health_info.append(observation.you.health)
        if self._include_enemy_health:
            enemy_snakes = [
                snake
                for snake in observation.snakes
                if snake.id != observation.you.id
            ]
            for enemy_snake in enemy_snakes:
                health_info.append(coord_to_scalar(enemy_snake.head))
                health_info.append(enemy_snake.health)
            num_missing_enemy_snakes = len(self._env_config.possible_agents) - len(enemy_snakes) - 1
            health_info.extend([np.iinfo(self.DTYPE).max, 0] * num_missing_enemy_snakes)
        if health_info:
            health_info = np.array(health_info, dtype=self.DTYPE)
            flat_array = np.concatenate((flat_array, health_info))
        return flat_array

    def empty_observation(self):
        return np.zeros(self.space.shape, dtype=self.DTYPE)


class ObservationToBinaryMatrices(TransformAllMixin, ObservationTransformer):

    DTYPE = np.ubyte

    def __init__(
        self,
        observation_to_image: ObservationToImage,
    ):
        assert(observation_to_image.NUM_CHANNELS == 1, "Only one input channel is currently supported")
        self._shape = (
            len(BoardEntity),
            *observation_to_image.space.shape[1:],
        )
        self._to_image = observation_to_image

    @functools.cached_property
    def space(self) -> gymnasium.spaces.Box:
        return gymnasium.spaces.Box(
            low=0,
            high=1,
            shape=self._shape,
            dtype=self.DTYPE,
        )

    def transform(self, observation: Observation) -> np.ndarray:
        image = self._to_image.transform(observation)
        binary_matrices = np.zeros(self.space.shape, dtype=self.DTYPE)
        for i, entity in enumerate(BoardEntity):
            binary_matrices[i] = image[0] == self._to_image.value_by_entity[entity]
        return binary_matrices

    def empty_observation(self):
        return np.zeros(self.space.shape, dtype=self.DTYPE)
