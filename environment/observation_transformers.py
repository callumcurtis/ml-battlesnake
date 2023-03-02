import abc
import functools

import numpy as np
import gymnasium

from environment.types import BattlesnakeEnvironmentConfiguration


class ObservationTransformer(abc.ABC):

    @abc.abstractmethod
    def transform(self, observation):
        pass

    @property
    @abc.abstractmethod
    def space(self):
        pass

    @abc.abstractmethod
    def empty_observation(self):
        pass


class ObservationToImage(ObservationTransformer):

    DTYPE = np.uint8
    NUM_CHANNELS = 1

    def __init__(
        self,
        env_config: BattlesnakeEnvironmentConfiguration,
        egocentric: bool = False,
    ):
        self._env_config = env_config
        self._egocentric = egocentric

        board_shape = (self.NUM_CHANNELS, self._env_config.height, self._env_config.width)
        view_shape = (self.NUM_CHANNELS, board_shape[1] * 2 - 1, board_shape[2] * 2 - 1) if egocentric else board_shape
        self._board_shape = board_shape
        self._view_shape = view_shape

    @functools.cached_property
    def space(self):
        return gymnasium.spaces.Box(
            low=np.iinfo(self.DTYPE).min,
            high=np.iinfo(self.DTYPE).max,
            shape=self._view_shape,
            dtype=self.DTYPE,
        )

    def transform(self, observation):
        """
        Transforms the observation into an image.

        The image is a 2D array of integers, where each integer
        represents a different element on the board. The integers
        representing elements are spread evenly across the range
        of the dtype.

        Snakes are encoded using a linked-list representation where
        the integer value of a snake part encodes the direction to
        the next snake part (ordered from tail to head).
        """
        possible_directions_to_next_snake_part = [
            "next_snake_part_is_on_top",
            "next_snake_part_is_up",
            "next_snake_part_is_down",
            "next_snake_part_is_left",
            "next_snake_part_is_right",
        ]
        elements = [
            "empty",  # always 0
            "food",
            *possible_directions_to_next_snake_part,
            "enemy_head",
            "your_head",
            "wall",
        ]
        encoded_value_spacing = self.space.high.item(0) // (len(elements) - 1)
        assert encoded_value_spacing > 0, "Not enough space to encode all elements"
        encoding_by_element = {element: self.space.low.item(0) + (i * encoded_value_spacing) for i, element in enumerate(elements)}

        coords_by_element = {
            "food": tuple((food["x"], food["y"]) for food in observation["board"]["food"]),
        }

        def get_coords_by_direction_to_next_snake_part(snake_dicts):
            coords_by_direction_to_next_snake_part = {d: [] for d in possible_directions_to_next_snake_part}
            for snake_dict in snake_dicts:
                for snake_part_coord, next_snake_part_coord in zip(snake_dict["body"][1:], snake_dict["body"]):
                    snake_part_coord = (snake_part_coord["x"], snake_part_coord["y"])
                    next_snake_part_coord = (next_snake_part_coord["x"], next_snake_part_coord["y"])
                    coord_delta = (
                        next_snake_part_coord[0] - snake_part_coord[0],
                        next_snake_part_coord[1] - snake_part_coord[1],
                    )
                    if coord_delta == (0, 1):
                        direction_to_next_snake_part = "next_snake_part_is_up"
                    elif coord_delta == (0, -1):
                        direction_to_next_snake_part = "next_snake_part_is_down"
                    elif coord_delta == (1, 0):
                        direction_to_next_snake_part = "next_snake_part_is_right"
                    elif coord_delta == (-1, 0):
                        direction_to_next_snake_part = "next_snake_part_is_left"
                    elif coord_delta == (0, 0):
                        direction_to_next_snake_part = "next_snake_part_is_on_top"
                    else:
                        raise ValueError(f"Unexpected coord delta: {coord_delta}")
                    coords_by_direction_to_next_snake_part[direction_to_next_snake_part].append(snake_part_coord)
            return coords_by_direction_to_next_snake_part

        for direction_to_next_snake_part, coords in get_coords_by_direction_to_next_snake_part(observation["board"]["snakes"]).items():
            coords_by_element[direction_to_next_snake_part] = tuple(coords)

        your_id = observation["you"]["id"]
        for snake_dict in observation["board"]["snakes"]:
            snake_pronoun = "your" if snake_dict["id"] == your_id else "enemy"
            coords_by_element[f"{snake_pronoun}_head"] = ((snake_dict["head"]["x"], snake_dict["head"]["y"]),)

        board_array = np.zeros(self._board_shape, dtype=self.DTYPE)
        assert board_array.shape[0] == 1, "Only one channel is currently supported"
        for element, coords in coords_by_element.items():
            if coords:
                board_array[0][tuple(zip(*coords))] = encoding_by_element[element]

        if self._egocentric:
            # place the board array within a larger array representing the egocentric view
            view_array = np.full(self._view_shape, encoding_by_element["wall"], dtype=self.DTYPE)
            x0 = self._view_shape[1] - self._board_shape[1] - observation["you"]["head"]["x"]
            x1 = x0 + self._board_shape[1]
            y0 = self._view_shape[2] - self._board_shape[2] - observation["you"]["head"]["y"]
            y1 = y0 + self._board_shape[2]
            view_array[0][x0:x1, y0:y1] = board_array[0]
        else:
            view_array = board_array

        # match the battlesnake api orientation by moving origin to bottom left
        view_array[0] = np.rot90(view_array[0])

        return view_array

    def empty_observation(self):
        return np.zeros(self.space.shape, dtype=self.DTYPE)


class ObservationToFlattenedArray(ObservationTransformer):

    DTYPE = np.uint8

    def __init__(
        self,
        env_config: BattlesnakeEnvironmentConfiguration,
        egocentric: bool = True,
    ):
        self._env_config = env_config
        self._egocentric = egocentric
        self._shape = (self._env_config.width * self._env_config.height,)
        self._to_image = ObservationToImage(env_config, egocentric=False)
    
    @functools.cached_property
    def space(self):
        return gymnasium.spaces.Box(
            low=np.iinfo(self.DTYPE).min,
            high=np.iinfo(self.DTYPE).max,
            shape=self._shape,
            dtype=self.DTYPE,
        )
    
    def transform(self, observation):
        image = self._to_image.transform(observation)
        board = image.reshape(self._env_config.height, self._env_config.width)
        if self._egocentric:
            # shift the board so that your head is at the battlesnake api origin
            shift = (observation["you"]["head"]["y"], -observation["you"]["head"]["x"])
            shift_scalar = (abs(shift[0]) * self._env_config.width) + abs(shift[1])
            board = np.roll(board, shift, axis=(0, 1))
            # encode the shift in your head position as it is fixed for each observation
            board[-1, 0] = shift_scalar
        return board.flatten()

    def empty_observation(self):
        return np.zeros(self.space.shape, dtype=self.DTYPE)
