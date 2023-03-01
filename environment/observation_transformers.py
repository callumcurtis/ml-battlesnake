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

    @functools.cached_property
    def space(self):
        shape = (self.NUM_CHANNELS, self._env_config.height, self._env_config.width)
        return gymnasium.spaces.Box(
            low=np.iinfo(self.DTYPE).min,
            high=np.iinfo(self.DTYPE).max,
            shape=shape,
            dtype=self.DTYPE,
        )

    def transform(self, observation):
        """
        Transforms the observation into an image.

        The image is a 2D array of integers, where each integer
        represents a different element on the board. The integers
        representing elements are spread evenly across the range
        of the dtype. The number of elements, and thus the spacing
        between integers, depends on the number of snakes.

        Snakes are encoded using a linked-list representation where
        the integer value of a snake part encodes the direction to
        the next snake part (ordered from tail to head).
        """
        your_id = observation["you"]["id"]
        all_enemy_ids = list(filter(lambda s: s != your_id, self._env_config.possible_agents))
        possible_directions_to_next_snake_part = [
            "next_is_on_top",
            "next_is_up",
            "next_is_down",
            "next_is_left",
            "next_is_right",
            "head",
        ]
        elements = [
            "empty",  # always 0
            "food",
            *[
                f"you_{direction_to_next_snake_part}"
                for direction_to_next_snake_part in possible_directions_to_next_snake_part
            ],
            *[
                f"{enemy_id}_{direction_to_next_snake_part}"
                for enemy_id in all_enemy_ids
                for direction_to_next_snake_part in possible_directions_to_next_snake_part
            ],
        ]
        encoded_value_spacing = self.space.high.item(0) // (len(elements) - 1)
        assert encoded_value_spacing > 0, "Not enough space to encode all elements"
        encoding_by_element = {element: self.space.low.item(0) + (i * encoded_value_spacing) for i, element in enumerate(elements)}

        coords_by_element = {
            "food": tuple((food["x"], food["y"]) for food in observation["board"]["food"]),
        }

        def get_coords_by_direction_to_next_snake_part(snake_dict):
            coords_by_direction_to_next_snake_part = {d: [] for d in possible_directions_to_next_snake_part}
            coords_by_direction_to_next_snake_part["head"] = [(snake_dict["head"]["x"], snake_dict["head"]["y"])]
            for snake_part_coord, next_snake_part_coord in zip(snake_dict["body"][1:], snake_dict["body"]):
                snake_part_coord = (snake_part_coord["x"], snake_part_coord["y"])
                next_snake_part_coord = (next_snake_part_coord["x"], next_snake_part_coord["y"])
                coord_delta = (
                    next_snake_part_coord[0] - snake_part_coord[0],
                    next_snake_part_coord[1] - snake_part_coord[1],
                )
                if coord_delta == (0, 1):
                    direction_to_next_snake_part = "next_is_up"
                elif coord_delta == (0, -1):
                    direction_to_next_snake_part = "next_is_down"
                elif coord_delta == (1, 0):
                    direction_to_next_snake_part = "next_is_right"
                elif coord_delta == (-1, 0):
                    direction_to_next_snake_part = "next_is_left"
                elif coord_delta == (0, 0):
                    direction_to_next_snake_part = "next_is_on_top"
                else:
                    raise ValueError(f"Unexpected coord delta: {coord_delta}")
                coords_by_direction_to_next_snake_part[direction_to_next_snake_part].append(snake_part_coord)
            return coords_by_direction_to_next_snake_part
        
        for snake_dict in observation["board"]["snakes"]:
            snake_id = "you" if snake_dict["id"] == your_id else snake_dict["id"]
            for direction_to_next_snake_part, coords in get_coords_by_direction_to_next_snake_part(snake_dict).items():
                coords_by_element[f"{snake_id}_{direction_to_next_snake_part}"] = tuple(coords)

        array = np.zeros(self.space.shape, dtype=self.DTYPE)
        assert array.shape[0] == 1, "Only one channel is currently supported"
        for element, coords in coords_by_element.items():
            if coords:
                array[0][tuple(zip(*coords))] = encoding_by_element[element]

        if self._egocentric:
            # move your head to numpy origin (top left)
            your_head_coord = (observation["you"]["head"]["x"], observation["you"]["head"]["y"])
            shift = (-your_head_coord[0], -your_head_coord[1])
            array = np.roll(array, shift, axis=(1, 2))
            shift_scalar = -((shift[0] * self._env_config.width) + shift[1])
            # encode the shift in the head position as the head no longer needs an integer encoding
            # as it is fixed to the origin
            array.put(0, shift_scalar)
        else:
            # match the battlesnake api orientation by moving origin to bottom left
            array[0] = np.rot90(array[0])

        return array

    def empty_observation(self):
        return np.zeros(self.space.shape, dtype=self.DTYPE)
