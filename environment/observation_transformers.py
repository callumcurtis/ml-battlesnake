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
        env_config: BattlesnakeEnvironmentConfiguration
    ):
        self._env_config = env_config

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
        static_elements = [
            "empty", 
            "food",
            "your_body",
            "your_head",
        ]
        your_id = observation["you"]["id"]
        all_enemies = list(filter(lambda s: s != your_id, self._env_config.possible_agents))
        dynamic_elements = [f"enemy_{i}_{part}" for i in range(len(all_enemies)) for part in ["body", "head"]]
        all_elements = static_elements + dynamic_elements
        encoded_value_spacing = self.space.high.item(0) // (len(all_elements) - 1)
        assert encoded_value_spacing > 0, "Not enough space to encode all elements"
        encoding_by_element = {element: self.space.low.item(0) + (i * encoded_value_spacing) for i, element in enumerate(all_elements)}

        element_coords = {
            "food": tuple((food["x"], food["y"]) for food in observation["board"]["food"]),
            "your_body": tuple((body["x"], body["y"]) for body in observation["you"]["body"]),
            "your_head": ((observation["you"]["head"]["x"], observation["you"]["head"]["y"]),),
        }
        enemies = list(filter(lambda s: s["id"] in all_enemies, observation["board"]["snakes"]))
        for i, enemy in enumerate(enemies):
            element_coords[f"enemy_{i}_body"] = tuple((body["x"], body["y"]) for body in enemy["body"])
            element_coords[f"enemy_{i}_head"] = ((enemy["head"]["x"], enemy["head"]["y"]),)

        array = np.zeros(self.space.shape, dtype=self.DTYPE)
        assert array.shape[0] == 1, "Only one channel is currently supported"
        for element, coords in element_coords.items():
            array[0][tuple(zip(*coords))] = encoding_by_element[element]
        # match the orientation of the game board by moving origin to bottom left
        array[0] = np.rot90(array[0])

        return array

    def empty_observation(self):
        return np.zeros(self.space.shape, dtype=self.DTYPE)
