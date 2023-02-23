import abc

import numpy as np
import gymnasium

from environment.configuration import BattlesnakeEnvironmentConfiguration


class ObservationTransformer(abc.ABC):

    @abc.abstractmethod
    def transform(self, observation):
        pass

    @property
    @abc.abstractmethod
    def space(self):
        pass


class ObservationToArray(ObservationTransformer):

    def __init__(
        self,
        env_config: BattlesnakeEnvironmentConfiguration
    ):
        shape = (1, env_config.height, env_config.width)
        self._space = gymnasium.spaces.Box(low=0, high=1, shape=shape, dtype=np.float16)

    @property
    def space(self):
        return self._space

    def transform(self, observation):
        elements = [
            "empty", 
            "food",
            "your_head",
            "your_body",
        ]
        your_id = observation["you"]["id"]
        enemies = list(filter(lambda s: s["id"] != your_id, observation["board"]["snakes"]))
        elements.extend(f"enemy_{i}_head" for i in range(len(enemies)))
        elements.extend(f"enemy_{i}_body" for i in range(len(enemies)))

        encoded_value_spacing = 1 / (len(elements) - 1)
        encoding_by_element = {element: i * encoded_value_spacing for i, element in enumerate(elements)}

        element_coords = {
            "food": tuple((food["x"], food["y"]) for food in observation["board"]["food"]),
            "your_head": ((observation["you"]["head"]["x"], observation["you"]["head"]["y"]),),
            "your_body": tuple((body["x"], body["y"]) for body in observation["you"]["body"]),
        }
        for i, enemy in enumerate(enemies):
            element_coords[f"enemy_{i}_head"] = ((enemy["head"]["x"], enemy["head"]["y"]),)
            element_coords[f"enemy_{i}_body"] = tuple((body["x"], body["y"]) for body in enemy["body"])

        array = np.zeros(self._space.shape, dtype=np.float16)
        assert array.shape[0] == 1, "Only one channel is supported"
        for element, coords in element_coords.items():
            array[0][list(zip(*coords))] = encoding_by_element[element]

        return array
