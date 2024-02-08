import pytest

from ml_battlesnake.learning.environment.types import (
    Coordinates,
    Observation,
    SnakeObservation,
    BattlesnakeEnvironmentConfiguration,
)
from ml_battlesnake.learning.environment.observation_transformers import ObservationToImage


@pytest.fixture
def observation_with_four_snakes():
    return Observation(
        turn=0,
        snakes=[
            SnakeObservation(
                id="agent_0",
                name="agent_0",
                health=100,
                body=[
                    Coordinates(x=5, y=9),
                    Coordinates(x=5, y=9),
                    Coordinates(x=5, y=9)
                ],
                head=Coordinates(x=5, y=9)
            ),
            SnakeObservation(
                id="agent_1",
                name="agent_1",
                health=100,
                body=[
                    Coordinates(x=5, y=1),
                    Coordinates(x=5, y=1),
                    Coordinates(x=5, y=1)
                ],
                head=Coordinates(x=5, y=1)
            ),
            SnakeObservation(
                id="agent_2",
                name="agent_2",
                health=100,
                body=[
                    Coordinates(x=1, y=5),
                    Coordinates(x=1, y=5),
                    Coordinates(x=1, y=5)
                ],
                head=Coordinates(x=1, y=5)
            ),
            SnakeObservation(
                id="agent_3",
                name="agent_3",
                health=100,
                body=[
                    Coordinates(x=9, y=5),
                    Coordinates(x=9, y=5),
                    Coordinates(x=9, y=5)
                ],
                head=Coordinates(x=9, y=5)
            )
        ],
        food=[
            Coordinates(x=6, y=10),
            Coordinates(x=6, y=0),
            Coordinates(x=0, y=6),
            Coordinates(x=10, y=4),
            Coordinates(x=5, y=5)
        ],
        you=SnakeObservation(
            id="agent_0",
            name="agent_0",
            health=100,
            body=[
                Coordinates(x=5, y=9),
                Coordinates(x=5, y=9),
                Coordinates(x=5, y=9)
            ],
            head=Coordinates(x=5, y=9)
        )
    )

@pytest.fixture
def env_config():
    return BattlesnakeEnvironmentConfiguration(
        possible_agents=["agent_0", "agent_1", "agent_2", "agent_3"],
        width=11,
        height=11,
    )

class TestObservationToImage:

    @pytest.fixture
    def observation_to_image(
        self,
        env_config: BattlesnakeEnvironmentConfiguration,
    ):
        return ObservationToImage(env_config)

    def test_snake_heads_appear(
        self,
        observation_to_image: ObservationToImage,
        observation_with_four_snakes: Observation,
    ):
        image = observation_to_image.transform(observation_with_four_snakes)
        assert image[0, 5, 9] == observation_to_image.value_by_pixel_class[observation_to_image.PixelClass.YOUR_HEAD]
        assert image[0, 5, 1] == observation_to_image.value_by_pixel_class[observation_to_image.PixelClass.ENEMY_HEAD]
        assert image[0, 1, 5] == observation_to_image.value_by_pixel_class[observation_to_image.PixelClass.ENEMY_HEAD]
        assert image[0, 9, 5] == observation_to_image.value_by_pixel_class[observation_to_image.PixelClass.ENEMY_HEAD]
