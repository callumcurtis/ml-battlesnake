import numpy as np
import pytest

from ml_battlesnake.learning.environment.types import (
    Coordinates,
    Observation,
    SnakeObservation,
    BattlesnakeEnvironmentConfiguration,
)
from ml_battlesnake.learning.environment.observation_transformers import (
    BoardEntity,
    ObservationToBinaryMatrices,
    ObservationToImage,
    BoardOrientationTransformer,
)


@pytest.fixture
def observation_of_initial_game_state_with_four_snakes():
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
def observation_of_early_game_state_with_two_snakes():
    return Observation(
        turn=6,
        snakes=[
            SnakeObservation(
                id="agent_2",
                name="agent_2",
                health=94,
                body=[
                    Coordinates(x=3, y=7),
                    Coordinates(x=2, y=7),
                    Coordinates(x=2, y=6)
                ],
                head=Coordinates(x=3, y=7)
            ),
            SnakeObservation(
                id="agent_3",
                name="agent_3",
                health=94,
                body=[
                    Coordinates(x=4, y=6),
                    Coordinates(x=5, y=6),
                    Coordinates(x=5, y=7)
                ],
                head=Coordinates(x=4, y=6)
            )
        ],
        food=[
            Coordinates(x=10, y=4),
            Coordinates(x=4, y=0),
            Coordinates(x=0, y=6),
            Coordinates(x=6, y=10),
            Coordinates(x=5, y=5),
            Coordinates(x=1, y=2)
        ],
        you=SnakeObservation(
            id="agent_2",
            name="agent_2",
            health=94,
            body=[
                Coordinates(x=3, y=7),
                Coordinates(x=2, y=7),
                Coordinates(x=2, y=6)
            ],
            head=Coordinates(x=3, y=7)
        )
    )


@pytest.fixture
def env_config():
    return BattlesnakeEnvironmentConfiguration(
        possible_agents=["agent_0", "agent_1", "agent_2", "agent_3"],
        width=11,
        height=11,
    )


@pytest.fixture
def board_orientation_transformer():
    return BoardOrientationTransformer()


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
        observation_of_initial_game_state_with_four_snakes: Observation,
        board_orientation_transformer: BoardOrientationTransformer,
    ):
        image = observation_to_image.transform(observation_of_initial_game_state_with_four_snakes)
        image = board_orientation_transformer.to_natural_orientation(image)
        assert image[0, 5, 9] == observation_to_image.value_by_entity[BoardEntity.YOUR_HEAD]
        assert image[0, 5, 1] == observation_to_image.value_by_entity[BoardEntity.ENEMY_HEAD]
        assert image[0, 1, 5] == observation_to_image.value_by_entity[BoardEntity.ENEMY_HEAD]
        assert image[0, 9, 5] == observation_to_image.value_by_entity[BoardEntity.ENEMY_HEAD]

    def test_empty_spaces_appear(
        self,
        observation_to_image: ObservationToImage,
        observation_of_initial_game_state_with_four_snakes: Observation,
        board_orientation_transformer: BoardOrientationTransformer,
    ):
        image = observation_to_image.transform(observation_of_initial_game_state_with_four_snakes)
        image = board_orientation_transformer.to_natural_orientation(image)
        for x in range(11):
            for y in range(11):
                if (x, y) not in [(5, 9), (5, 1), (1, 5), (9, 5), (6, 10), (6, 0), (0, 6), (10, 4), (5, 5)]:
                    assert image[0, x, y] == observation_to_image.value_by_entity[BoardEntity.EMPTY]

    def test_food_appears(
        self,
        observation_to_image: ObservationToImage,
        observation_of_initial_game_state_with_four_snakes: Observation,
        board_orientation_transformer: BoardOrientationTransformer,
    ):
        image = observation_to_image.transform(observation_of_initial_game_state_with_four_snakes)
        image = board_orientation_transformer.to_natural_orientation(image)
        assert image[0, 6, 10] == observation_to_image.value_by_entity[BoardEntity.FOOD]
        assert image[0, 6, 0] == observation_to_image.value_by_entity[BoardEntity.FOOD]
        assert image[0, 0, 6] == observation_to_image.value_by_entity[BoardEntity.FOOD]
        assert image[0, 10, 4] == observation_to_image.value_by_entity[BoardEntity.FOOD]
        assert image[0, 5, 5] == observation_to_image.value_by_entity[BoardEntity.FOOD]

    def test_snake_body_directions(
        self,
        observation_to_image: ObservationToImage,
        observation_of_early_game_state_with_two_snakes: Observation,
        board_orientation_transformer: BoardOrientationTransformer,
    ):
        image = observation_to_image.transform(observation_of_early_game_state_with_two_snakes)
        image = board_orientation_transformer.to_natural_orientation(image)
        assert image[0, 3, 7] == observation_to_image.value_by_entity[BoardEntity.YOUR_HEAD]
        assert image[0, 2, 7] == observation_to_image.value_by_entity[BoardEntity.NEXT_SNAKE_PART_IS_RIGHT]
        assert image[0, 2, 6] == observation_to_image.value_by_entity[BoardEntity.NEXT_SNAKE_PART_IS_UP]
        assert image[0, 4, 6] == observation_to_image.value_by_entity[BoardEntity.ENEMY_HEAD]
        assert image[0, 5, 6] == observation_to_image.value_by_entity[BoardEntity.NEXT_SNAKE_PART_IS_LEFT]
        assert image[0, 5, 7] == observation_to_image.value_by_entity[BoardEntity.NEXT_SNAKE_PART_IS_DOWN]


class TestObservationToBinaryMatrices:

    @pytest.fixture
    def observation_to_binary_matrices(
        self,
        env_config: BattlesnakeEnvironmentConfiguration,
    ):
        return ObservationToBinaryMatrices(env_config)

    def test_food_binary_matrix(
        self,
        observation_to_binary_matrices: ObservationToBinaryMatrices,
        observation_of_initial_game_state_with_four_snakes: Observation,
        board_orientation_transformer: BoardOrientationTransformer,
    ):
        matrices = observation_to_binary_matrices.transform(observation_of_initial_game_state_with_four_snakes)
        matrices = board_orientation_transformer.to_natural_orientation(matrices)
        food_matrix = matrices[observation_to_binary_matrices.get_matrix_index(BoardEntity.FOOD)]
        expected = np.zeros((11, 11))
        expected[6, 10] = 1
        expected[6, 0] = 1
        expected[0, 6] = 1
        expected[10, 4] = 1
        expected[5, 5] = 1
        assert np.array_equal(food_matrix, expected)

    def test_next_snake_part_is_on_top_binary_matrix(
        self,
        observation_to_binary_matrices: ObservationToBinaryMatrices,
        observation_of_initial_game_state_with_four_snakes: Observation,
        board_orientation_transformer: BoardOrientationTransformer,
    ):
        matrices = observation_to_binary_matrices.transform(observation_of_initial_game_state_with_four_snakes)
        matrices = board_orientation_transformer.to_natural_orientation(matrices)
        next_snake_part_is_on_top_matrix = matrices[observation_to_binary_matrices.get_matrix_index(BoardEntity.NEXT_SNAKE_PART_IS_ON_TOP)]
        expected = np.zeros((11, 11))
        expected[5, 9] = 1
        expected[5, 1] = 1
        expected[1, 5] = 1
        expected[9, 5] = 1
        assert np.array_equal(next_snake_part_is_on_top_matrix, expected)

    def test_next_snake_part_is_up_binary_matrix(
        self,
        observation_to_binary_matrices: ObservationToBinaryMatrices,
        observation_of_early_game_state_with_two_snakes: Observation,
        board_orientation_transformer: BoardOrientationTransformer,
    ):
        matrices = observation_to_binary_matrices.transform(observation_of_early_game_state_with_two_snakes)
        matrices = board_orientation_transformer.to_natural_orientation(matrices)
        next_snake_part_is_up_matrix = matrices[observation_to_binary_matrices.get_matrix_index(BoardEntity.NEXT_SNAKE_PART_IS_UP)]
        expected = np.zeros((11, 11))
        expected[2, 6] = 1
        assert np.array_equal(next_snake_part_is_up_matrix, expected)

    def test_next_snake_part_is_down_binary_matrix(
        self,
        observation_to_binary_matrices: ObservationToBinaryMatrices,
        observation_of_early_game_state_with_two_snakes: Observation,
        board_orientation_transformer: BoardOrientationTransformer,
    ):
        matrices = observation_to_binary_matrices.transform(observation_of_early_game_state_with_two_snakes)
        matrices = board_orientation_transformer.to_natural_orientation(matrices)
        next_snake_part_is_down_matrix = matrices[observation_to_binary_matrices.get_matrix_index(BoardEntity.NEXT_SNAKE_PART_IS_DOWN)]
        expected = np.zeros((11, 11))
        expected[5, 7] = 1
        assert np.array_equal(next_snake_part_is_down_matrix, expected)

    def test_next_snake_part_is_left_binary_matrix(
        self,
        observation_to_binary_matrices: ObservationToBinaryMatrices,
        observation_of_early_game_state_with_two_snakes: Observation,
        board_orientation_transformer: BoardOrientationTransformer,
    ):
        matrices = observation_to_binary_matrices.transform(observation_of_early_game_state_with_two_snakes)
        matrices = board_orientation_transformer.to_natural_orientation(matrices)
        next_snake_part_is_left_matrix = matrices[observation_to_binary_matrices.get_matrix_index(BoardEntity.NEXT_SNAKE_PART_IS_LEFT)]
        expected = np.zeros((11, 11))
        expected[5, 6] = 1
        assert np.array_equal(next_snake_part_is_left_matrix, expected)

    def test_next_snake_part_is_right_binary_matrix(
        self,
        observation_to_binary_matrices: ObservationToBinaryMatrices,
        observation_of_early_game_state_with_two_snakes: Observation,
        board_orientation_transformer: BoardOrientationTransformer,
    ):
        matrices = observation_to_binary_matrices.transform(observation_of_early_game_state_with_two_snakes)
        matrices = board_orientation_transformer.to_natural_orientation(matrices)
        next_snake_part_is_right_matrix = matrices[observation_to_binary_matrices.get_matrix_index(BoardEntity.NEXT_SNAKE_PART_IS_RIGHT)]
        expected = np.zeros((11, 11))
        expected[2, 7] = 1
        assert np.array_equal(next_snake_part_is_right_matrix, expected)

    def test_enemy_head_binary_matrix(
        self,
        observation_to_binary_matrices: ObservationToBinaryMatrices,
        observation_of_initial_game_state_with_four_snakes: Observation,
        board_orientation_transformer: BoardOrientationTransformer,
    ):
        matrices = observation_to_binary_matrices.transform(observation_of_initial_game_state_with_four_snakes)
        matrices = board_orientation_transformer.to_natural_orientation(matrices)
        enemy_head_matrix = matrices[observation_to_binary_matrices.get_matrix_index(BoardEntity.ENEMY_HEAD)]
        expected = np.zeros((11, 11))
        expected[5, 1] = 1
        expected[1, 5] = 1
        expected[9, 5] = 1
        assert np.array_equal(enemy_head_matrix, expected)

    def test_your_head_binary_matrix(
        self,
        observation_to_binary_matrices: ObservationToBinaryMatrices,
        observation_of_initial_game_state_with_four_snakes: Observation,
        board_orientation_transformer: BoardOrientationTransformer,
    ):
        matrices = observation_to_binary_matrices.transform(observation_of_initial_game_state_with_four_snakes)
        matrices = board_orientation_transformer.to_natural_orientation(matrices)
        your_head_matrix = matrices[observation_to_binary_matrices.get_matrix_index(BoardEntity.YOUR_HEAD)]
        expected = np.zeros((11, 11))
        expected[5, 9] = 1
        assert np.array_equal(your_head_matrix, expected)
