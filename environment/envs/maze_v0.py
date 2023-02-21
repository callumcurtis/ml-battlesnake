from typing import List
from pettingzoo.utils import parallel_to_aec

# Local Imports
from environment.constants import DEFAULT_COLORS
from environment.envs.base_env import BaseEnv
from environment.rules import Rules


def env(
    rules: Rules,
    num_agent: int = 4,
    colors: List[str] = DEFAULT_COLORS,
):
    env = BaseEnv(
        rules,
        width=21,
        height=19,
        game_map="arcade_maze",
        game_type="wrapped",
        num_agents=num_agent,
        colors=colors,
    )

    # Set the metadata enviorment name
    env.metadata["name"] = "battlesnake-maze_v0"

    # Convert from MARL to AEC API
    env = parallel_to_aec(env)

    return env
