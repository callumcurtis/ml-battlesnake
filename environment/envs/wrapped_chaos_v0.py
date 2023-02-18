from typing import List
from pettingzoo.utils import parallel_to_aec

# Local Imports
from environment.constants import DEFAULT_COLORS
from environment.envs.base_env import BaseEnv
from environment.rules import Rules


def env(
    rules: Rules,
    width: int = 19,
    height: int = 19,
    colors: List[str] = DEFAULT_COLORS,
):
    env = BaseEnv(
        rules,
        wdith=width,
        height=height,
        game_map="standard",
        game_type="wrapped",
        num_agents=8,
        colors=colors,
    )

    # Set the metadata enviorment name
    env.metadata["name"] = "battlesnake-duels_v0"

    # Convert from MARL to AEC API
    env = parallel_to_aec(env)

    return env
