from typing import List
from pettingzoo.utils import parallel_to_aec

# Local Imports
from environment.constants import DEFAULT_COLORS
from environment.envs.base_env import BaseEnv
from environment.rules import Rules


def env(
    rules: Rules,
    width: int = 11,
    height: int = 11,
    colors: List[str] = DEFAULT_COLORS,
):
    env = BaseEnv(
        rules,
        wdith=width,
        height=height,
        game_map="hz_rivers_bridges",
        game_type="standard",
        num_agents=4,
        colors=colors,
    )

    # Set the metadata enviorment name
    env.metadata["name"] = "battlesnake-lava-bridges_v0"

    # Convert from MARL to AEC API
    env = parallel_to_aec(env)

    return env
