class BattlesnakeEnvironmentConfiguration:

    DEFAULT_COLORS = [
        "#00FF00",
        "#0000FF",
        "#FF00FF",
        "#FFFF00",
    ]

    def __init__(
        self,
        possible_agents: list[str],
        render_mode = "human",
        width = 11,
        height = 11,
        colors = DEFAULT_COLORS,
        game_map = "standard",
        game_type = "standard",
        seed: int = None,
    ) -> None:
        self.possible_agents = possible_agents
        self.render_mode = render_mode
        self.width = width
        self.height = height
        self.colors = colors
        self.game_map = game_map
        self.game_type = game_type
        self.seed = seed
