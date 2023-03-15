import argparse
import logging
import sys

from ml_battlesnake.deployment import model
from ml_battlesnake.common import paths


logger = logging.getLogger(__name__)


class Arguments:

    def __init__(self, log_level: str) -> None:
        self._log_level = log_level
    
    @property
    def log_level(self):
        return self._log_level
    
    def __str__(self) -> str:
        return f"Arguments(log_level={self.log_level})"


class ArgumentParser:

    def __init__(self) -> None:
        parser = argparse.ArgumentParser(
            prog="Production RL-Battlesnake",
            description="Runs games of Battlesnake according to the input configuration",
        )
        parser.add_argument("--log-level", metavar="LOG-LEVEL", default="DEBUG")
        self._parser = parser

    def parse_args(self) -> Arguments:
        args = self._parser.parse_args()
        return Arguments(args.log_level)


def init_root_logger(level: str):
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    formatter = logging.Formatter('%(levelname)-8s %(pathname)s:%(lineno)d: %(message)s')

    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(formatter)

    root_logger.addHandler(ch)


def main():
    args = ArgumentParser().parse_args()
    init_root_logger(args.log_level)
    logger.info(f"Starting using provided {args}")


    snake_program = model.Program(
        name="official-starter-snake",
        entrypoint=["python", "main.py"],
        cwd=paths.SNAKES_DIR/"official-starter-snake",
    )

    snake_service_0 = model.Service(
        name="official-starter-snake-0",
        program=snake_program,
        env={"PORT": "5274"},
        routes=[
            model.Snake(
                name="rattley",
                baseroute="http://localhost:5274",
            ),
        ],
    )

    snake_service_1 = model.Service(
        name="official-starter-snake-1",
        program=snake_program,
        env={"PORT": "6830"},
        routes=[
            model.Snake(
                name="anacondie",
                baseroute="http://localhost:6830",
            ),
        ],
    )

    browser_spectator_program = model.Program(
        name="browser-spectator",
        entrypoint=["npm", "start"],
        cwd=paths.BROWSER_SPECTATOR_DIR,
    )

    browser_spectator_service = model.Service(
        name="browser-spectator-0",
        program=browser_spectator_program,
        env={"HOST": "127.0.0.1", "PORT": "9000"},
        routes=[
            model.BrowserSpectator(
                name="main-browser-spectator",
                baseroute="http://localhost:9000",
            ),
        ],
    )

    engine_program = model.Program(
        name="engine",
        entrypoint=["./engine", "play"],
        cwd=paths.BIN_DIR,
    )

    snake_args = []
    for service in [snake_service_0, snake_service_1]:
        for route in service.routes:
            if isinstance(route, model.Snake):
                snake_args.extend(["--name", route.name, "--url", route.baseroute])

    engine_service = model.Service(
        name="engine-0",
        program=engine_program,
        args=["--browser", "--board-url", "http://localhost:9000", *snake_args],
        routes=[
            model.Engine(
                name="engine-0",
                baseroute="http://localhost:8080",
            ),
        ],
    )

    import time
    snake_service_0.start()
    snake_service_1.start()
    browser_spectator_service.start()

    # Must wait for snakes to start
    # TODO: use a wait-for-it script
    time.sleep(5)
    engine_service.start(stderr=sys.stdout)

    # Wait for the game to finish
    # TODO: use a wait-for-it script
    time.sleep(100)


if __name__ == "__main__":
    main()
