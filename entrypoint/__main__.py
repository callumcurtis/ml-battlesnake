import argparse
import logging

from . import model
from . import paths


logger = logging.getLogger(__name__)


class Arguments:

    def __init__(self, log_level: str) -> None:
        self._log_level = log_level
    
    @property
    def log_level(self):
        return self._log_level
    
    def __str__(self) -> str:
        return f"Arguments(log_level={self.log_level})"


class EntrypointArgumentParser:

    def __init__(self) -> None:
        parser = argparse.ArgumentParser(
            prog="RL-Battlesnake Entrypoint",
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
    args = EntrypointArgumentParser().parse_args()
    init_root_logger(args.log_level)
    logger.info(f"Starting using provided {args}")

    process = model.Service(
        name="untimely-neglected-wearable-0",
        program=model.Program(
            name="untimely-neglected-wearable",
            entrypoint=["python", "server.py"],
            cwd=paths.SNAKES_DIR/"untimely-neglected-wearable"),
        env={"PORT": "5274"},
        routes=[
            model.Snake(
                name="untimely-neglected-wearable",
                baseroute="http://localhost:5274",
            ),
        ],
    )

    process.start()
    import time
    time.sleep(15)
    process.stop()


if __name__ == "__main__":
    main()
