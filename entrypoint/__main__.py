import argparse
import pathlib
import logging

from . import configuration as cfg


logger = logging.getLogger(__name__)


class Arguments:

    def __init__(
        self,
        configuration_filepath: pathlib.Path,
        configuration_filetype: cfg.ConfigurationType,
        log_level: str,
    ) -> None:
        self._configuration_filepath = configuration_filepath
        self._configuration_filetype = configuration_filetype
        self._log_level = log_level

    @property
    def configuration_filepath(self):
        return self._configuration_filepath
    
    @property
    def configuration_filetype(self):
        return self._configuration_filetype
    
    @property
    def log_level(self):
        return self._log_level
    
    def __str__(self) -> str:
        return f"Arguments(configuration_filepath={self.configuration_filepath}, configuration_filetype={self.configuration_filetype})"


class EntrypointArgumentParser:

    def __init__(self) -> None:
        parser = argparse.ArgumentParser(
            prog="RL-Battlesnake Entrypoint",
            description="Runs games of Battlesnake according to the input configuration",
        )
        configuration_type_group = parser.add_mutually_exclusive_group(required=True)
        configuration_type_group.add_argument("--yaml", metavar="YAML-CONFIG-FILEPATH")
        parser.add_argument("--log-level", metavar="LOG-LEVEL", default="INFO")
        self._parser = parser

    def parse_args(self) -> Arguments:
        args = self._parser.parse_args()

        for configuration_filetype, configuration_filepath in [
            (cfg.ConfigurationType.YAML, args.yaml),
        ]:
            if configuration_filepath is not None:
                break

        assert configuration_filepath is not None

        configuration_filepath = pathlib.Path(configuration_filepath)
        if not configuration_filepath.is_file():
            raise ValueError("Provided configuration filepath is invalid.")

        return Arguments(configuration_filepath, configuration_filetype, args.log_level)


def init_logger(level: str):
    logger.setLevel(level)

    formatter = logging.Formatter('%(levelname)-8s %(pathname)s:%(lineno)d: %(message)s')

    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(formatter)

    logger.addHandler(ch)


def main():
    args = EntrypointArgumentParser().parse_args()
    init_logger(args.log_level)
    logger.info(f"Starting using provided {args.configuration_filetype} configuration file: {args.configuration_filepath}")
    configuration = cfg.ConfigurationText(args.configuration_filepath.read_text(), args.configuration_filetype).parse().build()
    logger.debug(f"Parsed configuration: {configuration}")

if __name__ == "__main__":
    main()