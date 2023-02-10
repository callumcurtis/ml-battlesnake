import typing
import abc
import enum

import yaml

from . import model


class ConfigurationType(enum.Enum):
    YAML = enum.auto()

    def __str__(self) -> str:
        return self.name


class Configuration:

    def __init__(self, snakes: list[model.Snake]) -> None:
        self._snakes = snakes

    @property
    def snakes(self) -> list[model.Snake]:
        return self._snakes
    
    def __str__(self) -> str:
        return f"Configuration(snakes={self.snakes})"


class ConfigurationBuilder:

    def __init__(self, snakes: list[model.Snake]) -> None:
        self._snakes = snakes

    def build(self) -> Configuration:
        return Configuration(self._snakes)


class ConfigurationText:

    def __init__(self, payload: str, type_: ConfigurationType) -> None:
        self._payload = payload
        self._type = type_

    @property
    def payload(self) -> str:
        return self._payload

    @property
    def type(self) -> ConfigurationType:
        return self._type

    def parse(self) -> ConfigurationBuilder:
        return _get_parser().parse(_get_tokenizer(self.type).tokenize(self.payload))
    
    def __str__(self) -> str:
        return f"ConfigurationText(payload={self.payload}, type={self.type})"


_TOKENIZER_BY_CONFIGURATION_TYPE = {}


def _tokenizer_for(configuration_type: ConfigurationType):
    def wrapper(cls):
        assert configuration_type not in _TOKENIZER_BY_CONFIGURATION_TYPE
        _TOKENIZER_BY_CONFIGURATION_TYPE[configuration_type] = cls
        return cls
    return wrapper


class _Tokenizer(abc.ABC):

    @abc.abstractmethod
    def tokenize(self, text: str) -> typing.Any:
        pass


@_tokenizer_for(ConfigurationType.YAML)
class _YamlTokenizer(_Tokenizer):

    def tokenize(self, text: str) -> typing.Any:
        return yaml.load(text, yaml.Loader)


def _get_tokenizer(configuration_type: ConfigurationType) -> _Tokenizer:
    assert configuration_type in _TOKENIZER_BY_CONFIGURATION_TYPE
    return _TOKENIZER_BY_CONFIGURATION_TYPE[configuration_type]()


class _Parser:

    def parse(self, tokens: typing.Any) -> ConfigurationBuilder:
        snakes = [model.Snake(snake["name"], snake["path"]) for snake in tokens["snakes"]]
        return ConfigurationBuilder(snakes)


def _get_parser() -> _Parser:
    return _Parser()
