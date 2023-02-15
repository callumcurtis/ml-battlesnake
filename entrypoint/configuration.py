import typing
import abc
import pathlib
import enum
import re

import yaml

from . import model
from . import paths


class ConfigurationVariable(enum.Enum):
    BOARD_DIR = enum.auto()
    SNAKES_DIR = enum.auto()

    def resolve(self) -> typing.Any:
        if self == ConfigurationVariable.BOARD_DIR:
            return paths.BOARD_DIR
        elif self == ConfigurationVariable.SNAKES_DIR:
            return paths.SNAKES_DIR
        assert False, f"Cannot resolve configuration variable: {self}"


class ConfigurationType(enum.Enum):
    YAML = enum.auto()

    def __str__(self) -> str:
        return self.name


class Configuration:

    def __init__(
        self,
        snakes: list[model.Snake],
        board: typing.Optional[model.Board],
    ) -> None:
        self._snakes = snakes
        self._board = board

    @property
    def snakes(self) -> list[model.Snake]:
        return self._snakes
    
    @property
    def board(self) -> typing.Optional[model.Board]:
        return self._board

    def __str__(self) -> str:
        return f"Configuration(snakes={self.snakes}, board={self.board})"


class ConfigurationBuilder:

    def __init__(self, snakes: list[model.Snake]) -> None:
        self._snakes = snakes
        self._board = None
    
    def with_board(self, board: typing.Optional[model.Board]) -> 'ConfigurationBuilder':
        self._board = board
        return self

    def build(self) -> Configuration:
        return Configuration(self._snakes, self._board)


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
        return _get_parsing_pipeline_for(self.type).parse(self.payload)
    
    def __str__(self) -> str:
        return f"ConfigurationText(payload={self.payload}, type={self.type})"


_TOKENIZER_BY_CONFIGURATION_TYPE: dict[ConfigurationType, '_Tokenizer'] = {}


def _assign_tokenizer_to_type(configuration_type: ConfigurationType):
    def wrapper(cls: _Tokenizer):
        assert configuration_type not in _TOKENIZER_BY_CONFIGURATION_TYPE
        _TOKENIZER_BY_CONFIGURATION_TYPE[configuration_type] = cls
        return cls
    return wrapper


class _Tokenizer(abc.ABC):

    @abc.abstractmethod
    def tokenize(self, text: str) -> typing.Any:
        pass


@_assign_tokenizer_to_type(ConfigurationType.YAML)
class _YamlTokenizer(_Tokenizer):

    def tokenize(self, text: str) -> typing.Any:
        return yaml.load(text, yaml.Loader)


def _get_tokenizer_for_type(type_: ConfigurationType) -> _Tokenizer:
        assert type_ in _TOKENIZER_BY_CONFIGURATION_TYPE
        return _TOKENIZER_BY_CONFIGURATION_TYPE[type_]()


class _TokenSubstitutor:
    
    def substitute(self, tokens: typing.Any) -> typing.Any:
        if isinstance(tokens, str):
            with_substitutions = self._substitute_known_variables_in_string(tokens)
            unresolved_variables = self._get_unresolved_variables_in_string(with_substitutions)
            if unresolved_variables:
                formatted_unresolved_variables = ", ".join(map(lambda v: v.strip(), unresolved_variables))
                raise ValueError(f"Unresolved configuration variables: {formatted_unresolved_variables}")
            return with_substitutions
        elif isinstance(tokens, list) or isinstance(tokens, tuple):
            return [self.substitute(token) for token in tokens]
        elif isinstance(tokens, dict):
            return {self.substitute(key): self.substitute(value) for key, value in tokens.items()}
        else:
            return tokens

    def _substitute_known_variables_in_string(self, string: str) -> str:
        for var in ConfigurationVariable:
            string = re.sub(rf"\${{{{\s*{var.name}\s*}}}}", str(var.resolve()), string)
        return string

    def _get_unresolved_variables_in_string(self, string: str) -> list[str]:
        return re.findall(r"(?<=\${{)([^{}]+)(?=}})", string)


class _TokenTranslator:

    def translate(self, tokens: typing.Any) -> ConfigurationBuilder:
        snakes = [model.Snake(snake["name"], snake["path"]) for snake in tokens["snakes"]]
        board = tokens.get("board")
        if board is not None:
            board = model.Board(board["HOST"], board["PORT"], pathlib.Path(board["path"]))
        return ConfigurationBuilder(snakes).with_board(board)


class _ParsingPipeline:

    def __init__(
        self,
        tokenizer: _Tokenizer,
        substitutor: _TokenSubstitutor,
        translator: _TokenTranslator,
    ) -> None:
        self._tokenizer = tokenizer
        self._substitutor = substitutor
        self._translator = translator

    def parse(self, text: str) -> ConfigurationBuilder:
        tokens = self._tokenizer.tokenize(text)
        tokens = self._substitutor.substitute(tokens)
        return self._translator.translate(tokens)


def _get_parsing_pipeline_for(type_: ConfigurationType) -> _ParsingPipeline:
    tokenizer = _get_tokenizer_for_type(type_)
    substitutor = _TokenSubstitutor()
    translator = _TokenTranslator()
    return _ParsingPipeline(tokenizer, substitutor, translator)
