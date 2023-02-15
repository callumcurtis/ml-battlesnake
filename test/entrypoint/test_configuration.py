import pathlib

import pytest

from entrypoint import configuration as cfg
from entrypoint import model

def test_configuration_types():
    assert list(cfg.ConfigurationType) == [cfg.ConfigurationType.YAML]

def test_configuration_type_str():
    assert str(cfg.ConfigurationType.YAML) == "YAML"

def test_configuration_properties():
    snakes = [model.Snake("snake-1", "snake-1-url"), model.Snake("snake-2", "snake-2-url")]
    board = model.Board("fake-host", 1234, pathlib.Path("fake-board-path"))
    configuration = cfg.Configuration(snakes, board)
    assert configuration.snakes == snakes
    assert configuration.board == board

def test_configuration_str():
    snakes = [model.Snake("snake-1", "snake-1-url"), model.Snake("snake-2", "snake-2-url")]
    board = model.Board("fake-host", 1234, pathlib.Path("fake-board-path"))
    configuration = cfg.Configuration(snakes, board)
    assert str(configuration) == f"Configuration(snakes={snakes}, board={board})"

def test_configuration_builder_given_default_board():
    snakes = [model.Snake("snake-1", "snake-1-url"), model.Snake("snake-2", "snake-2-url")]
    configuration_builder = cfg.ConfigurationBuilder(snakes)
    configuration = configuration_builder.build()
    assert configuration.snakes == snakes
    assert configuration.board is None

def test_configuration_builder_given_board():
    snakes = [model.Snake("snake-1", "snake-1-url"), model.Snake("snake-2", "snake-2-url")]
    board = model.Board("fake-host", 1234, pathlib.Path("fake-board-path"))
    configuration_builder = cfg.ConfigurationBuilder(snakes).with_board(board)
    configuration = configuration_builder.build()
    assert configuration.snakes == snakes
    assert configuration.board == board

def test_configuration_text(monkeypatch):
    expected_payload = "expected-payload"
    expected_type = "expected-type"
    expected_parsed_configuration = "expected-parsed-configuration"

    class _MockParsingPipeline:
        def parse(self, text) -> cfg.Configuration:
            assert text == expected_payload
            return expected_parsed_configuration
    
    def _mock_get_parsing_pipeline_for(type_: cfg.ConfigurationType) -> _MockParsingPipeline:
        assert type_ == expected_type
        return _MockParsingPipeline()
    
    monkeypatch.setattr("entrypoint.configuration._get_parsing_pipeline_for", _mock_get_parsing_pipeline_for)

    configuration_text = cfg.ConfigurationText(expected_payload, expected_type)
    assert configuration_text.payload == expected_payload
    assert configuration_text.type == expected_type
    assert configuration_text.parse() == expected_parsed_configuration

def test_configuration_text_str():
    expected_payload = "expected-payload"
    expected_type = "expected-type"
    configuration_text = cfg.ConfigurationText(expected_payload, expected_type)
    assert str(configuration_text) == f"ConfigurationText(payload={expected_payload}, type={expected_type})"

def test_assign_tokenizer_to_type(monkeypatch):
    tokenizer_by_configuration_type = {}
    monkeypatch.setattr("entrypoint.configuration._TOKENIZER_BY_CONFIGURATION_TYPE", tokenizer_by_configuration_type)

    fake_type = "fake-configuration-type"
    fake_tokenizer = "fake-tokenizer"

    cfg._assign_tokenizer_to_type(fake_type)(fake_tokenizer)
    assert tokenizer_by_configuration_type[fake_type] == fake_tokenizer

def test_assign_tokenizer_to_type_given_duplicate():
    fake_tokenizer = "fake-tokenizer"
    with pytest.raises(AssertionError):
        cfg._assign_tokenizer_to_type(cfg.ConfigurationType.YAML)(fake_tokenizer)

def test_yaml_tokenizer():
    yaml_str = "{snakes: [{name: snake-1, path: snake-1-url}, {name: snake-2, path: snake-2-url}]}"
    expected_tokens = {"snakes": [{"name": "snake-1", "path": "snake-1-url"}, {"name": "snake-2", "path": "snake-2-url"}]}
    tokenizer = cfg._YamlTokenizer()
    assert tokenizer.tokenize(yaml_str) == expected_tokens

def test_get_tokenizer_for_type_given_exists():
    assert isinstance(cfg._get_tokenizer_for_type(cfg.ConfigurationType.YAML), cfg._YamlTokenizer)

def test_get_tokenizer_for_type_given_does_not_exist():
    with pytest.raises(AssertionError):
        cfg._get_tokenizer_for_type("fake-configuration-type")
