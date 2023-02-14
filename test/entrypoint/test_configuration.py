import pytest

from entrypoint import configuration as cfg
from entrypoint import model

def test_configuration_types():
    assert list(cfg.ConfigurationType) == [cfg.ConfigurationType.YAML]

def test_configuration_type_str():
    assert str(cfg.ConfigurationType.YAML) == "YAML"

def test_configuration_properties():
    snakes = [model.Snake("snake-1", "snake-1-url"), model.Snake("snake-2", "snake-2-url")]
    configuration = cfg.Configuration(snakes)
    assert configuration.snakes == snakes

def test_configuration_str():
    snakes = [model.Snake("snake-1", "snake-1-url"), model.Snake("snake-2", "snake-2-url")]
    configuration = cfg.Configuration(snakes)
    assert str(configuration) == f"Configuration(snakes={snakes})"

def test_configuration_builder():
    snakes = [model.Snake("snake-1", "snake-1-url"), model.Snake("snake-2", "snake-2-url")]
    configuration_builder = cfg.ConfigurationBuilder(snakes)
    configuration = configuration_builder.build()
    assert configuration.snakes == snakes

def test_configuration_text(monkeypatch):

    expected_tokens = "expected-tokens"
    expected_payload = "expected-payload"
    expected_type = "expected-type"
    expected_parsed_configuration = "expected-parsed-configuration"

    class _MockParser:

        def parse(self, tokens):
            assert tokens == expected_tokens
            return expected_parsed_configuration
    
    class _MockTokenizer:

        def tokenize(self, text):
            assert text == expected_payload
            return expected_tokens
    
    def _mock_get_tokenizer(type_):
        assert type_ == expected_type
        return _MockTokenizer()

    monkeypatch.setattr("entrypoint.configuration._get_parser", lambda: _MockParser())
    monkeypatch.setattr("entrypoint.configuration._get_tokenizer", _mock_get_tokenizer)

    configuration_text = cfg.ConfigurationText(expected_payload, expected_type)
    assert configuration_text.payload == expected_payload
    assert configuration_text.type == expected_type
    assert configuration_text.parse() == expected_parsed_configuration

def test_configuration_text_str():
    expected_payload = "expected-payload"
    expected_type = "expected-type"
    configuration_text = cfg.ConfigurationText(expected_payload, expected_type)
    assert str(configuration_text) == f"ConfigurationText(payload={expected_payload}, type={expected_type})"

def test_tokenizer_for_given_exists(monkeypatch):
    tokenizer_by_configuration_type = {}
    monkeypatch.setattr("entrypoint.configuration._TOKENIZER_BY_CONFIGURATION_TYPE", tokenizer_by_configuration_type)

    fake_type = "fake-configuration-type"
    fake_tokenizer = "fake-tokenizer"

    cfg._tokenizer_for(fake_type)(fake_tokenizer)
    assert tokenizer_by_configuration_type[fake_type] == fake_tokenizer

def test_tokenizer_for_given_duplicate():
    fake_tokenizer = "fake-tokenizer"
    with pytest.raises(AssertionError):
        cfg._tokenizer_for(cfg.ConfigurationType.YAML)(fake_tokenizer)

def test_yaml_tokenizer():
    yaml_str = "{snakes: [{name: snake-1, path: snake-1-url}, {name: snake-2, path: snake-2-url}]}"
    expected_tokens = {"snakes": [{"name": "snake-1", "path": "snake-1-url"}, {"name": "snake-2", "path": "snake-2-url"}]}
    tokenizer = cfg._YamlTokenizer()
    assert tokenizer.tokenize(yaml_str) == expected_tokens

def test_get_tokenizer_given_exists():
    assert isinstance(cfg._get_tokenizer(cfg.ConfigurationType.YAML), cfg._YamlTokenizer)

def test_get_tokenizer_given_does_not_exist():
    with pytest.raises(AssertionError):
        cfg._get_tokenizer("fake-configuration-type")

def test_get_parser():
    assert isinstance(cfg._get_parser(), cfg._Parser)
