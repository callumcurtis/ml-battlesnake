import pathlib
import subprocess
from unittest.mock import Mock

import pytest

from ml_battlesnake.deployment import model


@pytest.fixture
def program_generator():
    def generator():
        i = 0
        while i:= i + 1:
            yield model.Program(
                name=f"test-program-{i}",
                entrypoint=["start", "test", "program", str(i)],
                cwd=pathlib.Path(f"test-program-path-{i}"),
            )
    return generator()


@pytest.fixture
def snake_generator():
    def generator():
        i = 0
        while i:= i + 1:
            yield model.SnakeRoute(
                name=f"test-snake-{i}",
                baseroute=f"test-snake-{i}-baseroute",
            )
    return generator()


@pytest.fixture
def browser_spectator_generator():
    def generator():
        i = 0
        while i:= i + 1:
            yield model.BrowserSpectatorRoute(
                name=f"test-browser-spectator-{i}",
                baseroute=f"test-browser-spectator-{i}-baseroute",
            )
    return generator()


def test_program_properties():
    name = "test-program"
    entrypoint = ["start", "test", "program"]
    cwd = pathlib.Path("test-program-path")
    program = model.Program(
        name=name,
        entrypoint=entrypoint,
        cwd=cwd,
    )
    assert program.name == name
    assert program.entrypoint == entrypoint
    assert program.cwd == cwd


def test_program_str():
    name = "test-program"
    entrypoint = ["start", "test", "program"]
    cwd = pathlib.Path("test-program-path")
    program = model.Program(
        name=name,
        entrypoint=entrypoint,
        cwd=cwd,
    )
    assert str(program) == f"Program(name={name}, entrypoint={entrypoint}, cwd={cwd})"


def test_service_properties(program_generator, snake_generator, browser_spectator_generator):
    name = "test-service"
    env = {"TEST": "test"}
    args = ["arg-0", "arg-1", "arg-2"]
    program = next(program_generator)
    routes = [next(snake_generator), next(browser_spectator_generator), next(snake_generator)]
    service = model.Service(
        name=name,
        program=program,
        env=env,
        args=args,
        routes=routes
    )
    assert service.name == name
    assert service.program == program
    assert service.routes == routes


def test_service_str(program_generator, snake_generator, browser_spectator_generator):
    name = "test-service"
    program = next(program_generator)
    env = {"TEST": "test"}
    args = ["arg-0", "arg-1", "arg-2"]
    routes = [next(snake_generator), next(browser_spectator_generator), next(snake_generator)]
    service = model.Service(
        name=name,
        program=program,
        env=env,
        args=args,
        routes=routes,
    )
    assert str(service) == f"Service(name={name}, program={program}, env={env}, args={args}, routes={routes})"


def test_service_start(monkeypatch):

    old_env = {"OLD_ENVIRON": "old-environ"}
    monkeypatch.setattr("os.environ", old_env)
    added_env = {"HOST": "fake-host", "PORT": "fake-port"}
    args = ["arg-0", "arg-1", "arg-2"]
    entrypoint = ["start", "test", "program"]
    cwd = pathlib.Path("test-program-path")
    mock_popen = Mock()

    program = model.Program(
        name="test-program",
        entrypoint=entrypoint,
        cwd=cwd,
    )

    service = model.Service(
        name="test-service",
        program=program,
        env=added_env,
        args=args,
        routes=[]
    )

    monkeypatch.setattr("subprocess.Popen", mock_popen)
    # TODO: inject process abstractions into service model rather than using mocks through reassignment
    service.stop = Mock()
    service.start()
    mock_popen.assert_called_once_with(entrypoint + args, cwd=cwd, env={**old_env, **added_env}, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)


def test_snake_properties():
    name = "test-snake"
    baseroute = "test-snake-baseroute"
    snake = model.SnakeRoute(
        name=name,
        baseroute=baseroute,
    )
    assert snake.name == name
    assert snake.baseroute == baseroute


def test_snake_str():
    name = "test-snake"
    baseroute = "test-snake-baseroute"
    snake = model.SnakeRoute(
        name=name,
        baseroute=baseroute,
    )
    assert str(snake) == f"Snake(name={name}, baseroute={baseroute})"


def test_browser_spectator_properties():
    name = "test-browser-spectator"
    baseroute = "test-browser-spectator-baseroute"
    browser_spectator = model.BrowserSpectatorRoute(
        name=name,
        baseroute=baseroute,
    )
    assert browser_spectator.name == name
    assert browser_spectator.baseroute == baseroute


def test_browser_spectator_str():
    name = "test-browser-spectator"
    baseroute = "test-browser-spectator-baseroute"
    browser_spectator = model.BrowserSpectatorRoute(
        name=name,
        baseroute=baseroute,
    )
    assert str(browser_spectator) == f"BrowserSpectator(name={name}, baseroute={baseroute})"
