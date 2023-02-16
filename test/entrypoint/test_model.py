import pathlib

import pytest

from entrypoint import model


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
            yield model.Snake(
                name=f"test-snake-{i}",
                baseroute=f"test-snake-{i}-baseroute",
            )
    return generator()


@pytest.fixture
def board_generator():
    def generator():
        i = 0
        while i:= i + 1:
            yield model.Board(
                name=f"test-board-{i}",
                baseroute=f"test-board-{i}-baseroute",
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


def test_service_properties(program_generator, snake_generator, board_generator):
    name = "test-service"
    env = {"TEST": "test"}
    args = ["arg-0", "arg-1", "arg-2"]
    program = next(program_generator)
    routes = [next(snake_generator), next(board_generator), next(snake_generator)]
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


def test_service_str(program_generator, snake_generator, board_generator):
    name = "test-service"
    program = next(program_generator)
    env = {"TEST": "test"}
    args = ["arg-0", "arg-1", "arg-2"]
    routes = [next(snake_generator), next(board_generator), next(snake_generator)]
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
    popen_result = "popen-result"

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

    def verify_popen_usage(cmd, **kwargs):
        assert cmd == entrypoint + args
        assert kwargs["cwd"] == cwd
        assert kwargs["env"] == {**old_env, **added_env}
        return popen_result

    monkeypatch.setattr("subprocess.Popen", lambda *args, **kwargs: verify_popen_usage(*args, **kwargs))
    service.start()
    assert service._popen == popen_result


def test_snake_properties():
    name = "test-snake"
    baseroute = "test-snake-baseroute"
    snake = model.Snake(
        name=name,
        baseroute=baseroute,
    )
    assert snake.name == name
    assert snake.baseroute == baseroute


def test_snake_str():
    name = "test-snake"
    baseroute = "test-snake-baseroute"
    snake = model.Snake(
        name=name,
        baseroute=baseroute,
    )
    assert str(snake) == f"Snake(name={name}, baseroute={baseroute})"


def test_board_properties():
    name = "test-board"
    baseroute = "test-board-baseroute"
    board = model.Board(
        name=name,
        baseroute=baseroute,
    )
    assert board.name == name
    assert board.baseroute == baseroute


def test_board_str():
    name = "test-board"
    baseroute = "test-board-baseroute"
    board = model.Board(
        name=name,
        baseroute=baseroute,
    )
    assert str(board) == f"Board(name={name}, baseroute={baseroute})"
