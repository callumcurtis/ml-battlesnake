import ctypes
import json
import pathlib
import abc

from .types import Movement


class BattlesnakeEngine(abc.ABC):
    pass


def _battlesnake_dll_engine():

    def triggers_load(func):
        def decorator(self, *args, **kwargs):
            if not self._is_loaded():
                self._load()
            return func(self, *args, **kwargs)
        return decorator

    class BattlesnakeDllEngine(BattlesnakeEngine):

        def __init__(
            self,
            dll_path: pathlib.Path,
        ) -> None:
            self.dll_path = dll_path
            self._setup = None
            self._reset = None
            self._step = None
            self._done = None
            self._render = None
            self._responses = None
            self._loaded_attrs = [
                self._setup,
                self._reset,
                self._step,
                self._done,
                self._render,
                self._responses,
            ]
        
        def _load(self):
            assert not self._is_loaded(), "the dll is already loaded"
            dll = ctypes.CDLL(self.dll_path)

            setup = dll.setup
            setup.argtypes = [ctypes.c_char_p]

            reset = dll.reset
            reset.argtypes = [ctypes.c_char_p]
            reset.restype = ctypes.c_char_p

            step = dll.step
            step.argtypes = [ctypes.c_char_p]
            step.restype = ctypes.c_char_p

            done = dll.isGameOver
            done.restype = ctypes.c_int

            render = dll.render
            render.argtypes = [ctypes.c_int]

            responses = dll.responses
            responses.restype = ctypes.c_char_p

            self._setup = setup
            self._reset = reset
            self._step = step
            self._done = done
            self._render = render
            self._responses = responses

        def _is_loaded(self):
            assert (
                all(x is None for x in self._loaded_attrs)
                or all(x is not None for x in self._loaded_attrs)
            ), "the dll is in an inconsistent state"
            return self._setup is not None

        def active_snakes(self) -> list[str]:
            return list(snake for snake, response in self.responses().items() if not response["done"])

        @triggers_load
        def render(self, color: bool = True) -> None:
            self._render(1 if color else 0)

        @triggers_load
        def done(self) -> bool:
            return self._done() == 1

        @triggers_load
        def setup(self, settings: dict) -> None:
            settings = json.dumps(settings).encode("utf-8")
            self._setup(settings)

        @triggers_load
        def reset(self, settings) -> dict:
            settings = json.dumps(settings).encode("utf-8")
            res = self._reset(settings)
            return json.loads(res.decode("utf-8"))

        @triggers_load
        def step(self, moves: dict[str, Movement]) -> dict:
            moves = {agent: movement.name.lower() for agent, movement in moves.items()}
            moves = json.dumps(moves).encode("utf-8")
            res = self._step(moves)
            return json.loads(res.decode("utf-8"))

        @triggers_load
        def responses(self) -> dict:
            res = self._responses()
            return json.loads(res.decode("utf-8"))

    return BattlesnakeDllEngine


BattlesnakeDllEngine = _battlesnake_dll_engine()
