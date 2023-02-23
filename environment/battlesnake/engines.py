import ctypes
import json
import pathlib
import enum
import abc


class BattlesnakeEngine(abc.ABC):

    class Moves(enum.Enum):
        UP = 0
        DOWN = enum.auto()
        LEFT = enum.auto()
        RIGHT = enum.auto()

        def __str__(self):
            return self.name.lower()

    @abc.abstractmethod
    def reset(self, settings) -> dict:
        pass

    @abc.abstractmethod
    def step(self, moves) -> dict:
        pass

    @abc.abstractmethod
    def render(self) -> None:
        pass


def _battlesnake_dll_engine():

    loaded_attrs = []

    def triggers_load(func):
        loaded_attrs.append(func)
        def wrapper(self, *args, **kwargs):
            if not self._is_loaded():
                self._load()
            return func(self, *args, **kwargs)
        return wrapper

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

            self._setup = setup
            self._reset = reset
            self._step = step
            self._done = done
            self._render = render

        def _is_loaded(self):
            assert all(x is None for x in loaded_attrs) or all(x is not None for x in loaded_attrs), "the dll is in an inconsistent state"
            return self._setup is not None

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
        def step(self, moves) -> dict:
            moves = json.dumps(moves).encode("utf-8")
            res = self._step(moves)
            return json.loads(res.decode("utf-8"))

    return BattlesnakeDllEngine


BattlesnakeDllEngine = _battlesnake_dll_engine()
