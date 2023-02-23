import ctypes
import json
import pathlib


class Rules:

    _MOVES = ["up", "down", "left", "right"]

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
    
    def load(self):
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

    def unload(self):
        self._setup = None
        self._reset = None
        self._step = None
        self._done = None
        self._render = None

    def isloaded(self):
        loaded_attrs = [self._setup, self._reset, self._step, self._done, self._render]
        assert all(x is None for x in loaded_attrs) or all(x is not None for x in loaded_attrs), "the dll is in an inconsistent state"
        return self._setup is not None

    def moves(self) -> list[str]:
        return self._MOVES

    def render(self, color: bool = True) -> None:
        assert self.isloaded(), "the dll has not been loaded"
        self._render(1 if color else 0)

    def done(self) -> bool:
        assert self.isloaded(), "the dll has not been loaded"
        return self._done() == 1

    def setup(self, options: dict) -> None:
        assert self.isloaded(), "the dll has not been loaded"
        options = json.dumps(options).encode("utf-8")
        self._setup(options)

    def reset(self, options: dict) -> dict:
        assert self.isloaded(), "the dll has not been loaded"
        options = json.dumps(options).encode("utf-8")
        res = self._reset(options)
        return json.loads(res.decode("utf-8"))

    def step(self, actions: dict) -> dict:
        assert self.isloaded(), "the dll has not been loaded"
        actions = json.dumps(actions).encode("utf-8")
        res = self._step(actions)
        return json.loads(res.decode("utf-8"))
