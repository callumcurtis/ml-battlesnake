import ctypes
import json
import pathlib


class Rules:

    _MOVES = ["up", "down", "left", "right"]

    def __init__(
        self,
        dll_path: pathlib.Path,
    ) -> None:
        dll = ctypes.CDLL(dll_path)

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
    
    def moves(self) -> list[str]:
        return self._MOVES

    def render(self, color: bool = True) -> None:
        self._render(1 if color else 0)

    def done(self) -> bool:
        return self._done() == 1

    def setup(self, options: dict) -> None:
        options = json.dumps(options).encode("utf-8")
        self._setup(options)

    def reset(self, options: dict) -> dict:
        options = json.dumps(options).encode("utf-8")
        res = self._reset(options)
        return json.loads(res.decode("utf-8"))

    def step(self, actions: dict) -> dict:
        actions = json.dumps(actions).encode("utf-8")
        res = self._step(actions)
        return json.loads(res.decode("utf-8"))
