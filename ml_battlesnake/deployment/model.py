import abc
import os
import pathlib
import subprocess
import logging
import atexit

import psutil


logger = logging.getLogger(__name__)


class Route(abc.ABC):

    @property
    @abc.abstractmethod
    def baseroute(self):
        pass


class Snake(Route):

    def __init__(self, name: str, baseroute: str) -> None:
        self._name = name
        self._baseroute = baseroute

    @property
    def name(self):
        return self._name

    @property
    def baseroute(self):
        return self._baseroute
    
    def __str__(self) -> str:
        return f"Snake(name={self.name}, baseroute={self.baseroute})"


class BrowserSpectator(Route):

    def __init__(
        self,
        name: str,
        baseroute: str,
    ) -> None:
        self._name = name
        self._baseroute = baseroute
    
    @property
    def name(self):
        return self._name
    
    @property
    def baseroute(self):
        return self._baseroute

    def __str__(self) -> str:
        return f"BrowserSpectator(name={self.name}, baseroute={self.baseroute})"


class Engine(Route):
    
        def __init__(
            self,
            name: str,
            baseroute: str,
        ) -> None:
            self._name = name
            self._baseroute = baseroute
        
        @property
        def name(self):
            return self._name
        
        @property
        def baseroute(self):
            return self._baseroute
    
        def __str__(self) -> str:
            return f"Engine(name={self.name}, baseroute={self.baseroute})"


class Program:
    
    def __init__(
        self,
        name: str,
        entrypoint: list[str],
        cwd: pathlib.Path,
    ) -> None:
        self._name = name
        self._entrypoint = entrypoint
        self._cwd = cwd

    @property
    def name(self):
        return self._name
    
    @property
    def entrypoint(self):
        return self._entrypoint
    
    @property
    def cwd(self):
        return self._cwd
    
    def __str__(self) -> str:
        return f"Program(name={self.name}, entrypoint={self.entrypoint}, cwd={self.cwd})"


class Service:
    
    def __init__(
        self,
        name: str,
        program: Program,
        env: dict[str, str] = None,
        args: list[str] = None,
        routes: list[Route] = None,
    ) -> None:
        self._name = name
        self._program = program
        self._env = env if env is not None else {}
        self._args = args if args is not None else []
        self._routes = routes if routes is not None else []
        self._popen = None

        self._subscribe_to_exit_signal_for_cleanup()

    @property
    def name(self):
        return self._name
    
    @property
    def program(self):
        return self._program

    @property
    def routes(self):
        return self._routes

    def start(self, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) -> None:
        assert self._popen is None
        logger.debug(f"Starting {self}")
        self._popen = subprocess.Popen(
            self.program.entrypoint + self._args,
            cwd=self.program.cwd,
            env={**os.environ, **self._env},
            stdout=stdout,
            stderr=stderr,
        )
    
    def stop(self) -> None:
        assert self._popen is not None
        logger.debug(f"Stopping {self}")
        parent = psutil.Process(self._popen.pid)
        for child in parent.children(recursive=True):
            child.kill()
        parent.kill()
        self._popen = None
    
    def _subscribe_to_exit_signal_for_cleanup(self) -> None:
        def cleanup():
            if self._popen is not None:
                self.stop()
        atexit.register(cleanup)

    def __str__(self) -> str:
        return f"Service(name={self.name}, program={self.program}, env={self._env}, args={self._args}, routes={self._routes})"
