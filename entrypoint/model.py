import os
import pathlib
import subprocess
import logging
import atexit
import socket
import typing

import psutil


logger = logging.getLogger(__name__)


class Snake:

    def __init__(self, name: str, path: pathlib.Path) -> None:
        self._name = name
        self._path = path

    @property
    def name(self):
        return self._name

    @property
    def path(self):
        return self._path
    
    def __str__(self) -> str:
        return f"Snake(name={self.name}, path={self.path})"


class Board:

    def __init__(
        self,
        host: str,
        port: int,
        path: pathlib.Path,
    ) -> None:
        self._host = host
        self._port = port
        self._path = path
        self._server = Server(self, port)
    
    @property
    def host(self) -> str:
        return self._host
    
    @property
    def port(self) -> int:
        return self._port
    
    @property
    def path(self) -> pathlib.Path:
        return self._path

    def start(self):
        cmd = ["npm", "start"]
        env = {**os.environ, "HOST": self._host, "PORT": str(self._port)}
        cwd = self._path.resolve()
        self._server.start(cmd, cwd=cwd, env=env)

    def stop(self):
        self._server.stop()

    def __str__(self) -> str:
        return f"Board(host={self.host}, port={self.port}, path={self._path.resolve()})"


class Server:

    def __init__(self, artifact: typing.Any, port: int) -> None:
        self._artifact = artifact
        self._port = port
        self._popen = None
        self._register_cleanup()

    def start(self, cmd: list[str], **kwargs):
        assert self._popen is None
        if not self._is_port_available():
            raise ValueError(f"Port {self._port} is not available")
        logger.debug(f"Starting {self}")
        self._popen = subprocess.Popen(cmd, **kwargs, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def _is_port_available(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(('localhost', self._port)) != 0

    def stop(self):
        assert self._popen is not None
        logger.debug(f"Stopping {self}")
        process = psutil.Process(self._popen.pid)
        for proc in process.children(recursive=True):
            proc.kill()
        process.kill()
        self._popen = None

    def _register_cleanup(self):
        def stop_if_running():
            if self._popen is not None:
                self.stop()
        atexit.register(stop_if_running)

    def __str__(self) -> str:
        return f"Server({self._artifact})"
