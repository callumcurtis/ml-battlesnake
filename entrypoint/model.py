import pathlib


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
