import collections
import collections.abc
from typing import Union, Iterator

from environment.types import InitialState, Timestep


class MemoryBuffer(collections.abc.Sequence):

    def __init__(
        self,
        buffer_size: int,
    ):
        self.size = buffer_size
        self._buffer = collections.deque(maxlen=buffer_size)
    
    def reset(self, initial_state: InitialState) -> None:
        self._buffer.clear()
        self._buffer.append(initial_state)

    def add(self, timestep: Timestep) -> None:
        self._buffer.append(timestep)

    def __getitem__(self, index: int) -> Union[InitialState, Timestep]:
        return self._buffer[index]

    def __len__(self) -> int:
        return len(self._buffer)

    def __iter__(self) -> Iterator[Union[InitialState, Timestep]]:
        return iter(self._buffer)

    def __reversed__(self) -> Iterator[Union[InitialState, Timestep]]:
        return reversed(self._buffer)

    def __contains__(self, item: Union[InitialState, Timestep]) -> bool:
        return item in self._buffer

    def index(self, item: Union[InitialState, Timestep]) -> int:
        return self._buffer.index(item)

    def count(self, item: Union[InitialState, Timestep]) -> int:
        return self._buffer.count(item)
