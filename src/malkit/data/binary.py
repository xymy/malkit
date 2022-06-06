from typing import Tuple

from numpy.typing import NDArray

from ..features.binary import get_byte_entropy_hist, get_byte_hist, get_byte_seq
from ..typing import FilePath
from ._loader import Loader

__all__ = ["ByteSeqLoader", "ByteHistLoader", "ByteEntropyHistLoader"]


class ByteSeqLoader(Loader):
    def __init__(self, length: int, padding_value: int = 256) -> None:
        self.length = length
        self.padding_value = padding_value

    def __call__(self, path: FilePath) -> NDArray:
        with open(path, "rb") as f:
            buffer = f.read(self.length)
        return self.from_buffer(buffer)

    def from_buffer(self, buffer: bytes) -> NDArray:
        return get_byte_seq(buffer, length=self.length, padding_value=self.padding_value)

    def _get_args(self) -> Tuple[str, ...]:
        return ("length", "padding_value")


class ByteHistLoader(Loader):
    def __call__(self, path: FilePath) -> NDArray:
        with open(path, "rb") as f:
            buffer = f.read()
        return self.from_buffer(buffer)

    def from_buffer(self, buffer: bytes) -> NDArray:
        return get_byte_hist(buffer)


class ByteEntropyHistLoader(Loader):
    def __init__(self, window_size: int = 1024, step_size: int = 256) -> None:
        self.window_size = window_size
        self.step_size = step_size

    def __call__(self, path: FilePath) -> NDArray:
        with open(path, "rb") as f:
            buffer = f.read()
        return self.from_buffer(buffer)

    def from_buffer(self, buffer: bytes) -> NDArray:
        return get_byte_entropy_hist(buffer, window_size=self.window_size, step_size=self.step_size)

    def _get_args(self) -> Tuple[str, ...]:
        return ("window_size", "step_size")
