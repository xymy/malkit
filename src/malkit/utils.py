import functools
from typing import Any, Iterable, Optional

from ._parallel import execute_parallel
from ._typing import FilePath

__all__ = ["convert_bytes_to_binary", "convert_bytes_to_binary_parallel"]


def convert_bytes_to_binary(bytes_file: FilePath, binary_file: FilePath, *, qq: str = "00") -> None:
    """Converts bytes file to binary file."""

    with open(bytes_file, "r", encoding="ascii") as src, open(binary_file, "wb") as dst:
        for line in src:
            i = line.find(" ")
            if i < 0:
                raise ValueError(f"invalid bytes file {bytes_file!r}")
            data = line[i + 1 :].replace("??", qq)
            dst.write(bytes.fromhex(data))


def convert_bytes_to_binary_parallel(
    bytes_files: Iterable[FilePath],
    binary_files: Iterable[FilePath],
    *,
    qq: str = "00",
    n_jobs: Optional[int] = None,
    **kwargs: Any,
) -> None:
    """Converts bytes file to binary file in parallel."""

    function = functools.partial(convert_bytes_to_binary, qq=qq)
    execute_parallel(function, bytes_files, binary_files, n_jobs=n_jobs, **kwargs)
