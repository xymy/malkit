from typing import Any, Iterable, Optional

from ._parallel import execute_parallel
from ._typing import FilePath

__all__ = ["convert_bytes_to_binary", "convert_bytes_to_binary_parallel"]


def convert_bytes_to_binary(bytes_file: FilePath, binary_file: FilePath) -> None:
    """Converts bytes file to binary file."""

    with open(bytes_file, "r", encoding="ascii") as src, open(binary_file, "wb") as dst:
        for line in src:
            i = line.find(" ")
            if i < 0:
                raise ValueError(f"invalid bytes file {bytes_file!r}")
            data = line[i + 1 :].replace("??", "00")
            dst.write(bytes.fromhex(data))


def convert_bytes_to_binary_parallel(
    bytes_files: Iterable[FilePath], binary_files: Iterable[FilePath], *, n_jobs: Optional[int] = None, **kwargs: Any
) -> None:
    """Converts bytes file to binary file in parallel."""

    execute_parallel(convert_bytes_to_binary, bytes_files, binary_files, n_jobs=n_jobs, **kwargs)
