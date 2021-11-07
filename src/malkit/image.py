from typing import Callable, Dict, Iterable, Union

from joblib import Parallel, delayed
from PIL import Image

from .typing import FilePath

__all__ = ["get_image", "convert_binary_to_image", "convert_binary_to_image_parallel"]


def _get_image(buffer: bytes, *, width: int, drop: bool = False, padding: bytes = b"\x00") -> Image.Image:
    nbytes = len(buffer)
    height, rem = divmod(nbytes, width)
    if rem != 0:
        # The image height must be at least 1.
        if drop and height > 0:
            nbytes = width * height
            buffer = buffer[:nbytes]
        else:
            height += 1
            nbytes = width * height
            buffer = buffer.ljust(nbytes, padding)
    return Image.frombuffer("L", (width, height), buffer, "raw", "L", 0, 1)


def get_image(buffer: bytes, *, width: Union[int, str], drop: bool = False, padding: bytes = b"\x00") -> Image.Image:
    if isinstance(width, str):
        width = _registry[width](len(buffer))
    return _get_image(buffer, width=width, drop=drop, padding=padding)


def convert_binary_to_image(
    binary_file: FilePath, image_file: FilePath, *, width: Union[int, str], drop: bool = False, padding: bytes = b"\x00"
) -> None:
    with open(binary_file, "rb") as f:
        buffer = f.read()
    image = get_image(buffer, width=width, drop=drop, padding=padding)
    image.save(image_file)


def convert_binary_to_image_parallel(
    binary_files: Iterable[FilePath],
    image_files: Iterable[FilePath],
    *,
    width: Union[int, str],
    drop: bool = False,
    padding: bytes = b"\x00",
    n_jobs=None,
    verbose=0,
) -> None:
    Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(convert_binary_to_image)(binary_file, image_file, width=width, drop=drop, padding=padding)
        for binary_file, image_file in zip(binary_files, image_files)
    )


def _width_function_nkjm(filesize: int) -> int:
    k = filesize // 1024
    if k < 10:
        width = 32
    elif k < 30:
        width = 64
    elif k < 60:
        width = 128
    elif k < 100:
        width = 256
    elif k < 200:
        width = 384
    elif k < 500:
        width = 512
    elif k < 1000:
        width = 768
    else:
        width = 1024
    return width


_registry: Dict[str, Callable[[int], int]] = {"nkjm": _width_function_nkjm}
