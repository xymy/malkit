from joblib import Parallel, delayed

__all__ = ["convert_bytes_to_binary", "convert_bytes_to_binary_parallel"]


def convert_bytes_to_binary(bytes_file, binary_file):
    """Converts bytes file to binary file."""

    with open(bytes_file, "r") as src, open(binary_file, "wb") as dst:
        for line in src:
            i = line.find(" ")
            if i < 0:
                raise ValueError(f"invalid bytes file {bytes_file!r}")
            data = line[i + 1 :].replace("??", "00")
            dst.write(bytes.fromhex(data))


def convert_bytes_to_binary_parallel(bytes_files, binary_files, *, n_jobs=None, verbose=0):
    """Converts bytes file to binary file in parallel."""

    Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(convert_bytes_to_binary)(bytes_file, binary_file)
        for bytes_file, binary_file in zip(bytes_files, binary_files)
    )
