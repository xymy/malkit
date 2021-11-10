from typing import Any, Callable, Iterable, Optional

from joblib import Parallel, delayed

from ._typing import FilePath


def execute_parallel(
    function: Callable[[FilePath, FilePath], None],
    srcs: Iterable[FilePath],
    dsts: Iterable[FilePath],
    *,
    n_jobs: Optional[int] = None,
    **kwargs: Any,
) -> None:
    delayed_function = delayed(function)
    Parallel(n_jobs=n_jobs, **kwargs)(delayed_function(src, dst) for src, dst in zip(srcs, dsts))
