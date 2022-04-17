from typing import Tuple


class Loader:
    def __repr__(self) -> str:
        args = self._get_args()
        args_str = ", ".join(f"{a}={getattr(self, a)!r}" for a in args)
        return f"{type(self).__name__}({args_str})"

    def _get_args(self) -> Tuple[str, ...]:
        return ()
