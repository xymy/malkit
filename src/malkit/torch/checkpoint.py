import functools
import operator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, List, Mapping, Optional, Union

import torch

from .._typing import FilePath

__all__ = ["Checkpoint"]


@dataclass
class Item:
    priority: Union[int, float]
    filename: str = field(compare=False)


class PriorityQueue:
    def __init__(self, *, capacity: int = 1, cmp: Callable = operator.lt) -> None:
        if capacity < 1:
            raise ValueError(f"require capacity >= 1, got {capacity!r}")

        self._queue: List[Item] = []
        self._capacity = capacity
        self._cmp = cmp
        self._key = functools.cmp_to_key(cmp)

    def __len__(self) -> int:
        return len(self._queue)

    def check_priority(self, priority: Union[int, float]) -> bool:
        if len(self) < self._capacity:
            return True
        return self._cmp(priority, self._queue[-1].priority)

    def update(self, priority: Union[int, float], filename: str) -> Optional[Item]:
        last = None
        if len(self) == self._capacity:
            last = self.pop()
        self.push(priority, filename)
        return last

    def push(self, priority: Union[int, float], filename: str) -> None:
        if len(self) == self._capacity:
            raise ValueError("queue is full")
        item = Item(priority, filename)
        self._queue.append(item)
        self._queue.sort(key=self._key)

    def pop(self) -> Item:
        if not self:
            raise ValueError("queue is empty")
        return self._queue.pop()


class Checkpoint:
    def __init__(self, root: FilePath, filename_prefix: str, max_save: int = 1, cmp: Callable = operator.lt) -> None:
        self.root = Path(root)
        self.filename_prefix = filename_prefix
        self.max_save = max_save
        self.queue = PriorityQueue(capacity=max_save, cmp=cmp)

    def save(
        self,
        checkpoint: Mapping[str, Any],
        metric_name: str,
        metric_value: Union[int, float],
        global_step: Optional[int] = None,
    ) -> None:
        if not self.queue.check_priority(metric_value):
            return

        filename = self.filename_prefix
        if global_step is not None:
            filename += f"_{global_step}"
        if isinstance(metric_value, int):
            filename += f"_{metric_name}={metric_value}.pt"
        else:
            filename += f"_{metric_name}={metric_value:.4f}.pt"

        state_dicts = {}
        for key, obj in checkpoint.items():
            state_dicts[key] = obj.state_dict()
        torch.save(state_dicts, self.root / filename)

        old = self.queue.update(metric_value, filename)
        if old is not None:
            oldpath = self.root / old.filename
            oldpath.unlink(missing_ok=True)

    @staticmethod
    def load(checkpoint: Mapping[str, Any], filepath: FilePath, map_location=None) -> None:
        state_dicts = torch.load(filepath, map_location=map_location)
        for key, obj in checkpoint.items():
            if key not in state_dicts:
                raise ValueError(f"{key} does not exist")
            obj.load_state_dict(state_dicts[key])
