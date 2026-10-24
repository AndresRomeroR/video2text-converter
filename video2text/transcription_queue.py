"""Sequential transcription queue without dependencies on Tk or Whisper."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional
from uuid import uuid4


@dataclass
class QueueItem:
    id: str
    path: Path
    status: str = "pending"
    elapsed: float = 0.0
    error: str = ""
    outputs: Optional[tuple[Path, Path]] = None


def _path_key(path: Path) -> str:
    return os.path.normcase(str(path.resolve()))


class BatchQueue:
    """Keep a visible history and process only pending items, in insertion order.

    The caller owns thread coordination: edit the list while idle and call
    ``run`` from a single worker. Callbacks execute on that worker's thread.
    """

    def __init__(self) -> None:
        self.items: list[QueueItem] = []

    def add(self, paths: Iterable[Path]) -> tuple[list[QueueItem], int]:
        known = {_path_key(item.path) for item in self.items}
        added = []
        duplicate_count = 0
        for path in paths:
            path = Path(path).expanduser().resolve()
            key = _path_key(path)
            if key in known:
                duplicate_count += 1
                continue
            item = QueueItem(id=uuid4().hex, path=path)
            self.items.append(item)
            added.append(item)
            known.add(key)
        return added, duplicate_count

    def remove(self, ids: Iterable[str]) -> list[QueueItem]:
        selected = set(ids)
        removed = [
            item for item in self.items
            if item.id in selected and item.status != "processing"
        ]
        removed_ids = {item.id for item in removed}
        self.items[:] = [item for item in self.items if item.id not in removed_ids]
        return removed

    def clear_finished(self) -> list[QueueItem]:
        return self.remove(
            item.id for item in self.items if item.status in ("completed", "error")
        )

    def summary(self) -> dict[str, int]:
        counts = dict.fromkeys(("pending", "processing", "completed", "error"), 0)
        for item in self.items:
            counts[item.status] += 1
        return {"total": len(self.items), **counts}

    def run(
        self,
        process: Callable[[QueueItem], tuple[Path, Path]],
        on_change: Callable[[QueueItem], None],
        should_stop: Callable[[], bool] = lambda: False,
    ) -> None:
        """Run the pending snapshot; one file's failure never skips the next.

        ``should_stop`` is checked between files, so the active transcription
        can finish and its outputs are recorded before remaining work stops.
        """
        pending = [item for item in self.items if item.status == "pending"]
        for item in pending:
            if should_stop():
                break
            item.status = "processing"
            item.elapsed = 0.0
            item.error = ""
            item.outputs = None
            on_change(item)
            started_at = time.perf_counter()
            try:
                item.outputs = process(item)
            except Exception as exc:
                item.status = "error"
                item.error = str(exc)
            else:
                item.status = "completed"
            item.elapsed = time.perf_counter() - started_at
            on_change(item)
