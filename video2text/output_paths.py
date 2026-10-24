"""Plan transcript destinations without collisions inside a visible queue."""

from __future__ import annotations

import os
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from transcription_queue import QueueItem


def _path_key(path: Path) -> str:
    return os.path.normcase(str(path.resolve()))


def _output_pair(path: Path, include_extension: bool) -> tuple[Path, Path]:
    base = path.name if include_extension else path.stem
    return path.with_name(f"{base}.txt"), path.with_name(f"{base}.srt")


def plan_output_paths(items: Iterable[QueueItem]) -> dict[str, tuple[Path, Path]]:
    """Return stable TXT/SRT paths for every item, without changing the items.

    Completed outputs keep their exact destinations. Other files normally use
    their stem; conflicting groups retain the input extension, including any
    new conflicts this creates. A numeric suffix resolves remaining conflicts
    with destinations already used by completed entries.
    """
    items = list(items)
    planned = {
        item.id: tuple(Path(path) for path in item.outputs)
        for item in items if item.status == "completed" and item.outputs is not None
    }
    reserved = {_path_key(path) for pair in planned.values() for path in pair}
    pending = [item for item in items if item.id not in planned]
    extended: set[str] = set()

    # Promoting one file to its full name can collide with another file's
    # ordinary stem (clip.mp3 and clip.mp3.mp4), so propagate until stable.
    while True:
        owners: dict[str, list[str]] = defaultdict(list)
        for item in pending:
            for path in _output_pair(item.path, item.id in extended):
                owners[_path_key(path)].append(item.id)
        conflicts = {
            item_id for key, ids in owners.items()
            if len(ids) > 1 or key in reserved
            for item_id in ids
        }
        promoted = conflicts - extended
        if not promoted:
            break
        extended.update(promoted)

    candidates = {
        item.id: _output_pair(item.path, item.id in extended) for item in pending
    }
    # Do not let a fallback suffix take another file's ordinary destination.
    preferred = {_path_key(path) for pair in candidates.values() for path in pair}
    used = set(reserved)
    for item in pending:
        pair = candidates[item.id]
        if any(_path_key(path) in used for path in pair):
            base = pair[0].stem
            index = 2
            while True:
                pair = (
                    item.path.with_name(f"{base} ({index}).txt"),
                    item.path.with_name(f"{base} ({index}).srt"),
                )
                keys = {_path_key(path) for path in pair}
                if not keys.intersection(used | preferred):
                    break
                index += 1
        planned[item.id] = pair
        used.update(_path_key(path) for path in pair)
    return planned
