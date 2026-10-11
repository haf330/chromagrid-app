"""Small, thread-safe JSONL demo store; not a durable business database.

All writers in one Python process share a lock, including separate store instances.
Do not share this log between multiple processes. Reads are deliberately simple:
scan the small demo log to preserve retry protection across store recreation.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from threading import RLock
from uuid import NAMESPACE_URL, uuid5


_WRITE_LOCK = RLock()


class StorageError(Exception):
    """A record could not be safely written or read."""


class EventConflictError(StorageError):
    """A retry reused an event identifier with different data."""


def _invalid_constant(value: str):
    raise StorageError("The demo log contains a non-JSON numeric constant.")


@dataclass(frozen=True)
class WriteResult:
    record_id: str
    created: bool


class JsonlStore:
    def __init__(self, path: str | Path):
        # This path is application configuration, never a model tool argument.
        self.path = Path(path)

    def append_once(self, *, session_id: str, event_id: str, payload: dict) -> WriteResult:
        """Write once per (session, event), rejecting conflicting retry content.

        A new event can store identical content. Malformed existing logs fail
        closed: callers receive an error rather than a false recording success.
        """
        if any(not isinstance(value, str) or not value.strip()
               for value in (session_id, event_id)):
            raise StorageError("Session and event identifiers are required.")
        if not isinstance(payload, dict):
            raise StorageError("The record payload must be an object.")
        try:
            # Freeze caller-owned nested data and reject non-JSON values and NaN.
            data = json.loads(json.dumps(payload, ensure_ascii=False, allow_nan=False))
        except (TypeError, ValueError) as exc:
            raise StorageError("The record payload is not valid JSON data.") from exc
        identity = json.dumps([session_id, event_id], ensure_ascii=False)
        record_id = str(uuid5(NAMESPACE_URL, identity))
        try:
            with _WRITE_LOCK:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a+", encoding="utf-8") as stream:
                    stream.seek(0)
                    existing = None
                    for line in stream:
                        # A truncated final line must not be joined to the next write.
                        if not line.endswith("\n"):
                            raise StorageError("The demo log contains an incomplete record.")
                        row = json.loads(line, parse_constant=_invalid_constant)
                        if (not isinstance(row, dict)
                                or not isinstance(row.get("session_id"), str)
                                or not isinstance(row.get("event_id"), str)
                                or not isinstance(row.get("payload"), dict)
                                or not isinstance(row.get("record_id"), str)):
                            raise StorageError("The demo log contains an invalid record.")
                        if (row["session_id"], row["event_id"]) == (session_id, event_id):
                            if row["payload"] != data or row["record_id"] != record_id:
                                raise EventConflictError("Event already exists with different data.")
                            existing = row
                    if existing is not None:
                        return WriteResult(record_id, created=False)
                    record = {
                        "record_id": record_id,
                        "session_id": session_id,
                        "event_id": event_id,
                        "recorded_at": datetime.now(timezone.utc).isoformat(),
                        "payload": data,
                    }
                    stream.seek(0, os.SEEK_END)
                    stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                    return WriteResult(record_id, created=True)
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            # Do not expose paths or record contents in customer-facing errors.
            raise StorageError("The local demo log could not be read or written.") from exc
