"""Optional deletion of old report sessions from reports/ (evidence.png, data.json, report.pdf).

Disabled unless REPORT_RETENTION_HOURS is set to a positive number, because deleting stored
reports is not something to start doing silently on upgrade.
"""
import os
import re
import shutil
import time
from typing import List, Optional

# Only folders named like a session id (uuid4) are ever touched, never anything else in reports/.
_SESSION_DIR = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")


def retention_hours() -> float:
    try:
        return float(os.environ.get("REPORT_RETENTION_HOURS", "0"))
    except ValueError:
        return 0.0


def _newest_mtime(path: str) -> float:
    newest = os.path.getmtime(path)
    for dirpath, _dirs, files in os.walk(path):
        for name in files:
            try:
                newest = max(newest, os.path.getmtime(os.path.join(dirpath, name)))
            except OSError:
                pass
    return newest


def purge_old_reports(root: str, max_age_hours: float, now: Optional[float] = None) -> List[str]:
    """Delete session folders under `root` whose newest file is older than `max_age_hours`.
    Returns the deleted folder names. max_age_hours <= 0 deletes nothing."""
    if not max_age_hours or max_age_hours <= 0 or not os.path.isdir(root):
        return []
    now = time.time() if now is None else now
    cutoff = now - max_age_hours * 3600
    deleted = []
    for entry in os.scandir(root):
        if not entry.is_dir(follow_symlinks=False) or not _SESSION_DIR.match(entry.name):
            continue
        try:
            if _newest_mtime(entry.path) < cutoff:
                shutil.rmtree(entry.path)
                deleted.append(entry.name)
        except OSError as e:
            print(f"Retention: could not remove {entry.name}: {e}")
    return deleted
