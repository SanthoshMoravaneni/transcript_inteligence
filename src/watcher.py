# src/watcher.py
# ─────────────────────────────────────────────────────────────────────────────
# Watches the dataset directory for new transcript folders.
#
# HOW NEW TRANSCRIPTS FLOW THROUGH THE SYSTEM
# ─────────────────────────────────────────────
# 1. A new folder appears in the dataset directory
# 2. The watcher detects it (via OS events or polling)
# 3. It calls the on_new_transcript callback with the folder path
# 4. The engine processes it and inserts into SQLite
# 5. Every MCP tool query now sees the new data — no restart needed
#
# TWO MODES
# ─────────────────────────────────────────────
# Watchdog mode (recommended):
#   Uses OS file-system events (inotify on Linux, FSEvents on Mac).
#   Detects new folders in under 1 second. Zero CPU when idle.
#   Install: pip install watchdog
#
# Polling fallback (automatic if watchdog not installed):
#   Scans the directory every N seconds. Slightly slower but works
#   everywhere without an extra dependency.
# ─────────────────────────────────────────────────────────────────────────────

import time
import threading
import logging
from pathlib import Path
from typing import Callable

log = logging.getLogger("watcher")

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler, DirCreatedEvent
    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False


class TranscriptWatcher:
    """Detects new transcript folders and fires a callback for each one."""

    def __init__(
        self,
        dataset_dir: str,
        on_new_transcript: Callable[[Path], None],
        poll_interval: int = 10,
    ):
        self.dataset_dir       = Path(dataset_dir)
        self.on_new_transcript = on_new_transcript
        self.poll_interval     = poll_interval
        self._running          = False
        self._thread: threading.Thread = None
        self._observer = None

    def start(self):
        self._running = True
        if WATCHDOG_AVAILABLE:
            self._start_watchdog()
        else:
            log.warning(
                "watchdog not installed — using polling fallback. "
                "Run: pip install watchdog  for real-time detection."
            )
            self._start_polling()

    def stop(self):
        self._running = False
        if self._observer:
            self._observer.stop()
            self._observer.join()
        if self._thread:
            self._thread.join(timeout=5)

    def _start_watchdog(self):
        watcher = self

        class Handler(FileSystemEventHandler):
            def on_created(self, event):
                if isinstance(event, DirCreatedEvent):
                    path = Path(event.src_path)
                    log.info(f"New folder detected: {path.name}")
                    time.sleep(2)  # wait for all files to finish writing
                    watcher.on_new_transcript(path)

        self._observer = Observer()
        self._observer.schedule(Handler(), str(self.dataset_dir), recursive=False)
        self._observer.start()
        log.info(f"Watchdog active on {self.dataset_dir}")

    def _start_polling(self):
        def poll():
            seen: set[Path] = set()
            while self._running:
                try:
                    current = {
                        p for p in self.dataset_dir.iterdir()
                        if p.is_dir() and not p.name.startswith(".")
                    }
                    for folder in current - seen:
                        log.info(f"Polling detected: {folder.name}")
                        time.sleep(2)
                        self.on_new_transcript(folder)
                    seen = current
                except Exception as e:
                    log.error(f"Poll error: {e}")
                time.sleep(self.poll_interval)

        self._thread = threading.Thread(target=poll, daemon=True)
        self._thread.start()
        log.info(f"Polling active ({self.poll_interval}s) on {self.dataset_dir}")
