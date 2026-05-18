# engine.py
# ─────────────────────────────────────────────────────────────────────────────
# Orchestrates the full pipeline: detect → process → store → serve.
# This is the main entry point you run to start the system.
#
# USAGE
# ──────
# Process all existing transcripts once:
#   python engine.py ./dataset
#
# Process existing + watch for new arrivals (keeps running):
#   python engine.py ./dataset --watch
#
# Force reprocess everything even if already in DB:
#   python engine.py ./dataset --force
# ─────────────────────────────────────────────────────────────────────────────

import sys
import time
import logging
from pathlib import Path

from src.store import TranscriptStore
from src.processor import TranscriptProcessor
from src.watcher import TranscriptWatcher

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("engine")


class TranscriptEngine:
    """
    Ties together Store, Processor, and Watcher.

    When a new transcript folder appears:
      Watcher detects it → Engine.ingest_one() → Processor converts it →
      Store inserts into SQLite → MCP tools see updated data immediately.
    """

    def __init__(self, dataset_dir: str, db_path: str = "transcripts.db"):
        self.dataset_dir = Path(dataset_dir)
        self.store       = TranscriptStore(db_path)
        self.watcher     = TranscriptWatcher(
            dataset_dir=dataset_dir,
            on_new_transcript=self.ingest_one,
        )

    def ingest_all(self, force: bool = False) -> dict:
        """
        Process every transcript folder in the dataset directory.

        Skips folders whose content hash matches what's already in the DB.
        This means you can safely re-run at any time — it only processes
        folders that are new or have changed.

        Set force=True to reprocess everything regardless of hash.
        """
        folders = [
            p for p in self.dataset_dir.iterdir()
            if p.is_dir() and not p.name.startswith(".")
        ]

        processed = skipped = errors = 0

        for folder in folders:
            try:
                fhash = TranscriptStore.folder_hash(folder)
                if not force and self.store.is_processed(folder.name, fhash):
                    skipped += 1
                    continue

                meeting = TranscriptProcessor.process(folder)
                if meeting:
                    self.store.upsert(meeting)
                    processed += 1
                    log.info(
                        f"  ✓  [{meeting['call_type']:8s}] "
                        f"{meeting['topic_cluster'][:35]:35s}  "
                        f"{meeting['title'][:50]}"
                    )
                else:
                    errors += 1

            except Exception as e:
                log.error(f"  ✗  {folder.name}: {e}")
                errors += 1

        stats = self.store.get_stats()
        log.info(
            f"\nIngest complete: {processed} new  |  "
            f"{skipped} unchanged  |  {errors} errors  |  "
            f"{stats['total_meetings']} total in DB"
        )
        return stats

    def ingest_one(self, folder: Path, force: bool = False):
        """Process a single folder. Called by the watcher for each new arrival."""
        try:
            fhash = TranscriptStore.folder_hash(folder)
            if not force and self.store.is_processed(folder.name, fhash):
                log.info(f"Skipping {folder.name} — already processed")
                return

            meeting = TranscriptProcessor.process(folder)
            if meeting:
                self.store.upsert(meeting)
                log.info(
                    f"New transcript processed: "
                    f"[{meeting['call_type']}] {meeting['title']}"
                )
        except Exception as e:
            log.error(f"Error processing {folder}: {e}")

    def watch(self):
        """Start watching for new transcripts. Runs until Ctrl+C."""
        self.watcher.start()
        log.info("Watching for new transcripts. Press Ctrl+C to stop.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            log.info("Shutting down watcher.")
            self.watcher.stop()

    def run(self, force: bool = False, watch: bool = False):
        log.info(f"Dataset: {self.dataset_dir}")
        log.info(f"Database: {self.store.db_path}")
        log.info("─" * 60)
        self.ingest_all(force=force)
        if watch:
            self.watch()


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args       = sys.argv[1:]
    positional = [a for a in args if not a.startswith("--")]
    dataset    = positional[0] if positional else "./dataset"
    db         = positional[1] if len(positional) > 1 else "transcripts.db"
    force      = "--force" in args
    watch      = "--watch" in args

    engine = TranscriptEngine(dataset_dir=dataset, db_path=db)
    engine.run(force=force, watch=watch)
