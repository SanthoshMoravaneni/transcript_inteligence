# src/processor.py
# ─────────────────────────────────────────────────────────────────────────────
# Loads one transcript folder and turns it into a structured meeting record.
# Stateless — no DB or network calls. Pure data transformation.
# ─────────────────────────────────────────────────────────────────────────────

import json
from collections import defaultdict
from pathlib import Path
from typing import Optional

from .classifier import classify_call_type, classify_topic, extract_competitors
from .store import TranscriptStore


class TranscriptProcessor:
    """Process a single transcript folder into a meeting dict."""

    @staticmethod
    def process(folder: Path) -> Optional[dict]:
        """
        Load the three JSON files from a transcript folder and
        return a fully enriched meeting dict ready for DB insertion.

        Returns None if the folder is missing required files.
        """
        try:
            info       = json.load(open(folder / "meeting-info.json"))
            summary    = json.load(open(folder / "summary.json"))
            transcript = json.load(open(folder / "transcript.json"))
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"  Skipping {folder.name}: {e}")
            return None

        sentences = transcript.get("data", [])
        title     = info.get("title", "")
        emails    = info.get("allEmails", [])
        topics    = summary.get("topics", [])
        sum_text  = summary.get("summary", "")
        key_moments = summary.get("keyMoments", [])

        # Speaker turn statistics
        sp: dict = defaultdict(lambda: {"turns": 0, "words": 0, "neg": 0, "pos": 0, "neu": 0})
        for s in sentences:
            name = s["speaker_name"]
            sp[name]["turns"] += 1
            sp[name]["words"] += len(s["sentence"].split())
            sp[name][s.get("sentimentType", "neutral")[:3]] += 1

        call_type     = classify_call_type(title, emails)
        topic_cluster, _ = classify_topic(title, topics, sum_text)
        competitors   = extract_competitors(sum_text, key_moments, sentences)

        return {
            "id":            folder.name,
            "title":         title,
            "duration":      info.get("duration", 0),
            "emails":        emails,
            "sentiment":     summary.get("overallSentiment", ""),
            "score":         summary.get("sentimentScore", 3.0),
            "topics":        topics,
            "summary_text":  sum_text,
            "action_items":  summary.get("actionItems", []),
            "key_moments":   key_moments,
            "word_count":    sum(len(s["sentence"].split()) for s in sentences),
            "speaker_stats": {k: dict(v) for k, v in sp.items()},
            "competitors":   competitors,
            "call_type":     call_type,
            "topic_cluster": topic_cluster,
            "folder_hash":   TranscriptStore.folder_hash(folder),
        }
