# src/classifier.py
# ─────────────────────────────────────────────────────────────────────────────
# Call type classification and topic cluster assignment.
#
# WHY HYBRID (rule-based + LLM-tag reuse)?
# ─────────────────────────────────────────
# We looked at three options:
#
#   Option A — Pure unsupervised clustering (KMeans / DBSCAN on embeddings)
#     Problem: 100 records is too small. Clusters are noise-sensitive.
#     You get different results on each run. Hard to explain to leadership.
#
#   Option B — Pure LLM classification (one API call per meeting)
#     Problem: 100 API calls minimum before validating the taxonomy.
#     Expensive before you know if the categories are right.
#     Also: the dataset already HAS LLM-extracted topic tags in summary.json.
#     Ignoring them and paying again is wasteful.
#
#   Option C — Hybrid (our choice)
#     Step 1: Rule-based call type (deterministic, zero LLM cost)
#     Step 2: Keyword scoring over title + existing LLM topic tags + summary
#     The existing topic tags ARE the first-pass LLM layer. We reuse them.
#     Fast, explainable, and traceable. Every decision maps to a keyword hit.
#
# At 10k+ meetings: swap step 2 for embedding-based clustering with
# LangSmith tracing and a validation eval set.
# ─────────────────────────────────────────────────────────────────────────────

from typing import Optional
from .taxonomy import TOPIC_TAXONOMY, COMPETITORS


def classify_call_type(title: str, emails: list[str]) -> str:
    """
    Determine whether a meeting is a support, external, or internal call.

    Decision logic (in order):
      1. Title contains 'support case', 'escalation', or 'urgent:' → support
      2. Title contains 'aegis /' → external  (account manager calling a customer)
      3. All participants are @aegiscloud.com → internal
      4. Multiple email domains present → external  (customer is on the call)
      5. Default → internal

    This is intentionally deterministic. No LLM needed here — the naming
    conventions are consistent enough that rules work perfectly.
    """
    t = title.lower()
    e = [x.lower() for x in emails]

    if "support case" in t or "escalation" in t or "urgent:" in t:
        return "support"

    if "aegis /" in t:
        return "external"

    aegis_count = sum(1 for x in e if "aegiscloud.com" in x)
    if aegis_count == len(e):
        return "internal"

    domains = set(x.split("@")[1] for x in e if "@" in x)
    if len(domains) > 1:
        return "external"

    return "internal"


def classify_topic(title: str, topics: list[str], summary: str) -> tuple[str, dict]:
    """
    Assign a topic cluster using keyword scoring over three text sources:
      - Meeting title         (highest signal — curated by organizer)
      - LLM-extracted topics  (already curated signal from the dataset)
      - Summary text          (first 500 chars for context)

    The cluster with the most keyword hits wins. Ties break by taxonomy
    order — higher-urgency clusters are listed first.

    Returns: (cluster_name, {cluster: score, ...})
    """
    text = (title + " " + " ".join(topics) + " " + summary[:500]).lower()

    scores = {}
    for cluster, cfg in TOPIC_TAXONOMY.items():
        score = sum(1 for kw in cfg["keywords"] if kw in text)
        if score > 0:
            scores[cluster] = score

    if not scores:
        return "General Business", {}

    return max(scores, key=scores.get), scores


def extract_competitors(summary: str, key_moments: list[dict], sentences: list[dict]) -> list[str]:
    """
    Find which competitors are mentioned anywhere in the meeting.

    Searches across summary, key moment texts, and first 40 transcript
    sentences (enough context without processing the full transcript).
    """
    text = (
        summary + " "
        + " ".join(km.get("text", "") for km in key_moments) + " "
        + " ".join(s.get("sentence", "") for s in sentences[:40])
    ).lower()

    return [c for c in COMPETITORS if c in text]


def extract_company(title: str) -> Optional[str]:
    """
    Pull the customer company name out of a meeting title.

    Handles the two naming patterns in this dataset:
      'Aegis / Company Name - Topic'  → returns 'Company Name'
      'Support Case #XXXX - Company Name Topic' → returns first 3 words after dash
    """
    if "Aegis /" in title:
        return title.split("Aegis / ")[1].split(" - ")[0].strip()

    for prefix in ["Support Case #", "ESCALATION: ", "URGENT: ", "INCIDENT: "]:
        if prefix in title:
            rest = title.replace(prefix, "")
            parts = rest.split(" - ")
            if len(parts) > 1:
                words = parts[1].split()
                return " ".join(words[:3])

    return None
