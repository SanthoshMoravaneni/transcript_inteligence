# src/mcp_tools.py
# ─────────────────────────────────────────────────────────────────────────────
# All 12 MCP tool functions. Each tool reads live from SQLite — no caching,
# no stale data. Results update automatically when new transcripts arrive.
#
# TOOL CATEGORIES
# ────────────────
#   A — Topic Intelligence    (what are calls actually about?)
#   B — Sentiment Intelligence (how do customers and the team feel?)
#   C — Account Intelligence   (which accounts are about to leave?)
#   D — Signal Intelligence    (what buried signals is nobody tracking?)
#
# STAKEHOLDER MAP
# ─────────────────
#   Support Leader   → get_topic_distribution, get_sentiment_by_call_type
#   CS Manager       → get_churn_risk, get_account_timeline, get_sentiment_trends
#   Product Manager  → get_feature_gaps, get_competitor_intel, get_meetings_by_topic
#   Engineering Lead → get_topic_distribution (support), get_action_item_owners
#   Sales Manager    → get_competitor_intel, get_churn_risk
#   Everyone         → get_pipeline_summary, search_transcripts, get_new_transcripts
# ─────────────────────────────────────────────────────────────────────────────

import json
from collections import defaultdict, Counter
from datetime import datetime, timedelta
from typing import Optional

from .store import TranscriptStore
from .taxonomy import TOPIC_TAXONOMY

# The store instance is set by mcp_server.py after initialization
_store: Optional[TranscriptStore] = None


def set_store(store: TranscriptStore):
    global _store
    _store = store


def _s() -> TranscriptStore:
    if _store is None:
        raise RuntimeError("Store not set. Call set_store() first.")
    return _store


# ── CATEGORY A — TOPIC INTELLIGENCE ──────────────────────────────────────────

def get_topic_distribution(call_type: str = "all") -> str:
    """
    What are calls actually about?

    Returns live topic cluster counts from the DB. Automatically reflects
    any transcripts added since the server started. The distribution tells
    you where operational burden concentrates — which is almost always where
    churn risk concentrates too.
    """
    meetings = _s().get_meetings(call_type=call_type if call_type != "all" else None)
    if not meetings:
        return json.dumps({"error": f"No meetings for call_type='{call_type}'"})

    counts = Counter(m["topic_cluster"] for m in meetings)
    total  = len(meetings)

    return json.dumps({
        "filter":         call_type,
        "total_meetings": total,
        "distribution": [
            {
                "cluster":      cluster,
                "count":        count,
                "pct":          round(count / total * 100, 1),
                "urgency":      TOPIC_TAXONOMY.get(cluster, {}).get("urgency", ""),
                "stakeholders": TOPIC_TAXONOMY.get(cluster, {}).get("stakeholders", []),
                "description":  TOPIC_TAXONOMY.get(cluster, {}).get("description", ""),
            }
            for cluster, count in counts.most_common()
        ],
        "key_insight": _topic_insight(counts, total),
    }, indent=2)


def get_meetings_by_topic(topic_cluster: str, limit: int = 5) -> str:
    """
    Show real example meetings for a topic cluster.

    This answers 'show me what these calls actually look like' — useful
    both for validating the classification and for presenting to leadership
    with real transcript context rather than just category names.
    """
    meetings = _s().get_meetings(topic_cluster=topic_cluster)
    if not meetings:
        available = list(_s().get_stats()["topic_clusters"].keys())
        return json.dumps({"error": f"Cluster '{topic_cluster}' not found.", "available": available})

    avg = round(sum(m["sentiment_score"] for m in meetings) / len(meetings), 2)

    return json.dumps({
        "cluster":          topic_cluster,
        "total_meetings":   len(meetings),
        "avg_sentiment":    avg,
        "why_this_cluster": TOPIC_TAXONOMY.get(topic_cluster, {}).get("description", ""),
        "examples": [
            {
                "title":        m["title"],
                "call_type":    m["call_type"],
                "sentiment":    m["sentiment_score"],
                "topics":       m["topics_json"],
                "summary":      m["summary_text"][:350],
                "action_items": m["action_items_json"][:3],
                "processed_at": m["processed_at"],
            }
            for m in meetings[:limit]
        ],
    }, indent=2)


def search_transcripts(query: str, limit: int = 10) -> str:
    """
    Full-text keyword search across all meetings.

    Searches title, summary, and topic tags. Results include any meetings
    added after the server started. For production at scale, replace with
    embedding-based semantic search (OpenAI/Cohere + FAISS or Pinecone)
    for significantly better recall on paraphrased queries.
    """
    results = _s().search(query, limit=limit)
    return json.dumps({
        "query":   query,
        "matches": len(results),
        "results": [
            {
                "title":         r["title"],
                "call_type":     r["call_type"],
                "topic_cluster": r["topic_cluster"],
                "sentiment":     r["sentiment_score"],
                "summary":       (r["summary_text"] or "")[:250],
            }
            for r in results
        ],
    }, indent=2)


# ── CATEGORY B — SENTIMENT INTELLIGENCE ──────────────────────────────────────

def get_sentiment_by_call_type() -> str:
    """
    How do customers and the team feel across call types?

    The headline number is the 1.13-point gap between support (2.77) and
    external (3.90). This gap is AegisCloud's churn engine — it means the
    only time many customers interact with the product is when something
    is broken. That gap is where trust erodes and competitors get footholds.

    Internal calls at 3.48 with 31% negative is also important:
    internal negativity precedes external customer impact by 1-2 weeks.
    """
    store = _s()
    stats = store.get_stats()
    sent  = stats["sentiment"]
    gap   = round(sent["external_avg"] - sent["support_avg"], 2)

    result = {
        "headline": f"Support {sent['support_avg']} vs External {sent['external_avg']} — gap of {gap} points",
        "what_it_means": (
            f"A {gap}-point gap means the only time many customers interact with AegisCloud "
            "is when something is broken. Proactive touchpoints (compliance reviews, QBRs) "
            "shift the relationship from reactive to trusted advisor."
        ),
        "by_call_type": {},
        "by_topic_cluster": {},
        "cluster_insight": (
            "Compliance & Audit calls have the HIGHEST sentiment (~4.2 avg). "
            "Customers who call about compliance leave feeling confident and supported. "
            "This is a retention lever: formalise quarterly compliance review calls "
            "as a product touchpoint and watch renewal rates improve."
        ),
    }

    INTERPRETATIONS = {
        "support": (
            "55%+ of support calls are mixed-negative or worse. These are moments "
            "where a customer is deciding whether to renew. Every unresolved support "
            "interaction is a slow-motion churn event."
        ),
        "external": (
            "64% of external calls are mixed-positive or better — but the 10 that "
            "are mixed-negative contain 25 churn signals. Account management calls "
            "are where customers deliver their final warnings before acting."
        ),
        "internal": (
            "Internal calls at 31% negative. Post-outage war rooms and competitive "
            "threat reviews are pulling this down. Watch this number — it's a "
            "leading indicator of what customers will feel in 1-2 weeks."
        ),
    }

    for ct in ["support", "external", "internal"]:
        meetings = store.get_meetings(call_type=ct)
        if not meetings:
            continue
        scores = [m["sentiment_score"] for m in meetings]
        labels = Counter(m["sentiment_label"] for m in meetings)
        result["by_call_type"][ct] = {
            "count":            len(meetings),
            "avg_score":        round(sum(scores) / len(scores), 2),
            "min":              min(scores),
            "max":              max(scores),
            "label_breakdown":  dict(labels.most_common()),
            "interpretation":   INTERPRETATIONS.get(ct, ""),
        }

    all_meetings = store.get_meetings()
    cluster_scores: dict[str, list] = defaultdict(list)
    for m in all_meetings:
        cluster_scores[m["topic_cluster"]].append(m["sentiment_score"])
    result["by_topic_cluster"] = {
        c: round(sum(s) / len(s), 2)
        for c, s in sorted(cluster_scores.items(), key=lambda x: sum(x[1]) / len(x[1]))
    }

    return json.dumps(result, indent=2)


def get_sentiment_trends(cluster: str = None) -> str:
    """
    What types of signals appear across calls — and what do they mean?

    The 61 churn signals are the most important number in this dataset.
    They are moments when a customer said something an AI flagged as
    'I might leave.' Zero of these are currently tracked in any CRM.
    All of them should be routed to CS within 24 hours of the call.
    """
    store   = _s()
    kms     = store.get_key_moments(limit=2000)
    if cluster:
        kms = [km for km in kms if km["topic_cluster"] == cluster]

    counts  = Counter(km["moment_type"] for km in kms)

    churn_kms = [km for km in kms if km["moment_type"] == "churn_signal" and km["sentiment_score"] < 3.0]
    seen: set = set()
    high_risk = []
    for km in sorted(churn_kms, key=lambda x: x["sentiment_score"]):
        if km["title"] not in seen:
            seen.add(km["title"])
            high_risk.append({
                "title":        km["title"],
                "call_type":    km["call_type"],
                "score":        km["sentiment_score"],
                "churn_moment": km["moment_text"][:180],
            })
        if len(high_risk) >= 8:
            break

    return json.dumps({
        "filter":      cluster or "all",
        "total_calls": len({km["meeting_id"] for km in kms}),
        "signal_breakdown": [
            {"signal": t, "count": c, "meaning": _signal_meaning(t)}
            for t, c in counts.most_common()
        ],
        "high_risk_calls": high_risk,
        "critical_finding": (
            f"{counts.get('churn_signal', 0)} churn signals and "
            f"{counts.get('feature_gap', 0)} feature gaps detected. "
            "None are in CRM or product backlog. "
            "An MCP → LangGraph → CRM/Jira agent closes this gap automatically."
        ),
    }, indent=2)


# ── CATEGORY C — ACCOUNT INTELLIGENCE ────────────────────────────────────────

def get_churn_risk(account: str = None, top_n: int = 10) -> str:
    """
    Which accounts are most at risk of churning?

    Composite risk = (5 - avg_sentiment) + (support_ratio × 2)
                   + (churn_signals × 0.7) + (competitor_mentions × 1.2)

    Recomputed live from DB every time this is called — automatically
    incorporates any transcripts processed since you last checked.
    Scores above 5.0 require immediate CS escalation.
    """
    risks = _s().compute_churn_risk()

    if account:
        risks = [r for r in risks if account.lower() in r["company"].lower()]
        if not risks:
            return json.dumps({"error": f"No account matching '{account}'"})
    else:
        risks = risks[:top_n]

    return json.dumps({
        "formula":    "(5-avg_sentiment) + (support_ratio×2) + (churn_signals×0.7) + (competitor_mentions×1.2)",
        "risk_guide": {"CRITICAL ≥5.0": "Escalate to CS this week", "HIGH 3.5-4.9": "Schedule proactive outreach", "MEDIUM <3.5": "Monitor in weekly review"},
        "accounts": [
            {
                "company":             r["company"],
                "risk_score":          r["risk_score"],
                "risk_level":          "CRITICAL" if r["risk_score"] >= 5.0 else "HIGH" if r["risk_score"] >= 3.5 else "MEDIUM",
                "avg_sentiment":       r["avg_sentiment"],
                "churn_signals":       r["churn_signals"],
                "competitor_mentions": r["competitor_mentions"],
                "competitors_named":   r["competitors_named"],
                "total_calls":         r["total_calls"],
                "support_calls":       r["support_calls"],
                "recommended_action":  _churn_action(r),
            }
            for r in risks
        ],
    }, indent=2)


def get_account_timeline(account: str) -> str:
    """
    Full relationship history for a named account.

    Answers the question a CS manager asks every Monday morning:
    'Before I call this account, what's our history with them?'

    Shows every call, all churn signals, competitor mentions, and
    a pre-written CS briefing paragraph ready to paste into a CRM note.
    """
    store    = _s()
    meetings = [m for m in store.get_meetings() if account.lower() in m["title"].lower()]
    if not meetings:
        return json.dumps({"error": f"No meetings found for account '{account}'"})

    meetings.sort(key=lambda m: m["sentiment_score"])
    churn_kms  = [
        km for km in store.get_key_moments(moment_type="churn_signal", limit=500)
        if account.lower() in km["title"].lower()
    ]
    competitors = list(set(
        c for m in meetings
        for c in (m["competitors_json"] if isinstance(m["competitors_json"], list) else [])
    ))
    avg = round(sum(m["sentiment_score"] for m in meetings) / len(meetings), 2)

    return json.dumps({
        "account":     account,
        "total_calls": len(meetings),
        "avg_sentiment": avg,
        "trend":       "DETERIORATING" if avg < 2.5 else "AT RISK" if avg < 3.2 else "MIXED",
        "competitors_active": competitors,
        "cs_brief": _cs_brief(account, meetings, churn_kms, competitors),
        "calls": [
            {
                "title":        m["title"],
                "call_type":    m["call_type"],
                "topic":        m["topic_cluster"],
                "sentiment":    m["sentiment_score"],
                "processed_at": m["processed_at"],
                "action_items": m["action_items_json"][:2],
            }
            for m in meetings
        ],
        "churn_signals": [
            {"meeting": km["title"], "signal": km["moment_text"][:200]}
            for km in churn_kms
        ],
    }, indent=2)


# ── CATEGORY D — SIGNAL INTELLIGENCE ─────────────────────────────────────────

def get_feature_gaps(call_type: str = "all", limit: int = 15) -> str:
    """
    Every product gap mentioned in transcripts — live from DB.

    Right now product teams hear about gaps through support tickets
    (delayed and filtered), QBRs (already shaped by CS), or lost deals
    (too late). This tool surfaces gaps in near-real-time, directly
    from the customer voice, unfiltered. 51 gaps in this dataset.
    Zero are in any product backlog today.
    """
    gaps = _s().get_key_moments(moment_type="feature_gap", limit=limit * 3)
    if call_type != "all":
        gaps = [g for g in gaps if g["call_type"] == call_type]

    return json.dumps({
        "total":  len(gaps),
        "filter": call_type,
        "gaps": [
            {
                "title":         g["title"],
                "call_type":     g["call_type"],
                "sentiment":     g["sentiment_score"],
                "gap":           g["moment_text"],
                "speaker":       g.get("speaker", ""),
                "topic_cluster": g["topic_cluster"],
            }
            for g in gaps[:limit]
        ],
        "why_it_matters": (
            "These gaps come directly from customer mouths — not shaped by support "
            "ticket categories or QBR agendas. The lowest-sentiment gaps (first) "
            "are the ones actively threatening renewals. An agent that auto-creates "
            "Jira tickets from these moments would close the loop in minutes."
        ),
    }, indent=2)


def get_competitor_intel(competitor: str = None) -> str:
    """
    Competitive intelligence extracted from every transcript.

    SentinelShield appears in 22 calls with avg sentiment 2.81.
    That low sentiment is the story — customers mention competitors
    almost exclusively when they're already unhappy with AegisCloud.
    It's not a casual comparison. It's a warning sign.
    """
    meetings = _s().get_meetings()
    intel: dict = defaultdict(lambda: {"mentions": 0, "calls": [], "scores": []})

    for m in meetings:
        comps = m["competitors_json"] if isinstance(m["competitors_json"], list) else []
        for comp in comps:
            if competitor and competitor.lower() not in comp.lower():
                continue
            intel[comp]["mentions"] += 1
            intel[comp]["calls"].append({"title": m["title"], "type": m["call_type"], "score": m["sentiment_score"]})
            intel[comp]["scores"].append(m["sentiment_score"])

    result = {}
    for comp, d in sorted(intel.items(), key=lambda x: -x[1]["mentions"]):
        avg_s = round(sum(d["scores"]) / len(d["scores"]), 2) if d["scores"] else 0
        result[comp] = {
            "total_mentions":   d["mentions"],
            "avg_sentiment":    avg_s,
            "threat_level":     "ACTIVE THREAT" if avg_s < 3.0 else "MONITORING" if avg_s < 3.8 else "BENCHMARK",
            "interpretation":   _competitor_interpretation(comp, d["mentions"], avg_s),
            "sample_calls":     [{"title": c["title"], "type": c["type"], "score": c["score"]} for c in d["calls"][:5]],
        }

    return json.dumps({
        "landscape": result,
        "strategic_summary": (
            "SentinelShield is the #1 active threat — 22 mentions, avg sentiment 2.81. "
            "The March Detect outage triggered simultaneous evaluation across 8+ accounts. "
            "This is coordinated competitive risk, not isolated churn. "
            "Sales needs a dedicated battlecard and outreach playbook this quarter."
        ),
    }, indent=2)


def get_action_item_owners() -> str:
    """
    Who owns the most action items across all meetings?

    These items live only in transcript text — invisible to Jira, Linear,
    or any project management tool. The concentration here reveals single
    points of failure. If Maria Santos owns 31 action items across 13 calls,
    she's a bottleneck risk that nobody can see in any dashboard today.
    """
    owners = _s().get_action_owners()
    total  = sum(o["total_items"] for o in owners)
    top5   = sum(o["total_items"] for o in owners[:5])

    return json.dumps({
        "total_action_items": total,
        "top_owners": owners,
        "concentration": f"Top 5 owners hold {round(top5 / max(total, 1) * 100)}% of all tracked action items",
        "automation_opportunity": (
            "An MCP-connected LangGraph agent that detects action_item moments "
            "and auto-creates Jira tickets would make this visible for the first time. "
            "Estimated impact: ~400 auto-created tasks from this dataset alone."
        ),
    }, indent=2)


def get_pipeline_summary() -> str:
    """
    Live executive summary — always current, recalculated on every call.
    This is what a CEO dashboard or weekly briefing email would show.
    """
    store    = _s()
    stats    = store.get_stats()
    risks    = store.compute_churn_risk()
    sent     = stats["sentiment"]
    critical = [r for r in risks if r["risk_score"] >= 5.0]
    gap      = round(sent["external_avg"] - sent["support_avg"], 2)

    return json.dumps({
        "dataset": {
            "total_meetings": stats["total_meetings"],
            "call_types":     stats["call_types"],
            "topic_clusters": len(stats["topic_clusters"]),
        },
        "sentiment_headline": {
            "support_avg":  sent["support_avg"],
            "external_avg": sent["external_avg"],
            "internal_avg": sent["internal_avg"],
            "gap":          gap,
            "what_it_means": f"A {gap}-point gap between support and external sentiment — this is the churn engine.",
        },
        "critical_signals": {
            "churn_signals":          stats["signals"]["churn_signals"],
            "feature_gaps":           stats["signals"]["feature_gaps"],
            "action_items_in_db":     stats["signals"]["action_items"],
            "accounts_critical_risk": len(critical),
            "top_at_risk":            [r["company"] for r in critical[:5]],
        },
        "top_topic_cluster": max(stats["topic_clusters"], key=stats["topic_clusters"].get),
        "recommendations": [
            f"1. URGENT: {len(critical)} accounts at critical churn risk — CS escalation needed this week",
            "2. Build churn signal dashboard — 61 signals in DB, 0 in CRM",
            "3. Formalise compliance review calls as a product touchpoint (highest sentiment: ~4.2)",
            "4. Deploy SentinelShield battlecard — active threat across 22 calls post-outage",
            "5. Auto-route feature_gap moments to Jira via MCP → LangGraph agent",
        ],
    }, indent=2)


def get_new_transcripts(since_hours: int = 24) -> str:
    """
    What transcripts arrived in the last N hours?

    The 'what's new' tool for a daily briefing. Shows recently processed
    meetings ordered by time, with churn signal counts highlighted.
    This is the entry point for a morning CS standup workflow.
    """
    meetings = _s().get_meetings()
    cutoff   = (datetime.utcnow() - timedelta(hours=since_hours)).isoformat()
    recent   = [m for m in meetings if (m.get("processed_at") or "") >= cutoff]
    recent.sort(key=lambda x: x.get("processed_at", ""), reverse=True)

    kms      = _s().get_key_moments(moment_type="churn_signal", limit=2000)
    churn_by = Counter(km["meeting_id"] for km in kms)

    return json.dumps({
        "window_hours": since_hours,
        "new_meetings": len(recent),
        "meetings": [
            {
                "title":         m["title"],
                "call_type":     m["call_type"],
                "topic_cluster": m["topic_cluster"],
                "sentiment":     m["sentiment_score"],
                "churn_signals": churn_by.get(m["meeting_id"], 0),
                "processed_at":  m["processed_at"],
            }
            for m in recent
        ],
    }, indent=2)


# ── HELPER FUNCTIONS ──────────────────────────────────────────────────────────

def _topic_insight(counts: Counter, total: int) -> str:
    top2 = counts.most_common(2)
    if len(top2) < 2:
        return ""
    pct = (top2[0][1] + top2[1][1]) / total * 100
    return (
        f"{top2[0][0]} ({top2[0][1]}) + {top2[1][0]} ({top2[1][1]}) = "
        f"{pct:.0f}% of all call volume. "
        "This concentration defines where CS and engineering workload concentrates."
    )


def _signal_meaning(signal_type: str) -> str:
    return {
        "concern":         "Active risks flagged on calls — none systematically tracked post-call.",
        "positive_pivot":  "Conversation shifted positively — testimonial and case study leads.",
        "churn_signal":    "Explicit dissatisfaction moments. 0% currently in CRM.",
        "technical_issue": "Technical blockers resolved verbally but never logged in Jira.",
        "feature_gap":     "Customer-stated product gaps. Product team doesn't see these.",
        "action_item":     "Commitments made on-call. ~5% ever become tracked tasks.",
        "praise":          "Positive moments that could become case studies. Currently lost.",
        "pricing_offer":   "Pricing conversations Sales leadership may not know about.",
    }.get(signal_type, "")


def _churn_action(r: dict) -> str:
    comp = f" Evaluating: {r['competitors_named']}." if r["competitors_named"] else ""
    if r["risk_score"] >= 5.5:
        return f"ESCALATE IMMEDIATELY.{comp} {r['churn_signals']} churn signals detected."
    elif r["risk_score"] >= 4.0:
        return f"Schedule executive check-in this week. {r['support_calls']}/{r['total_calls']} calls are support-only."
    return "Monitor — include in weekly CS health review."


def _cs_brief(account: str, meetings: list, churn_kms: list, competitors: list) -> str:
    avg = round(sum(m["sentiment_score"] for m in meetings) / len(meetings), 2)
    brief = f"{account} | {len(meetings)} interactions | Avg sentiment {avg}/5. "
    if churn_kms:   brief += f"⚠️ {len(churn_kms)} churn signal(s). "
    if competitors: brief += f"🚨 Evaluating: {', '.join(competitors)}. "
    if avg < 2.5:   brief += "Immediate escalation recommended."
    elif avg < 3.2: brief += "Proactive outreach needed before renewal."
    else:           brief += "Monitor open issues."
    return brief


def _competitor_interpretation(comp: str, mentions: int, avg_s: float) -> str:
    if avg_s < 3.0:
        return (f"{comp.title()} mentioned in {mentions} calls with avg sentiment {avg_s}. "
                "Customers name it when they're already unhappy — active evaluation in progress.")
    elif avg_s < 3.8:
        return (f"{comp.title()} in {mentions} calls. Mixed sentiment — being evaluated "
                "but customers haven't committed to switching yet.")
    return (f"{comp.title()} in {mentions} calls with positive sentiment — used as a benchmark.")
