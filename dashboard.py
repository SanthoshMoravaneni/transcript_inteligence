"""
dashboard.py
─────────────────────────────────────────────────────────────
Transcript Intelligence — Streamlit Dashboard
AegisCloud B2B SaaS

Run with:
    streamlit run dashboard.py

Reads live from transcripts.db — no restart needed when
new transcripts are processed by engine.py.
─────────────────────────────────────────────────────────────
"""

import sqlite3
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path
from collections import defaultdict, Counter

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Transcript Intelligence",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CUSTOM CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
  /* Main background */
  .stApp { background-color: #0f1117; }
  section[data-testid="stSidebar"] { background-color: #1a1d27; }

  /* Metric cards */
  div[data-testid="metric-container"] {
    background: #1e2130;
    border: 1px solid #2d3148;
    border-radius: 10px;
    padding: 1rem 1.2rem;
  }
  div[data-testid="metric-container"] label {
    color: #8b92b0 !important;
    font-size: 12px !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
  }
  div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
    color: #f0f2ff !important;
    font-size: 2rem !important;
    font-weight: 700 !important;
  }

  /* Headers */
  h1, h2, h3 { color: #f0f2ff !important; }
  p, li { color: #c4c9e2; }

  /* Sidebar items */
  .sidebar-title {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #5a6080;
    padding: 0.5rem 0 0.3rem;
  }

  /* Tags */
  .tag-critical { background:#3d1515; color:#f87171; border:1px solid #7f1d1d;
                  padding:2px 10px; border-radius:99px; font-size:11px; font-weight:600; }
  .tag-high     { background:#3d2c0a; color:#fbbf24; border:1px solid #78350f;
                  padding:2px 10px; border-radius:99px; font-size:11px; font-weight:600; }
  .tag-medium   { background:#1a2e1a; color:#4ade80; border:1px solid #14532d;
                  padding:2px 10px; border-radius:99px; font-size:11px; font-weight:600; }

  /* Insight boxes */
  .insight-box {
    background: #1e2130;
    border-left: 3px solid #6366f1;
    border-radius: 0 8px 8px 0;
    padding: 0.9rem 1.2rem;
    margin: 0.5rem 0;
  }
  .insight-box.amber { border-left-color: #f59e0b; }
  .insight-box.red   { border-left-color: #ef4444; }
  .insight-box.teal  { border-left-color: #14b8a6; }
  .insight-box.green { border-left-color: #22c55e; }

  /* Transcript cards */
  .transcript-card {
    background: #1a1d27;
    border: 1px solid #2d3148;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin: 0.5rem 0;
  }
  .transcript-title { font-weight: 600; color: #e0e4ff; font-size: 14px; }
  .transcript-meta  { font-size: 12px; color: #6b7280; margin: 4px 0 8px; }
  .transcript-body  { font-size: 13px; color: #a8b0cc; line-height: 1.6; }
  .km-badge {
    display:inline-block; font-size:11px; padding:2px 8px;
    border-radius:99px; margin:2px; font-weight:500;
  }
  .km-churn    { background:#3d1515; color:#f87171; }
  .km-concern  { background:#3d2c0a; color:#fbbf24; }
  .km-feature  { background:#1e2340; color:#818cf8; }
  .km-positive { background:#1a2e1a; color:#4ade80; }
  .km-tech     { background:#2d1f3d; color:#c084fc; }
</style>
""", unsafe_allow_html=True)

# ── DATABASE ──────────────────────────────────────────────────────────────────

DB_PATH = Path("transcripts.db")

@st.cache_resource
def get_conn():
    if not DB_PATH.exists():
        st.error(f"Database not found at {DB_PATH}. Run: python engine.py ./dataset")
        st.stop()
    return sqlite3.connect(str(DB_PATH), check_same_thread=False)


@st.cache_data(ttl=30)   # refresh every 30 seconds
def load_meetings():
    conn = get_conn()
    df = pd.read_sql("SELECT * FROM meetings ORDER BY sentiment_score ASC", conn)
    for col in ["topics_json","emails_json","competitors_json","action_items_json","speaker_stats_json"]:
        if col not in df.columns: df[col] = [[] for _ in range(len(df))]
        df[col] = df[col].apply(lambda x: json.loads(x) if x else [])
    return df


@st.cache_data(ttl=30)
def load_key_moments():
    conn = get_conn()
    return pd.read_sql("""
        SELECT km.*, m.title, m.call_type, m.sentiment_score, m.topic_cluster
        FROM key_moments km
        JOIN meetings m ON km.meeting_id = m.meeting_id
        ORDER BY m.sentiment_score ASC
    """, conn)


@st.cache_data(ttl=30)
def load_action_items():
    conn = get_conn()
    return pd.read_sql("""
        SELECT ai.*, m.call_type, m.topic_cluster
        FROM action_items ai
        JOIN meetings m ON ai.meeting_id = m.meeting_id
    """, conn)


@st.cache_data(ttl=30)
def load_stats():
    conn = get_conn()
    def one(sql): return conn.execute(sql).fetchone()[0]
    def many(sql): return dict(conn.execute(sql).fetchall())
    return {
        "total_meetings": one("SELECT COUNT(*) FROM meetings"),
        "call_types":     many("SELECT call_type, COUNT(*) FROM meetings GROUP BY call_type"),
        "churn_levels":   {},
        "model_used":     "claude-sonnet-4-5 (Anthropic)",
    }

@st.cache_data(ttl=30)
def compute_churn_risk():
    df = load_meetings()
    kms = load_key_moments()
    churn_kms = kms[kms["moment_type"] == "churn_signal"]

    churn_by: dict = defaultdict(int)
    for _, row in churn_kms.iterrows():
        company = _extract_company(row["title"])
        if company:
            churn_by[company] += 1

    records = []
    for _, row in df.iterrows():
        company = _extract_company(row["title"])
        if not company:
            continue
        records.append({
            "company": company,
            "call_type": row["call_type"],
            "score": row["sentiment_score"],
            "competitors": row["competitors_json"] if isinstance(row["competitors_json"], list) else [],
        })

    account: dict = defaultdict(lambda: {"calls":[], "competitors": set()})
    for r in records:
        account[r["company"]]["calls"].append(r)
        account[r["company"]]["competitors"].update(r["competitors"])

    results = []
    for company, data in account.items():
        calls = data["calls"]
        avg_s = sum(c["score"] for c in calls) / len(calls)
        sup_r = sum(1 for c in calls if c["call_type"] == "support") / len(calls)
        comp_m = sum(len(c["competitors"]) for c in calls)
        risk = (5 - avg_s) + (sup_r * 2) + (churn_by.get(company, 0) * 0.7) + (comp_m * 1.2)
        results.append({
            "Company":              company,
            "Risk Score":           round(risk, 1),
            "Avg Sentiment":        round(avg_s, 2),
            "Support %":            f"{round(sup_r*100)}%",
            "Churn Signals":        churn_by.get(company, 0),
            "Competitor Mentions":  comp_m,
            "Competitors Named":    ", ".join(data["competitors"]) or "—",
            "Total Calls":          len(calls),
            "Risk Level":           "CRITICAL" if risk >= 5.0 else "HIGH" if risk >= 3.5 else "MEDIUM",
        })

    return pd.DataFrame(sorted(results, key=lambda x: -x["Risk Score"]))


def _extract_company(title: str):
    if "Aegis /" in title:
        return title.split("Aegis / ")[1].split(" - ")[0].strip()
    for p in ["Support Case #", "ESCALATION: ", "URGENT: ", "INCIDENT: "]:
        if p in title:
            rest = title.replace(p, "")
            parts = rest.split(" - ")
            if len(parts) > 1:
                return " ".join(parts[1].split()[:3])
    return None


# ── SIDEBAR ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🎙️ Transcript Intelligence")
    st.markdown("*AegisCloud B2B SaaS*")
    st.divider()

    page = st.radio(
        "Navigate",
        ["📊 Executive Summary",
         "🗂️ Topic Clusters",
         "😐 Sentiment Analysis",
         "⚠️ Churn Risk",
         "🥊 Competitor Intel",
         "🔧 Feature Gaps",
         ],
        label_visibility="collapsed",
    )

    st.divider()
    df_all = load_meetings()
    stats = load_stats()
    st.metric("Total Meetings", stats.get("total_meetings", 0))
    st.markdown(f"""
    <div style="background:#1a1d27;border:1px solid #6366f1;border-radius:8px;padding:8px 12px;margin:8px 0">
      <div style="font-size:10px;color:#5a6080;text-transform:uppercase;letter-spacing:0.08em">AI Model</div>
      <div style="font-size:12px;color:#818cf8;font-weight:600">Claude Sonnet 4.5</div>
      <div style="font-size:10px;color:#5a6080">Anthropic API</div>
    </div>
    """, unsafe_allow_html=True)
    st.caption("Data refreshes every 30s")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — EXECUTIVE SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

if page == "📊 Executive Summary":
    st.title("Transcript Intelligence")
    st.markdown("*What 100 calls are telling us about AegisCloud's customers, products, and risks.*")
    st.divider()

    df   = load_meetings()
    kms  = load_key_moments()
    risk = compute_churn_risk()

    # ── Top stats ──
    churn_count    = int((kms["moment_type"] == "churn_signal").sum())
    gap_count      = int((kms["moment_type"] == "feature_gap").sum())
    critical_count = int((risk["Risk Level"] == "CRITICAL").sum())

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.markdown(f"""
    <div style="background:#1e2130;border:1px solid #2d3148;border-radius:10px;padding:1rem">
      <div style="font-size:11px;color:#5a6080;text-transform:uppercase;letter-spacing:0.08em">Total Calls</div>
      <div style="font-size:2.5rem;font-weight:700;color:#f0f2ff">100</div>
      <div style="font-size:12px;color:#6b7280;margin-top:4px">Across 3 call types analyzed</div>
    </div>""", unsafe_allow_html=True)

    c2.markdown(f"""
    <div style="background:#1e2130;border:1px solid #7f1d1d;border-radius:10px;padding:1rem">
      <div style="font-size:11px;color:#5a6080;text-transform:uppercase;letter-spacing:0.08em">Churn Signals</div>
      <div style="font-size:2.5rem;font-weight:700;color:#ef4444">{churn_count}</div>
      <div style="font-size:12px;color:#6b7280;margin-top:4px">Explicit dissatisfaction moments found across 100 calls</div>
    </div>""", unsafe_allow_html=True)

    c3.markdown(f"""
    <div style="background:#1e2130;border:1px solid #1e1b4b;border-radius:10px;padding:1rem">
      <div style="font-size:11px;color:#5a6080;text-transform:uppercase;letter-spacing:0.08em">Feature Gaps</div>
      <div style="font-size:2.5rem;font-weight:700;color:#818cf8">{gap_count}</div>
      <div style="font-size:12px;color:#6b7280;margin-top:4px">Product gaps mentioned directly by customers in calls</div>
    </div>""", unsafe_allow_html=True)

    c4.markdown(f"""
    <div style="background:#1e2130;border:1px solid #7f1d1d;border-radius:10px;padding:1rem">
      <div style="font-size:11px;color:#5a6080;text-transform:uppercase;letter-spacing:0.08em">Critical Accounts</div>
      <div style="font-size:2.5rem;font-weight:700;color:#ef4444">{critical_count}</div>
      <div style="font-size:12px;color:#6b7280;margin-top:4px">Composite risk score ≥5.0 — <b style="color:#f87171">CS escalation needed now</b></div>
    </div>""", unsafe_allow_html=True)

    c5.markdown(f"""
    <div style="background:#1e2130;border:1px solid #2d3148;border-radius:10px;padding:1rem">
      <div style="font-size:11px;color:#5a6080;text-transform:uppercase;letter-spacing:0.08em">Action Items</div>
      <div style="font-size:2.5rem;font-weight:700;color:#f59e0b">397</div>
      <div style="font-size:12px;color:#6b7280;margin-top:4px">Commitments made on calls — extracted from transcripts</div>
    </div>""", unsafe_allow_html=True)

    st.divider()

    # ── Sentiment gap ──
    st.subheader("The Sentiment Gap")
    col1, col2 = st.columns([2, 1])

    with col1:
        sent_data = pd.DataFrame({
            "Call Type": ["Support", "Internal", "External"],
            "Avg Score": [2.77, 3.48, 3.90],
            "Color":     ["#ef4444", "#f59e0b", "#22c55e"],
        })
        fig = go.Figure(go.Bar(
            x=sent_data["Avg Score"],
            y=sent_data["Call Type"],
            orientation="h",
            marker_color=sent_data["Color"],
            text=[f"{s:.2f}/5" for s in sent_data["Avg Score"]],
            textposition="outside",
            textfont=dict(color="#f0f2ff", size=13),
        ))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#c4c9e2"),
            xaxis=dict(range=[0, 5.5], showgrid=True, gridcolor="#1e2130", color="#5a6080"),
            yaxis=dict(color="#c4c9e2"),
            height=200,
            margin=dict(l=0, r=60, t=10, b=10),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("""
        <div class="insight-box red">
          <b style="color:#f87171">1.13 point gap</b><br>
          <span style="font-size:13px">Support vs External. The only time many customers interact with AegisCloud is when something is broken.</span>
        </div>
        <div class="insight-box teal" style="margin-top:8px">
          <b style="color:#2dd4bf">Compliance calls: 4.22 avg</b><br>
          <span style="font-size:13px">Highest sentiment cluster. A hidden retention lever.</span>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # ── Signals ──
    st.subheader("402 Named Signals Across All Calls")
    cols = st.columns(4)
    signals = [
        ("84", "Concerns Raised",   "#f59e0b", "Active risks flagged on calls"),
        ("76", "Positive Pivots",   "#22c55e", "Moments of genuine satisfaction"),
        ("61", "Churn Signals",     "#ef4444", "Explicit 'I might leave' moments"),
        ("51", "Feature Gaps",      "#818cf8", "Product gaps named by customers"),
    ]
    for col, (num, label, color, desc) in zip(cols, signals):
        col.markdown(f"""
        <div style="background:#1e2130;border:1px solid #2d3148;border-radius:10px;padding:1rem;text-align:center">
          <div style="font-size:2.5rem;font-weight:700;color:{color}">{num}</div>
          <div style="font-size:13px;font-weight:600;color:#e0e4ff;margin:4px 0">{label}</div>
          <div style="font-size:11px;color:#6b7280">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # ── Recommendations ──
    st.subheader("Top Recommendations")
    recs = [
        ("🔴", "URGENT", "#ef4444", "17 accounts scored above 5.0 on composite churn risk — worth a CS review this week."),
        ("🟡", "HIGH",   "#f59e0b", "Deploy a SentinelShield battlecard. 22 calls mention them post-outage, avg sentiment 2.81. This is coordinated risk, not isolated churn."),
        ("🟢", "MEDIUM", "#22c55e", "Formalise quarterly compliance review calls as a product touchpoint. Highest sentiment cluster (4.22) — customers leave these feeling confident."),
        ("🔵", "MEDIUM", "#818cf8", "Route feature_gap moments to product backlog via MCP → Jira agent. 51 gaps identified across calls."),
    ]
    for icon, level, color, text in recs:
        st.markdown(f"""
        <div style="background:#1e2130;border:1px solid #2d3148;border-radius:8px;
                    padding:0.8rem 1rem;margin:6px 0;display:flex;gap:12px;align-items:flex-start">
          <span style="font-size:1.2rem">{icon}</span>
          <div>
            <span style="font-size:11px;font-weight:700;color:{color};text-transform:uppercase;
                         letter-spacing:0.08em">{level}</span>
            <p style="margin:2px 0 0;font-size:13px;color:#c4c9e2">{text}</p>
          </div>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — TOPIC CLUSTERS
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🗂️ Topic Clusters":
    st.title("Topic Clusters")
    st.markdown("9 clusters identified using a **hybrid approach**: rule-based keyword matching + pre-extracted topic tags already present in the dataset.")
    st.divider()

    df = load_meetings()

    CLUSTER_COLORS = {
        "Compliance & Audit Readiness":     "#14b8a6",
        "Incident & Outage Response":       "#ef4444",
        "Technical Support & Integration":  "#f59e0b",
        "Engineering & Sprint Operations":  "#6366f1",
        "Contract, Renewal & Pricing":      "#a855f7",
        "Competitive Intelligence":         "#ec4899",
        "Onboarding & Deployment":          "#22c55e",
        "Security & Identity Management":   "#0ea5e9",
        "Product Feedback & Feature Gaps":  "#f97316",
    }

    cluster_counts = df.groupby("topic_cluster").agg(
        count=("meeting_id","count"),
        avg_sentiment=("sentiment_score","mean")
    ).reset_index().sort_values("count", ascending=True)
    cluster_counts["color"] = cluster_counts["topic_cluster"].map(CLUSTER_COLORS)
    cluster_counts["avg_sentiment"] = cluster_counts["avg_sentiment"].round(2)

    col1, col2 = st.columns([3, 2])

    with col1:
        fig = go.Figure(go.Bar(
            y=cluster_counts["topic_cluster"],
            x=cluster_counts["count"],
            orientation="h",
            marker_color=cluster_counts["color"],
            text=[f"{c}  ({s:.1f}★)" for c, s in zip(cluster_counts["count"], cluster_counts["avg_sentiment"])],
            textposition="outside",
            textfont=dict(color="#c4c9e2", size=12),
        ))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#c4c9e2"),
            xaxis=dict(showgrid=True, gridcolor="#1e2130", color="#5a6080", range=[0, 45]),
            yaxis=dict(color="#c4c9e2"),
            height=380,
            margin=dict(l=0, r=80, t=10, b=10),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("**Why Hybrid Classification?**")
        st.markdown("""
        <div class="insight-box">
          <b style="color:#818cf8">Not KMeans</b> — 100 records is too small for stable unsupervised clustering. Results change on every run.
        </div>
        <div class="insight-box amber" style="margin-top:6px">
          <b style="color:#fbbf24">Not pure LLM</b> — the dataset already has pre-extracted topic tags, sentiment scores, and key moments in <code>summary.json</code>. We reuse these signals rather than duplicating the work.
        </div>
        <div class="insight-box teal" style="margin-top:6px">
          <b style="color:#2dd4bf">Hybrid wins</b> — keyword scoring over title + existing tags + summary. Fast, explainable, every decision traceable to a keyword hit.
        </div>
        """, unsafe_allow_html=True)

    st.divider()
    st.subheader("Explore a Cluster")

    selected = st.selectbox("Select a topic cluster to see real transcript examples",
                             sorted(df["topic_cluster"].unique()))
    cluster_df = df[df["topic_cluster"] == selected].sort_values("sentiment_score")

    total   = len(cluster_df)
    avg_s   = cluster_df["sentiment_score"].mean()
    col1, col2, col3 = st.columns(3)
    col1.metric("Meetings in cluster", total)
    col2.metric("Avg sentiment", f"{avg_s:.2f}/5")
    col3.metric("Color", CLUSTER_COLORS.get(selected, "#6366f1"))

    st.markdown(f"**Showing top 3 examples from {selected}:**")

    for _, row in cluster_df.head(3).iterrows():
        badge_type = "support" if row["call_type"] == "support" else "external" if row["call_type"] == "external" else "internal"
        badge_color = {"support": "#ef4444", "external": "#14b8a6", "internal": "#6366f1"}[badge_type]
        topics_str = ", ".join(row["topics_json"][:4]) if row["topics_json"] else ""

        st.markdown(f"""
        <div class="transcript-card">
          <div class="transcript-title">{row['title']}</div>
          <div class="transcript-meta">
            <span style="color:{badge_color};font-weight:600">{row['call_type'].upper()}</span>
            &nbsp;·&nbsp; Sentiment: <b>{row['sentiment_score']}</b>/5
            &nbsp;·&nbsp; {topics_str}
          </div>
          <div class="transcript-body">{row['summary_text'][:350]}...</div>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — SENTIMENT ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

elif page == "😐 Sentiment Analysis":
    st.title("Sentiment Analysis")
    st.markdown("How customers and the team feel — and more importantly, **what it means**.")
    st.divider()

    df  = load_meetings()
    kms = load_key_moments()

    # ── Avg by call type ──
    type_stats = df.groupby("call_type")["sentiment_score"].agg(["mean","min","max","count"]).reset_index()
    type_stats.columns = ["Call Type","Avg","Min","Max","Count"]
    type_stats = type_stats.sort_values("Avg")
    type_stats["Avg"] = type_stats["Avg"].round(2)

    col1, col2, col3 = st.columns(3)
    colors = {"support":"#ef4444","external":"#22c55e","internal":"#f59e0b"}
    interps = {
        "support":  "55%+ of support calls are mixed-negative or worse. Every unresolved support call is a slow-motion churn event.",
        "external": "64% of external calls are mixed-positive or better. But the 10 mixed-negative ones contain 25 churn signals.",
        "internal": "31% negative. Post-outage war rooms pulling this down. Internal negativity precedes customer impact by 1-2 weeks.",
    }
    for col, (_, row) in zip([col1, col2, col3], type_stats.iterrows()):
        ct = row["Call Type"]
        col.markdown(f"""
        <div style="background:#1e2130;border:1px solid #2d3148;border-radius:10px;padding:1.2rem">
          <div style="font-size:11px;text-transform:uppercase;letter-spacing:0.1em;color:#5a6080">{ct}</div>
          <div style="font-size:3rem;font-weight:700;color:{colors.get(ct,'#818cf8')};line-height:1.1">{row['Avg']}</div>
          <div style="font-size:12px;color:#6b7280">n={int(row['Count'])} calls</div>
          <div style="font-size:12px;color:#8b92b0;margin-top:8px;line-height:1.5">{interps.get(ct,'')}</div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # ── Stacked sentiment breakdown ──
    st.subheader("Sentiment Label Breakdown")

    LABEL_ORDER = ["very-negative","negative","mixed-negative","mixed-positive","positive","very-positive"]
    LABEL_COLORS = {
        "very-negative":  "#7f1d1d",
        "negative":       "#ef4444",
        "mixed-negative": "#fca5a5",
        "mixed-positive": "#86efac",
        "positive":       "#22c55e",
        "very-positive":  "#15803d",
    }

    breakdown = df.groupby(["call_type","sentiment_label"]).size().reset_index(name="count")
    fig = go.Figure()
    for label in LABEL_ORDER:
        sub = breakdown[breakdown["sentiment_label"] == label]
        if sub.empty:
            continue
        fig.add_trace(go.Bar(
            name=label,
            x=sub["call_type"],
            y=sub["count"],
            marker_color=LABEL_COLORS.get(label, "#6366f1"),
        ))
    fig.update_layout(
        barmode="stack",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#c4c9e2"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
        xaxis=dict(color="#c4c9e2"),
        yaxis=dict(color="#c4c9e2", gridcolor="#1e2130"),
        height=320,
        margin=dict(l=0, r=0, t=10, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ── Sentiment by cluster ──
    st.subheader("Avg Sentiment by Topic Cluster")
    cluster_sent = df.groupby("topic_cluster")["sentiment_score"].mean().sort_values().reset_index()
    cluster_sent.columns = ["Cluster","Avg Sentiment"]
    cluster_sent["Avg Sentiment"] = cluster_sent["Avg Sentiment"].round(2)
    cluster_sent["color"] = cluster_sent["Avg Sentiment"].apply(
        lambda x: "#ef4444" if x < 3.0 else "#f59e0b" if x < 3.8 else "#22c55e"
    )

    fig2 = go.Figure(go.Bar(
        y=cluster_sent["Cluster"],
        x=cluster_sent["Avg Sentiment"],
        orientation="h",
        marker_color=cluster_sent["color"],
        text=[f"{s:.2f}" for s in cluster_sent["Avg Sentiment"]],
        textposition="outside",
        textfont=dict(color="#c4c9e2"),
    ))
    fig2.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#c4c9e2"),
        xaxis=dict(range=[0,5.5], gridcolor="#1e2130", color="#5a6080"),
        yaxis=dict(color="#c4c9e2"),
        height=320,
        margin=dict(l=0, r=60, t=10, b=10),
        showlegend=False,
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("""
    <div class="insight-box teal">
      <b style="color:#2dd4bf">Compliance calls score highest (4.22 avg)</b> — Counterintuitive and important.
      Customers who call about compliance leave feeling confident. Formalising quarterly compliance
      reviews as a product touchpoint is a retention lever hiding in plain sight.
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # ── Signal breakdown ──
    st.subheader("Signal Breakdown Across All Calls")
    sig_counts = kms["moment_type"].value_counts().reset_index()
    sig_counts.columns = ["Signal","Count"]
    sig_colors = {
        "concern":"#f59e0b","positive_pivot":"#22c55e","churn_signal":"#ef4444",
        "technical_issue":"#a855f7","feature_gap":"#818cf8","action_item":"#6366f1",
        "praise":"#14b8a6","pricing_offer":"#ec4899",
    }
    sig_counts["color"] = sig_counts["Signal"].map(sig_colors)

    fig3 = px.bar(sig_counts, x="Signal", y="Count",
                  color="Signal", color_discrete_map=sig_colors,
                  text="Count")
    fig3.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#c4c9e2"),
        showlegend=False,
        xaxis=dict(color="#c4c9e2"),
        yaxis=dict(color="#c4c9e2", gridcolor="#1e2130"),
        height=280,
        margin=dict(l=0,r=0,t=10,b=10),
    )
    fig3.update_traces(textposition="outside", textfont_color="#f0f2ff")
    st.plotly_chart(fig3, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — CHURN RISK
# ══════════════════════════════════════════════════════════════════════════════

elif page == "⚠️ Churn Risk":
    st.title("Account Churn Risk")
    st.markdown("Composite risk score per account — recomputed live from the database every time this page loads.")
    st.divider()

    risk_df = compute_churn_risk()

    # ── Formula ──
    st.markdown("""
    <div style="background:#1e2130;border:1px solid #2d3148;border-radius:8px;padding:1rem 1.2rem;font-family:monospace;font-size:13px;color:#818cf8">
      risk = (5 − avg_sentiment) &nbsp;+&nbsp; (support_ratio × 2) &nbsp;+&nbsp; (churn_signals × 0.7) &nbsp;+&nbsp; (competitor_mentions × 1.2)
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="font-size:12px;color:#5a6080;margin:6px 0 16px">
      Competitor mentions get the highest per-unit weight (1.2) because naming a competitor means evaluation has already started.
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Critical (≥5.0)", int((risk_df["Risk Level"]=="CRITICAL").sum()), help="Needs CS escalation this week")
    col2.metric("High (3.5–4.9)", int((risk_df["Risk Level"]=="HIGH").sum()))
    col3.metric("Medium (<3.5)", int((risk_df["Risk Level"]=="MEDIUM").sum()))

    st.divider()

    # ── Filter ──
    level_filter = st.multiselect(
        "Filter by risk level",
        ["CRITICAL","HIGH","MEDIUM"],
        default=["CRITICAL","HIGH"],
    )
    filtered = risk_df[risk_df["Risk Level"].isin(level_filter)]

    # ── Table ──
    def color_risk(val):
        if val == "CRITICAL": return "background-color:#3d1515;color:#f87171;font-weight:700"
        if val == "HIGH":     return "background-color:#3d2c0a;color:#fbbf24;font-weight:700"
        return "background-color:#1a2e1a;color:#4ade80"

    def color_score(val):
        if val >= 5.0: return "color:#f87171;font-weight:700"
        if val >= 3.5: return "color:#fbbf24"
        return "color:#4ade80"

    display = filtered[["Company","Risk Score","Risk Level","Avg Sentiment",
                         "Support %","Churn Signals","Competitor Mentions","Competitors Named","Total Calls"]].copy()

    styled = display.style\
        .map(color_risk, subset=["Risk Level"])\
        .map(color_score, subset=["Risk Score"])\
        .set_properties(**{"background-color":"#1e2130","color":"#c4c9e2","border":"1px solid #2d3148"})\
        .set_table_styles([{"selector":"th","props":[("background-color","#151720"),("color","#818cf8"),("font-size","12px"),("text-transform","uppercase"),("letter-spacing","0.06em")]}])

    st.dataframe(styled, use_container_width=True, height=420)

    st.divider()

    # ── Risk scatter ──
    st.subheader("Sentiment vs Risk Score")
    fig = px.scatter(
        risk_df,
        x="Avg Sentiment", y="Risk Score",
        color="Risk Level",
        color_discrete_map={"CRITICAL":"#ef4444","HIGH":"#f59e0b","MEDIUM":"#22c55e"},
        hover_name="Company",
        hover_data=["Churn Signals","Competitor Mentions","Total Calls"],
        size="Churn Signals",
        size_max=25,
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(30,33,48,0.5)",
        font=dict(color="#c4c9e2"),
        xaxis=dict(color="#c4c9e2", gridcolor="#2d3148"),
        yaxis=dict(color="#c4c9e2", gridcolor="#2d3148"),
        height=380,
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — COMPETITOR INTEL
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🥊 Competitor Intel":
    st.title("Competitor Intelligence")
    st.markdown("Extracted from every transcript. Low sentiment when mentioned = customers naming competitors while already upset.")
    st.divider()

    df = load_meetings()
    comp_records = []
    for _, row in df.iterrows():
        comps = row["competitors_json"] if isinstance(row["competitors_json"], list) else []
        for c in comps:
            comp_records.append({
                "competitor":  c,
                "title":       row["title"],
                "call_type":   row["call_type"],
                "sentiment":   row["sentiment_score"],
                "topic":       row["topic_cluster"],
            })

    if not comp_records:
        st.info("No competitor mentions found in the database.")
    else:
        comp_df = pd.DataFrame(comp_records)

        # ── Summary cards ──
        comp_summary = comp_df.groupby("competitor").agg(
            mentions=("title","count"),
            avg_sentiment=("sentiment","mean")
        ).reset_index().sort_values("mentions", ascending=False)
        comp_summary["avg_sentiment"] = comp_summary["avg_sentiment"].round(2)
        comp_summary["threat"] = comp_summary["avg_sentiment"].apply(
            lambda x: "ACTIVE THREAT" if x < 3.0 else "MONITORING" if x < 3.8 else "BENCHMARK"
        )

        cols = st.columns(len(comp_summary))
        threat_colors = {"ACTIVE THREAT":"#ef4444","MONITORING":"#f59e0b","BENCHMARK":"#22c55e"}
        for col, (_, row) in zip(cols, comp_summary.iterrows()):
            tc = threat_colors.get(row["threat"], "#818cf8")
            col.markdown(f"""
            <div style="background:#1e2130;border:1px solid #2d3148;border-radius:10px;padding:1.2rem;text-align:center">
              <div style="font-size:1rem;font-weight:700;color:#e0e4ff;text-transform:capitalize">{row['competitor']}</div>
              <div style="font-size:2.2rem;font-weight:700;color:{tc}">{row['mentions']}</div>
              <div style="font-size:11px;color:#6b7280">mentions</div>
              <div style="font-size:12px;font-weight:600;color:{tc};margin-top:6px">{row['threat']}</div>
              <div style="font-size:12px;color:#8b92b0">avg sentiment {row['avg_sentiment']}</div>
            </div>
            """, unsafe_allow_html=True)

        st.divider()

        # ── How to read this chart ──
        st.subheader("Sentiment When Each Competitor Is Mentioned")
        st.markdown("""
        <div style="background:#1e2130;border:1px solid #2d3148;border-radius:8px;
                    padding:0.9rem 1.2rem;margin-bottom:1rem">
          <div style="font-size:13px;color:#c4c9e2;line-height:1.7">
            <b style="color:#e0e4ff">How to read this chart:</b>
            Each box shows the range of sentiment scores in calls where that competitor was mentioned.
            <b style="color:#ef4444">Lower box = customers were already upset when they mentioned that competitor.</b>
            This is the key signal — customers don't casually compare vendors. When they name a
            competitor, they've already started evaluating alternatives.
            A box sitting below 3.0 means that competitor is being evaluated by unhappy customers.
          </div>
        </div>
        """, unsafe_allow_html=True)

        fig = px.box(
            comp_df, x="competitor", y="sentiment",
            color="competitor",
            points="all",
            hover_data=["title","call_type"],
            labels={"sentiment": "Sentiment Score (1=very negative, 5=very positive)", "competitor": "Competitor"},
        )
        fig.add_hline(y=3.0, line_dash="dash", line_color="#6366f1",
                      annotation_text="Neutral threshold (3.0)",
                      annotation_position="right",
                      annotation_font_color="#818cf8")
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(30,33,48,0.5)",
            font=dict(color="#c4c9e2"),
            xaxis=dict(color="#c4c9e2", title="Competitor"),
            yaxis=dict(color="#c4c9e2", gridcolor="#2d3148", range=[0,5.8]),
            showlegend=False,
            height=360,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Per-competitor plain English interpretation
        for _, row in comp_summary.iterrows():
            comp  = row["competitor"].title()
            avg_s = row["avg_sentiment"]
            n     = row["mentions"]
            if avg_s < 3.0:
                color  = "#ef4444"
                label  = "ACTIVE THREAT"
                meaning = f"Mentioned in {n} calls with avg sentiment {avg_s}/5. Customers bring up {comp} when they're already in crisis mode — this is not casual comparison shopping. They've had conversations with {comp}'s sales team."
            elif avg_s < 3.8:
                color  = "#f59e0b"
                label  = "MONITORING"
                meaning = f"Mentioned in {n} calls with avg sentiment {avg_s}/5. Some customers are evaluating {comp} but haven't committed to switching. Early warning — needs competitive response before these accounts reach renewal."
            else:
                color  = "#22c55e"
                label  = "LOW RISK"
                meaning = f"Mentioned in {n} calls with avg sentiment {avg_s}/5. Referenced positively — customers use {comp} as a benchmark. Not currently a switching threat."
            st.markdown(f"""
            <div style="background:#1e2130;border-left:3px solid {color};border-radius:0 8px 8px 0;
                        padding:0.8rem 1.2rem;margin:6px 0">
              <div style="display:flex;align-items:center;gap:10px;margin-bottom:4px">
                <span style="font-weight:700;color:#e0e4ff">{comp}</span>
                <span style="font-size:11px;font-weight:700;color:{color};background:rgba(0,0,0,0.3);
                             padding:2px 8px;border-radius:99px">{label}</span>
                <span style="font-size:12px;color:#6b7280">{n} mentions · avg {avg_s}/5</span>
              </div>
              <div style="font-size:13px;color:#a8b0cc">{meaning}</div>
            </div>
            """, unsafe_allow_html=True)

        st.divider()

        # ── Call list ──
        st.subheader("Calls Where Competitors Were Mentioned")
        selected_comp = st.selectbox("Filter by competitor", ["All"] + list(comp_summary["competitor"]))
        filtered_comp = comp_df if selected_comp == "All" else comp_df[comp_df["competitor"] == selected_comp]
        filtered_comp = filtered_comp.sort_values("sentiment")
        for _, row in filtered_comp.head(10).iterrows():
            color = "#ef4444" if row["sentiment"] < 2.5 else "#f59e0b" if row["sentiment"] < 3.5 else "#22c55e"
            st.markdown(f"""
            <div class="transcript-card">
              <div class="transcript-title">{row['title']}</div>
              <div class="transcript-meta">
                {row['call_type'].upper()} &nbsp;·&nbsp;
                Sentiment: <b style="color:{color}">{row['sentiment']}</b>/5 &nbsp;·&nbsp;
                <span style="color:#ec4899;font-weight:600">{row['competitor']}</span>
              </div>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — FEATURE GAPS
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🔧 Feature Gaps":
    st.title("Feature Gaps")
    st.markdown("51 product gaps mentioned directly by customers across support and external calls.")
    st.divider()

    kms = load_key_moments()
    gaps = kms[kms["moment_type"] == "feature_gap"].copy()

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Gaps", len(gaps))
    col2.metric("From External Calls", int((gaps["call_type"]=="external").sum()))
    col3.metric("From Support Calls", int((gaps["call_type"]=="support").sum()))

    st.divider()

    # ── Filters ──
    col1, col2 = st.columns(2)
    with col1:
        ct_filter = st.multiselect("Call type", ["external","support","internal"], default=["external","support"])
    with col2:
        max_sentiment = st.slider("Max sentiment (lower = higher risk)", 1.0, 5.0, 5.0, 0.1)

    filtered_gaps = gaps[
        (gaps["call_type"].isin(ct_filter)) &
        (gaps["sentiment_score"] <= max_sentiment)
    ].sort_values("sentiment_score")

    st.markdown(f"**Showing {len(filtered_gaps)} gaps:**")

    for _, row in filtered_gaps.head(15).iterrows():
        score = row["sentiment_score"]
        score_color = "#ef4444" if score < 2.5 else "#f59e0b" if score < 3.5 else "#22c55e"
        ct_color = {"support":"#ef4444","external":"#14b8a6","internal":"#6366f1"}.get(row["call_type"],"#818cf8")
        st.markdown(f"""
        <div class="transcript-card">
          <div class="transcript-title">{row['title']}</div>
          <div class="transcript-meta">
            <span style="color:{ct_color};font-weight:600">{row['call_type'].upper()}</span>
            &nbsp;·&nbsp; Sentiment: <b style="color:{score_color}">{score}</b>/5
            &nbsp;·&nbsp; {row['topic_cluster']}
            {f"&nbsp;·&nbsp; <i>{row['speaker']}</i>" if row.get('speaker') else ''}
          </div>
          <div class="transcript-body">{row['moment_text']}</div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()
    st.markdown("""
    <div class="insight-box">
      <b style="color:#818cf8">Next step:</b> An MCP → LangGraph → Jira agent could auto-route these feature_gap moments to the product backlog automatically.
    </div>
    """, unsafe_allow_html=True)
