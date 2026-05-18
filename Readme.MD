# Transcript Intelligence

A pipeline that processes 100 B2B SaaS call transcripts, stores everything in SQLite, and presents findings through a Streamlit dashboard. An MCP server is also included — built as a foundation for future agentic workflows.
---

## Two Commands to Run Everything

```bash
# Step 1 — process all transcripts and save to database
python engine.py ./dataset

# Step 2 — launch the dashboard
streamlit run dashboard.py
```

That is it. Opens at http://localhost:8501

---

## What We Built

### `engine.py` — The Pipeline
Reads every transcript folder, runs classification and signal extraction, saves results to `transcripts.db` (SQLite). Watches the dataset folder for new arrivals — drop a new folder in and it gets processed automatically without restarting.

### `dashboard.py` — The Streamlit Dashboard
Six interactive pages built on top of `transcripts.db`:

| Page | What it shows |
|---|---|
| Executive Summary | Headline numbers with plain English context |
| Topic Clusters | 9 clusters, counts, avg sentiment, real transcript examples |
| Sentiment Analysis | Call type comparison and cluster breakdown |
| Churn Risk | Ranked account table with composite risk scores |
| Competitor Intel | Mention counts, sentiment context, what each one means |
| Feature Gaps | 51 gaps filterable by call type and sentiment |

### `mcp_server.py` — MCP Server (future use)
12 tools built on top of the same database. Not used in the current demo. Built as a foundation so any AI agent — a LangGraph workflow, a Slack bot, a CRM automation — can query transcript intelligence without touching SQL. The tools are working and tested. Wiring them to external systems is the next step.

---

## Why We Built the MCP Server

The dashboard answers questions visually. But the real value of transcript intelligence is when it drives action automatically — a churn signal fires, CS gets a Slack alert. A feature gap is mentioned, a Jira ticket is created. A competitor comes up in a call, a sales rep gets a briefing.

MCP (Model Context Protocol) is the standard way to expose data as callable tools for AI agents. By building the server now, we have a clean integration layer ready. Any agent that speaks MCP can call `get_churn_risk()` or `get_feature_gaps()` and get structured answers back — no custom API, no SQL, no re-engineering.

The 12 tools cover four categories:

| Category | Tools |
|---|---|
| Topic Intelligence | `get_topic_distribution`, `get_meetings_by_topic`, `search_transcripts` |
| Sentiment Intelligence | `get_sentiment_by_call_type`, `get_sentiment_trends` |
| Account Intelligence | `get_churn_risk`, `get_account_timeline` |
| Signal Intelligence | `get_feature_gaps`, `get_competitor_intel`, `get_action_item_owners`, `get_pipeline_summary`, `get_new_transcripts` |

To test all 12 tools locally:
```bash
python mcp_server.py transcripts.db --demo
```

---

## What We Found

All numbers come directly from `transcripts.db`.

**Call types:**
- 39 external calls (account management) — avg sentiment 3.90
- 32 support calls — avg sentiment 2.77
- 29 internal calls — avg sentiment 3.48

The 1.13 point gap between support (2.77) and external (3.90) shows that most customer interactions happen when something is broken. Proactive touchpoints — like compliance reviews — would shift that balance.

**Topic clusters:**
- Compliance & Audit Readiness: 33 calls, avg sentiment 4.22
- Incident & Outage Response: 23 calls, avg sentiment 2.63
- Technical Support & Integration: 20 calls, avg sentiment 2.96
- Engineering & Sprint Operations: 8 calls, avg sentiment 3.33
- Contract, Renewal & Pricing: 6 calls, avg sentiment 3.72
- Competitive Intelligence: 4 calls, avg sentiment 2.76
- Onboarding & Deployment: 3 calls, avg sentiment 4.37
- Security & Identity Management: 2 calls, avg sentiment 4.80
- Product Feedback & Feature Gaps: 1 call, avg sentiment 3.60

Compliance calls have the highest sentiment. Customers who discuss compliance leave calls feeling supported. Incident calls have the lowest — these are the highest-risk moments for churn.

**Signals found across all calls:**
- 84 concerns raised
- 76 positive pivots
- 61 churn signals
- 54 technical issues
- 51 feature gaps
- 43 action items
- 23 praise moments
- 10 pricing conversations

---

## How Classification Works

The dataset came with `summary.json` in each folder already containing pre-extracted fields — topics, sentiment score, key moments, action items, and a summary. Our pipeline reuses those signals and adds on top:

**Call type classification** — deterministic rules:
- Title contains "Support Case #" or "ESCALATION" → support
- Title starts with "Aegis /" → external
- All participants on @aegiscloud.com → internal

**Topic cluster assignment** — keyword scoring:
- Score each of 9 clusters by how many keywords appear in the title + existing topic tags + first 500 chars of summary
- Top scoring cluster wins
- Every decision is traceable to a specific keyword hit

**Why not KMeans clustering?** 100 records is too small — results vary between runs and are hard to explain.
**Why not pure LLM calls?** The dataset already has pre-extracted topic tags. Calling an API to re-extract them wastes cost and time.

**Churn risk formula:**
```
risk = (5 - avg_sentiment) + (support_ratio x 2) + (churn_signals x 0.7) + (competitor_mentions x 1.2)
```
Competitor mentions get the highest weight because naming a specific competitor means evaluation has already started.

## Setup

```bash
# Create virtual environment
python -m venv venv

# Activate — Mac / Linux
source venv/bin/activate

# Activate — Windows
venv\Scripts\activate

# Install all dependencies
pip install -r requirements.txt

# Create src/__init__.py — Mac / Linux
touch src/__init__.py

# Create src/__init__.py — Windows
New-Item src\__init__.py -ItemType File
```

Then put your dataset folder in the project root and run the two commands at the top.

---

## What to Change for a Different Company

**`src/taxonomy.py`**
- `TOPIC_TAXONOMY` — update cluster names and keywords to match your products
- `COMPETITORS` — replace with your competitor names in lowercase

**`src/classifier.py`**
- Change `"aegis /"` to your naming convention for external calls
- Change `"aegiscloud.com"` to your company's email domain

**`src/store.py`**
- Adjust churn risk formula weights based on what actually predicts churn in your business

---

## Dependencies

```
mcp>=1.0.0              MCP server protocol
watchdog>=3.0.0         real-time file system monitoring
streamlit>=1.30.0       dashboard framework
plotly>=5.0.0           interactive charts
pandas>=2.0.0           data handling
```
