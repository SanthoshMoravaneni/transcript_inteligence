# mcp_server.py
# ─────────────────────────────────────────────────────────────────────────────
# MCP server entry point. Keeps the server boilerplate separate from
# the tool logic (which lives in src/mcp_tools.py).
#
# USAGE
# ──────
# Run in demo mode (test all tools, print results):
#   python mcp_server.py transcripts.db --demo
#
# Run as live MCP server (stdio transport for Claude Desktop):
#   python mcp_server.py transcripts.db
#
# Run server + start engine watcher together:
#   python mcp_server.py transcripts.db --dataset ./dataset
#
# CONNECTING TO CLAUDE DESKTOP
# ─────────────────────────────
# Add to ~/Library/Application Support/Claude/claude_desktop_config.json:
#   {
#     "mcpServers": {
#       "transcript-intelligence": {
#         "command": "python",
#         "args": ["/full/path/to/mcp_server.py", "transcripts.db"]
#       }
#     }
#   }
# Then restart Claude Desktop. You'll see 12 new tools available.
# ─────────────────────────────────────────────────────────────────────────────

import json
import sys
import asyncio
import logging
import threading
from pathlib import Path
from typing import Optional

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from src.store import TranscriptStore
from src.mcp_tools import (
    set_store,
    get_topic_distribution, get_meetings_by_topic, search_transcripts,
    get_sentiment_by_call_type, get_sentiment_trends,
    get_churn_risk, get_account_timeline,
    get_feature_gaps, get_competitor_intel,
    get_action_item_owners, get_pipeline_summary, get_new_transcripts,
)

log = logging.getLogger("mcp-server")
app = Server("transcript-intelligence")

# ── TOOL REGISTRY ─────────────────────────────────────────────────────────────

TOOLS = [
    Tool(name="get_topic_distribution",
         description="Live topic cluster distribution, optionally filtered by call_type (support/external/internal/all).",
         inputSchema={"type":"object","properties":{"call_type":{"type":"string","default":"all"}}}),
    Tool(name="get_meetings_by_topic",
         description="Real example meetings for a topic cluster with summaries and action items.",
         inputSchema={"type":"object","properties":{"topic_cluster":{"type":"string"},"limit":{"type":"integer","default":5}},"required":["topic_cluster"]}),
    Tool(name="search_transcripts",
         description="Full-text keyword search across all meetings in the live database.",
         inputSchema={"type":"object","properties":{"query":{"type":"string"},"limit":{"type":"integer","default":10}},"required":["query"]}),
    Tool(name="get_sentiment_by_call_type",
         description="Sentiment breakdown across support, external, and internal calls with interpretation.",
         inputSchema={"type":"object","properties":{}}),
    Tool(name="get_sentiment_trends",
         description="Signal breakdown (churn signals, feature gaps, concerns, praise) with high-risk call examples.",
         inputSchema={"type":"object","properties":{"cluster":{"type":"string"}}}),
    Tool(name="get_churn_risk",
         description="Live composite churn risk scores per account. Recomputed on every call — always includes new transcripts.",
         inputSchema={"type":"object","properties":{"account":{"type":"string"},"top_n":{"type":"integer","default":10}}}),
    Tool(name="get_account_timeline",
         description="Full call history for a named account: sentiment arc, churn signals, competitor mentions, CS brief.",
         inputSchema={"type":"object","properties":{"account":{"type":"string"}},"required":["account"]}),
    Tool(name="get_feature_gaps",
         description="All product feature gaps mentioned in transcripts. Unfiltered customer voice.",
         inputSchema={"type":"object","properties":{"call_type":{"type":"string","default":"all"},"limit":{"type":"integer","default":15}}}),
    Tool(name="get_competitor_intel",
         description="Competitive intelligence extracted from all transcripts. Shows threat level per competitor.",
         inputSchema={"type":"object","properties":{"competitor":{"type":"string"}}}),
    Tool(name="get_action_item_owners",
         description="Action item ownership concentration. Reveals workload bottlenecks invisible to any PM tool.",
         inputSchema={"type":"object","properties":{}}),
    Tool(name="get_pipeline_summary",
         description="Live executive summary — always current. Entry point for dashboards and briefings.",
         inputSchema={"type":"object","properties":{}}),
    Tool(name="get_new_transcripts",
         description="Transcripts processed in the last N hours. The morning standup tool.",
         inputSchema={"type":"object","properties":{"since_hours":{"type":"integer","default":24}}}),
]

DISPATCH = {
    "get_topic_distribution":    lambda a: get_topic_distribution(**a),
    "get_meetings_by_topic":     lambda a: get_meetings_by_topic(**a),
    "search_transcripts":        lambda a: search_transcripts(**a),
    "get_sentiment_by_call_type": lambda a: get_sentiment_by_call_type(),
    "get_sentiment_trends":      lambda a: get_sentiment_trends(**a),
    "get_churn_risk":            lambda a: get_churn_risk(**a),
    "get_account_timeline":      lambda a: get_account_timeline(**a),
    "get_feature_gaps":          lambda a: get_feature_gaps(**a),
    "get_competitor_intel":      lambda a: get_competitor_intel(**a),
    "get_action_item_owners":    lambda a: get_action_item_owners(),
    "get_pipeline_summary":      lambda a: get_pipeline_summary(),
    "get_new_transcripts":       lambda a: get_new_transcripts(**a),
}


@app.list_tools()
async def list_tools() -> list[Tool]:
    return TOOLS


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    fn = DISPATCH.get(name)
    if not fn:
        result = json.dumps({"error": f"Unknown tool: {name}"})
    else:
        try:
            result = fn(arguments)
        except Exception as e:
            result = json.dumps({"error": str(e)})
    return [TextContent(type="text", text=result)]


# ── DEMO MODE ─────────────────────────────────────────────────────────────────

def run_demo():
    demos = [
        ("get_pipeline_summary",         {},                                    "EXECUTIVE SUMMARY"),
        ("get_topic_distribution",       {"call_type": "all"},                 "ALL TOPICS"),
        ("get_topic_distribution",       {"call_type": "support"},             "SUPPORT TOPICS"),
        ("get_sentiment_by_call_type",   {},                                   "SENTIMENT ANALYSIS"),
        ("get_sentiment_trends",         {},                                   "SIGNAL BREAKDOWN"),
        ("get_churn_risk",               {"top_n": 8},                        "CHURN RISK"),
        ("get_account_timeline",         {"account": "Coastal Living"},       "ACCOUNT: Coastal Living"),
        ("get_competitor_intel",         {},                                   "COMPETITOR INTEL"),
        ("get_feature_gaps",             {"call_type": "external","limit":6}, "FEATURE GAPS"),
        ("get_meetings_by_topic",        {"topic_cluster": "Incident & Outage Response","limit":3}, "INCIDENT EXAMPLES"),
        ("search_transcripts",           {"query": "SentinelShield renewal"}, "SEARCH"),
        ("get_action_item_owners",       {},                                   "ACTION OWNERS"),
        ("get_new_transcripts",          {"since_hours": 720},                "RECENT (30 days)"),
    ]

    print("=" * 70)
    print("TRANSCRIPT INTELLIGENCE — MCP SERVER DEMO")
    print("=" * 70)

    fn_map = {k: v for k, v in [
        ("get_topic_distribution",    lambda a: get_topic_distribution(**a)),
        ("get_meetings_by_topic",     lambda a: get_meetings_by_topic(**a)),
        ("search_transcripts",        lambda a: search_transcripts(**a)),
        ("get_sentiment_by_call_type", lambda a: get_sentiment_by_call_type()),
        ("get_sentiment_trends",      lambda a: get_sentiment_trends(**a)),
        ("get_churn_risk",            lambda a: get_churn_risk(**a)),
        ("get_account_timeline",      lambda a: get_account_timeline(**a)),
        ("get_feature_gaps",          lambda a: get_feature_gaps(**a)),
        ("get_competitor_intel",      lambda a: get_competitor_intel(**a)),
        ("get_action_item_owners",    lambda a: get_action_item_owners()),
        ("get_pipeline_summary",      lambda a: get_pipeline_summary()),
        ("get_new_transcripts",       lambda a: get_new_transcripts(**a)),
    ]}

    for tool_name, args, label in demos:
        print(f"\n{'─'*70}")
        print(f"▶  {tool_name}  |  {label}")
        print(f"{'─'*70}")
        try:
            raw = fn_map[tool_name](args)
            print(json.dumps(json.loads(raw), indent=2)[:1400])
            if len(raw) > 1400:
                print("  ... [truncated for demo]")
        except Exception as e:
            print(f"ERROR: {e}")

    print(f"\n{'='*70}")
    print("12 tools. All live from SQLite. Ready to connect to Claude Desktop.")
    print("=" * 70)


# ── ENTRY POINT ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args       = sys.argv[1:]
    positional = [a for a in args if not a.startswith("--")]
    db_path    = positional[0] if positional else "transcripts.db"
    dataset    = None
    demo_mode  = "--demo" in args

    for a in args:
        if a.startswith("--dataset="):
            dataset = a.split("=", 1)[1]

    store = TranscriptStore(db_path)
    set_store(store)
    log.info(f"MCP server connected to: {db_path}")

    if dataset:
        from engine import TranscriptEngine
        def _start():
            e = TranscriptEngine(dataset_dir=dataset, db_path=db_path)
            e.ingest_all()
            e.watch()
        threading.Thread(target=_start, daemon=True).start()
        log.info(f"Engine started — watching {dataset}")

    if demo_mode:
        run_demo()
    else:
        asyncio.run(stdio_server(app))
