# src/taxonomy.py
# ─────────────────────────────────────────────────────────────────────────────
# Topic taxonomy, competitor list, and stakeholder mappings.
# Kept in its own file so it is easy to extend without touching pipeline logic.
#
# To add a new topic cluster: add an entry to TOPIC_TAXONOMY.
# To track a new competitor:  add its name (lowercase) to COMPETITORS.
# ─────────────────────────────────────────────────────────────────────────────

TOPIC_TAXONOMY = {
    "Incident & Outage Response": {
        "keywords": [
            "outage", "incident", "downtime", "remediation", "sla breach",
            "platform outage", "service disruption", "detect outage",
            "post-mortem", "postmortem", "recovery", "availability",
            "escalation bridge", "war room", "pipeline failure",
        ],
        "stakeholders": ["Engineering Lead", "Support Leader", "CS Manager"],
        "urgency": "critical",
        "description": (
            "Live or post-incident calls: outages, escalations, SLA impacts, "
            "remediation plans. These are the highest-urgency calls — customers "
            "are in crisis and churn risk is at its peak."
        ),
    },
    "Technical Support & Integration": {
        "keywords": [
            "support case", "connector", "logvault", "integration", "timeout",
            "configuration", "backup", "throughput", "latency", "api",
            "authentication", "sso", "sync", "agent version", "workaround",
            "false positive", "alert noise", "certificate", "saml", "ldap",
            "scim", "mfa token", "policy sync",
        ],
        "stakeholders": ["Support Leader", "Engineering Lead"],
        "urgency": "high",
        "description": (
            "Customer-reported technical issues, integration problems, and "
            "configuration support. High volume, high repetition — a product "
            "signal hiding inside support queues."
        ),
    },
    "Compliance & Audit Readiness": {
        "keywords": [
            "compliance", "audit", "iso 27001", "pci dss", "soc 2", "hipaa",
            "gdpr", "regulatory", "comply", "certification", "audit preparation",
            "multi-framework", "framework", "evidence", "control mapping",
            "cmmc", "compliance reporting",
        ],
        "stakeholders": ["Sales Manager", "CS Manager", "Product Manager"],
        "urgency": "medium",
        "description": (
            "Compliance framework prep, audit readiness, and regulatory reporting. "
            "Counterintuitively the highest-sentiment cluster — customers who call "
            "about compliance leave feeling good. A hidden retention lever."
        ),
    },
    "Contract, Renewal & Pricing": {
        "keywords": [
            "contract", "renewal", "pricing", "negotiation", "discount",
            "tier", "expansion", "churn", "cancellation", "billing", "cost",
            "commercial", "upsell", "budget", "invoice", "overage",
            "annual review", "multi-year",
        ],
        "stakeholders": ["Sales Manager", "CS Manager"],
        "urgency": "high",
        "description": (
            "Commercial conversations: renewals, pricing negotiations, billing "
            "disputes, and upsell discussions. These calls decide whether a "
            "customer stays or goes."
        ),
    },
    "Product Feedback & Feature Gaps": {
        "keywords": [
            "roadmap", "feature request", "product feedback", "enhancement",
            "gap", "missing", "improvement", "prioritization", "backlog",
            "wishlist", "product demo", "release", "feature gap",
            "not supported", "not available", "on the roadmap",
        ],
        "stakeholders": ["Product Manager", "Engineering Lead"],
        "urgency": "medium",
        "description": (
            "Customer or internal feedback on missing product capabilities. "
            "The unfiltered customer voice — not shaped by support tickets "
            "or QBR summaries. Product teams rarely see this data."
        ),
    },
    "Competitive Intelligence": {
        "keywords": [
            "competitive", "competitor", "sentinelshield", "vaultedge",
            "cybernova", "fortiguard", "win-loss", "differentiation",
            "positioning", "market", "alternative", "switching",
            "vendor evaluation", "vendor comparison", "bake-off",
        ],
        "stakeholders": ["Sales Manager", "Product Manager"],
        "urgency": "high",
        "description": (
            "Competitive threat signals, win/loss patterns, and vendor comparison "
            "calls. When a customer names a competitor, they've already started "
            "the evaluation process — these calls need same-day action."
        ),
    },
    "Onboarding & Deployment": {
        "keywords": [
            "onboarding", "deployment", "kickoff", "implementation", "go-live",
            "training", "adoption", "getting started", "rollout", "pilot",
            "setup", "deploy", "launch",
        ],
        "stakeholders": ["CS Manager", "Engineering Lead"],
        "urgency": "medium",
        "description": (
            "New customer onboarding, module deployments, and implementation "
            "kickoffs. These are naturally positive calls — customers are "
            "excited and expectations are being set."
        ),
    },
    "Engineering & Sprint Operations": {
        "keywords": [
            "standup", "sprint", "engineering", "architecture", "infrastructure",
            "monitoring", "alerting", "retro", "on-call", "runbook", "capacity",
            "performance", "sprint planning", "sprint retro", "reliability",
            "scalability", "refactor",
        ],
        "stakeholders": ["Engineering Lead"],
        "urgency": "low",
        "description": (
            "Internal engineering syncs, sprint planning, retros, and architecture "
            "discussions. Contains leading indicators of customer impact — internal "
            "negativity here often precedes external churn by 1-2 weeks."
        ),
    },
    "Security & Identity Management": {
        "keywords": [
            "identity", "mfa", "access control", "rbac", "zero trust", "threat",
            "vulnerability", "security review", "permissions", "privilege",
            "breach", "identity module", "jit provisioning",
        ],
        "stakeholders": ["Engineering Lead", "CS Manager"],
        "urgency": "high",
        "description": (
            "Identity, access control, MFA, and security architecture discussions. "
            "High satisfaction when resolved — customers feel protected. "
            "High urgency when unresolved — regulatory exposure is real."
        ),
    },
}

# Competitors to track across all transcripts (lowercase for matching)
COMPETITORS = ["sentinelshield", "vaultedge", "cybernova", "fortiguard"]

# Sentiment labels in order from worst to best
SENTIMENT_ORDER = [
    "very-negative", "negative", "mixed-negative",
    "mixed-positive", "positive", "very-positive",
]
