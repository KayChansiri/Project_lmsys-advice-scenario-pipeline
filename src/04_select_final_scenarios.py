"""
Reproducibility pipeline for LMSYS-Chat-1M advice-scenario construction.

IMPORTANT
---------
- LMSYS-Chat-1M is gated. Do not redistribute source conversation text.
- Supply credentials through environment variables; never commit API keys/tokens.
- These scripts preserve the historical study output basenames used downstream.
"""

from pathlib import Path
import os
import pandas as pd

# ---------------- Configuration ----------------
PROJECT_DIR = Path(os.environ.get("FAACT_PROJECT_DIR", ".")).resolve()
REVIEW_DIR = PROJECT_DIR / "data" / "human_review"
OUTPUT_DIR = PROJECT_DIR / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FINAL_REVIEW_PATH = REVIEW_DIR / "lmsys_human_review_reviewer_final.xlsx"
FINAL_SAMPLE_PATH = OUTPUT_DIR / "lmsys_final_150_scenarios.csv"

RANDOM_SEED = 20260916
TARGET_DOMAINS = ["relationship", "career", "emotional_wellbeing"]

final_review = pd.read_excel(FINAL_REVIEW_PATH)

required = {
    "source_text_hash",
    "source_text",
    "word_count",
    "human_eligible",
    "human_primary_domain",
    "human_sufficient_context",
}
missing = required - set(final_review.columns)
if missing:
    raise ValueError(f"Missing required columns: {sorted(missing)}")

eligible = final_review[
    final_review["human_eligible"].eq(1)
    & final_review["human_sufficient_context"].eq(1)
    & final_review["human_primary_domain"].isin(TARGET_DOMAINS)
].copy()

print(f"Human-confirmed eligible cases: {len(eligible):,}")
print(eligible["human_primary_domain"].value_counts())

# Historical adjudicated pool reported by the study.
historical = {
    "relationship": 197,
    "career": 100,
    "emotional_wellbeing": 62,
}
for domain, expected in historical.items():
    observed = int((eligible["human_primary_domain"] == domain).sum())
    if observed != expected:
        print(f"WARNING: {domain}={observed}; historical adjudicated pool={expected}.")

for domain in TARGET_DOMAINS:
    n = int((eligible["human_primary_domain"] == domain).sum())
    if n < 50:
        raise ValueError(f"Only {n} eligible scenarios in {domain}; at least 50 required.")

selected = (
    eligible
    .groupby("human_primary_domain", group_keys=False)
    .sample(n=50, random_state=RANDOM_SEED)
    .copy()
    .sort_values(["human_primary_domain", "source_text_hash"])
    .reset_index(drop=True)
)

selected.insert(0, "scenario_id", [f"S{i:03d}" for i in range(1, 151)])

assert len(selected) == 150
assert selected["source_text_hash"].nunique() == 150
for domain in TARGET_DOMAINS:
    assert int((selected["human_primary_domain"] == domain).sum()) == 50

selected.to_csv(FINAL_SAMPLE_PATH, index=False)

print(f"Saved: {FINAL_SAMPLE_PATH}")
print(selected["human_primary_domain"].value_counts())
