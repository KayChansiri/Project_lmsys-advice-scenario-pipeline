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
RESTRICTED_DIR = PROJECT_DIR / "data" / "restricted"
REVIEW_DIR = PROJECT_DIR / "data" / "human_review"
REVIEW_DIR.mkdir(parents=True, exist_ok=True)

INPUT_PATH = RESTRICTED_DIR / "lmsys_llm_screening_checkpoint.parquet"
OUTPUT_PATH = REVIEW_DIR / "lmsys_human_review_candidates.csv"

screening = pd.read_parquet(INPUT_PATH)
eligible = screening[screening["eligible"].eq(True)].copy()

# Historical study run: all 624 computationally eligible cases proceeded to human review.
# Do NOT reintroduce the obsolete 250-per-domain cap from an intermediate notebook.
review_pool = (
    eligible
    .sort_values(["primary_domain", "random_priority", "source_text_hash"])
    .reset_index(drop=True)
)

review_pool.insert(
    0, "review_id",
    [f"R{i:04d}" for i in range(1, len(review_pool) + 1)]
)

# Blinded reviewer file: computational judgments are intentionally omitted.
human_review = review_pool[
    ["review_id", "source_text_hash", "source_text", "word_count"]
].copy()

human_review["human_eligible"] = ""
human_review["human_primary_domain"] = ""
human_review["human_sufficient_context"] = ""
human_review["human_notes"] = ""

human_review.to_csv(OUTPUT_PATH, index=False)

print(f"Computationally eligible cases sent to review: {len(human_review):,}")
print(f"Saved blinded human-review file: {OUTPUT_PATH}")

if len(human_review) != 624:
    print(
        f"WARNING: current export has {len(human_review):,} rows; "
        "the historical study human-review pool had 624."
    )
