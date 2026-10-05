from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
RESTRICTED_DIR = PROJECT_DIR / "data" / "restricted"
REVIEW_DIR = PROJECT_DIR / "data" / "human_review"

CHECKPOINT_PATH = (
    RESTRICTED_DIR
    / "lmsys_llm_screening_checkpoint.parquet"
)

OUTPUT_PATH = (
    REVIEW_DIR
    / "lmsys_human_review_batch1.csv"
)

REVIEW_N_PER_DOMAIN = 75

REVIEW_DIR.mkdir(parents=True, exist_ok=True)

screening_results = pd.read_parquet(CHECKPOINT_PATH)

eligible_cases = screening_results[
    screening_results["eligible"].eq(True)
].copy()

eligible_cases = eligible_cases.sort_values("random_priority")

review_parts = []

for domain in [
    "relationship",
    "career",
    "emotional_wellbeing",
]:
    domain_cases = eligible_cases[
        eligible_cases["primary_domain"].eq(domain)
    ].head(REVIEW_N_PER_DOMAIN)

    review_parts.append(domain_cases)

review_pool = pd.concat(
    review_parts,
    ignore_index=True,
)

review_pool["human_eligible"] = ""
review_pool["human_primary_domain"] = ""
review_pool["human_sufficient_context"] = ""
review_pool["human_notes"] = ""

review_pool.to_csv(
    OUTPUT_PATH,
    index=False,
)

print("Human-review pool:", len(review_pool))
print()
print(review_pool["primary_domain"].value_counts())
print()
print("Saved:", OUTPUT_PATH)
