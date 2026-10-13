from pathlib import Path

import pandas as pd


# ============================================================
# Paths
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]

RESTRICTED_DIR = (
    PROJECT_DIR
    / "data"
    / "restricted"
)

REVIEW_DIR = (
    PROJECT_DIR
    / "data"
    / "human_review"
)

CHECKPOINT_PATH = (
    RESTRICTED_DIR
    / "lmsys_llm_screening_checkpoint.parquet"
)

REVIEWER1_PATH = (
    REVIEW_DIR
    / "lmsys_human_review_reviewer1.csv"
)

REVIEWER2_PATH = (
    REVIEW_DIR
    / "lmsys_human_review_reviewer2.csv"
)

REVIEW_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Load computational screening results
# ============================================================

screening_results = pd.read_parquet(
    CHECKPOINT_PATH
)

eligible_cases = screening_results[
    screening_results["eligible"].eq(True)
].copy()


# ============================================================
# Verify uniqueness
# ============================================================

duplicate_count = (
    eligible_cases["source_text_hash"]
    .duplicated()
    .sum()
)

if duplicate_count > 0:
    raise ValueError(
        f"Found {duplicate_count} duplicate eligible cases. "
        "Resolve duplicates before creating the human-review pool."
    )

print(
    "Computationally eligible cases:",
    len(eligible_cases)
)

print()
print(
    "Automated domain distribution:"
)

print(
    eligible_cases["primary_domain"]
    .value_counts()
)


# ============================================================
# Create human-review pool
# ============================================================

# All computationally eligible cases undergo human review.
# random_priority provides a reproducible shuffled presentation order.

review_pool = (
    eligible_cases
    .sort_values("random_priority")
    .reset_index(drop=True)
)

review_pool.insert(
    0,
    "review_id",
    range(1, len(review_pool) + 1)
)


# ============================================================
# Create blinded reviewer files
# ============================================================

# Automated screening judgments, including primary_domain,
# are intentionally omitted so that human reviewers make
# independent judgments.

review_columns = [
    "review_id",
    "source_text_hash",
    "source_text",
    "word_count",
]

reviewer1 = review_pool[
    review_columns
].copy()

reviewer2 = review_pool[
    review_columns
].copy()

for reviewer_df in [
    reviewer1,
    reviewer2,
]:
    reviewer_df["human_eligible"] = ""
    reviewer_df["human_primary_domain"] = ""
    reviewer_df["human_sufficient_context"] = ""
    reviewer_df["human_notes"] = ""


# ============================================================
# Save reviewer files
# ============================================================

reviewer1.to_csv(
    REVIEWER1_PATH,
    index=False,
)

reviewer2.to_csv(
    REVIEWER2_PATH,
    index=False,
)

print()
print(
    "Human-review pool:",
    len(review_pool)
)

print()
print(
    "Reviewer 1 file:",
    REVIEWER1_PATH
)

print(
    "Reviewer 2 file:",
    REVIEWER2_PATH
)
