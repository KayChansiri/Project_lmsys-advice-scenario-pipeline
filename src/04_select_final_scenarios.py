"""
Select the final balanced scenario sample after human adjudication.

This script:
1. Loads the adjudicated human-review file.
2. Retains human-confirmed eligible scenarios with sufficient context.
3. Uses the adjudicated human-assigned domain.
4. Randomly samples 50 scenarios per domain using a fixed seed.
5. Saves the final 150-scenario sample locally.

IMPORTANT:
The input and generated output contain LMSYS source text and should
not be committed to the public repository.
"""

from pathlib import Path
import pandas as pd


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parents[1]

REVIEW_DIR = PROJECT_DIR / "data" / "human_review"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"

FINAL_REVIEW_PATH = (
    REVIEW_DIR / "lmsys_human_review_reviewer_final.xlsx"
)

FINAL_SAMPLE_PATH = (
    PROCESSED_DIR / "lmsys_final_150_scenarios.csv"
)

RANDOM_SEED = 20260916
N_PER_DOMAIN = 50

TARGET_DOMAINS = [
    "relationship",
    "career",
    "emotional_wellbeing",
]

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# Load adjudicated human review
# ------------------------------------------------------------

df = pd.read_excel(FINAL_REVIEW_PATH)

required_columns = [
    "source_text_hash",
    "source_text",
    "human_eligible",
    "human_primary_domain",
    "human_sufficient_context",
]

missing = [c for c in required_columns if c not in df.columns]

if missing:
    raise ValueError(
        f"Missing required columns: {missing}"
    )


# ------------------------------------------------------------
# Integrity checks
# ------------------------------------------------------------

duplicate_count = df["source_text_hash"].duplicated().sum()

if duplicate_count:
    raise ValueError(
        f"Found {duplicate_count} duplicate source_text_hash values. "
        "Resolve duplicates before final sampling."
    )

inconsistent = df[
    (df["human_eligible"] == 1)
    & (df["human_sufficient_context"] == 0)
]

if len(inconsistent):
    raise ValueError(
        f"Found {len(inconsistent)} cases coded eligible but with "
        "insufficient context. Resolve these cases before sampling."
    )


# ------------------------------------------------------------
# Construct adjudicated eligible pool
# ------------------------------------------------------------

eligible = df[
    (df["human_eligible"] == 1)
    & (df["human_sufficient_context"] == 1)
    & (df["human_primary_domain"].isin(TARGET_DOMAINS))
].copy()

print("Human-confirmed eligible pool:")
print(
    eligible["human_primary_domain"]
    .value_counts()
    .reindex(TARGET_DOMAINS, fill_value=0)
)


# ------------------------------------------------------------
# Confirm sufficient cases in every domain
# ------------------------------------------------------------

counts = eligible["human_primary_domain"].value_counts()

for domain in TARGET_DOMAINS:
    n = counts.get(domain, 0)

    if n < N_PER_DOMAIN:
        raise ValueError(
            f"{domain}: only {n} eligible scenarios; "
            f"{N_PER_DOMAIN} required."
        )


# ------------------------------------------------------------
# Reproducibly sample 50 per domain
# ------------------------------------------------------------

selected = (
    eligible
    .groupby("human_primary_domain", group_keys=False)
    .sample(
        n=N_PER_DOMAIN,
        random_state=RANDOM_SEED,
    )
    .reset_index(drop=True)
)

# Stable presentation order after random selection
selected = (
    selected
    .sort_values(
        ["human_primary_domain", "source_text_hash"]
    )
    .reset_index(drop=True)
)

selected.insert(
    0,
    "scenario_id",
    [f"S{i:03d}" for i in range(1, len(selected) + 1)]
)


# ------------------------------------------------------------
# Final checks
# ------------------------------------------------------------

assert len(selected) == 150
assert selected["source_text_hash"].nunique() == 150

for domain in TARGET_DOMAINS:
    assert (
        selected["human_primary_domain"] == domain
    ).sum() == N_PER_DOMAIN

print("\nFinal sample:")
print(selected["human_primary_domain"].value_counts())


# ------------------------------------------------------------
# Save locally
# ------------------------------------------------------------

selected.to_csv(
    FINAL_SAMPLE_PATH,
    index=False,
)

print(f"\nSaved: {FINAL_SAMPLE_PATH}")
