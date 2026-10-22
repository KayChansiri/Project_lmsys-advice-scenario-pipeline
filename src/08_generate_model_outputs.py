"""FAccT 2027: GPT-6 Sol model-output generation.

Design: 150 scenarios x 3 communicative orientations x 3 reasoning-effort
conditions x 5 independent generations = 6,750 responses.

Set OPENAI_API_KEY in the environment before running.
"""

from pathlib import Path
import hashlib
import os
import random
import time
import pandas as pd
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT / "data/processed/communicative_style_generation.csv"
OUTPUT_PATH = ROOT / "data/model_outputs/gpt6sol_advice_generation_6750.csv"
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

MODEL = "gpt-6-sol"
REASONING_LEVELS = ["none", "medium", "high"]
N_REPLICATIONS = 5
EXPECTED_SCENARIOS = 150
EXPECTED_STIMULI = 450
EXPECTED_TOTAL = 6750
GENERATION_ORDER_SEED = 20261019

RESPONSE_INSTRUCTION = """Respond to the user's message as an AI assistant. Provide guidance based on the information
the user provides. Respond naturally as you would to a user seeking advice.""".strip()

api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("Set OPENAI_API_KEY before running this script.")
client = OpenAI(api_key=api_key)


def text_hash(text):
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def extract_usage(response):
    usage = getattr(response, "usage", None)
    if usage is None:
        return dict(input_tokens=None, output_tokens=None,
                    reasoning_tokens=None, total_tokens=None)

    details = getattr(usage, "output_tokens_details", None)
    return {
        "input_tokens": getattr(usage, "input_tokens", None),
        "output_tokens": getattr(usage, "output_tokens", None),
        "reasoning_tokens": (
            getattr(details, "reasoning_tokens", None)
            if details is not None else None
        ),
        "total_tokens": getattr(usage, "total_tokens", None),
    }


def generate_response(user_message, reasoning_effort):
    # reasoning.effort is the only experimental API parameter varied.
    # temperature and top_p are not explicitly set.
    response = client.responses.create(
        model=MODEL,
        instructions=RESPONSE_INSTRUCTION,
        input=user_message,
        reasoning={"effort": reasoning_effort},
    )
    output_text = response.output_text.strip()
    if not output_text:
        raise ValueError("OpenAI returned an empty response.")
    return output_text, response.id, extract_usage(response)


# Load and verify the 450 transformed stimuli.
stimuli = pd.read_csv(INPUT_PATH)
assert len(stimuli) == EXPECTED_STIMULI
required = [
    "scenario_id", "human_primary_domain", "style",
    "source_text", "transformed_text"
]
assert all(c in stimuli.columns for c in required)
assert stimuli["scenario_id"].nunique() == EXPECTED_SCENARIOS
assert set(stimuli["style"].dropna().unique()) == {
    "experiential", "analytical", "integrated"
}
assert (stimuli.groupby("scenario_id")["style"].nunique() == 3).all()
assert stimuli["transformed_text"].notna().all()
assert stimuli["transformed_text"].astype(str).str.strip().ne("").all()

stimuli["stimulus_id"] = (
    stimuli["scenario_id"].astype(str) + "_" + stimuli["style"].astype(str)
)
assert stimuli["stimulus_id"].nunique() == EXPECTED_STIMULI
stimuli["stimulus_text_hash"] = stimuli["transformed_text"].apply(text_hash)

# Build full factorial generation plan.
rows = []
for _, row in stimuli.iterrows():
    for effort in REASONING_LEVELS:
        for rep in range(1, N_REPLICATIONS + 1):
            rows.append({
                "generation_id": f"{row['stimulus_id']}_{effort}_R{rep}",
                "stimulus_id": row["stimulus_id"],
                "scenario_id": row["scenario_id"],
                "human_primary_domain": row["human_primary_domain"],
                "style": row["style"],
                "reasoning_effort": effort,
                "replication": rep,
                "source_text": row["source_text"],
                "transformed_text": row["transformed_text"],
                "stimulus_text_hash": row["stimulus_text_hash"],
            })

plan = pd.DataFrame(rows)
assert len(plan) == EXPECTED_TOTAL
assert plan["generation_id"].nunique() == EXPECTED_TOTAL

balance = plan.groupby(["style", "reasoning_effort"]).size().unstack()
assert (balance == 750).all().all()

# Randomize generation order reproducibly.
plan = plan.sample(
    frac=1, random_state=GENERATION_ORDER_SEED
).reset_index(drop=True)
plan["generation_order"] = range(1, len(plan) + 1)

# Resume from checkpoint if one exists.
if OUTPUT_PATH.exists():
    existing = pd.read_csv(OUTPUT_PATH).drop_duplicates(
        subset=["generation_id"], keep="first"
    )
    results = existing.to_dict("records")
    completed = set(existing["generation_id"].astype(str))
    print(f"Resuming: {len(completed)}/{EXPECTED_TOTAL} complete.")
else:
    results = []
    completed = set()
    print("Starting new generation run.")

# Generate and checkpoint after every successful response.
for _, row in plan.iterrows():
    generation_id = str(row["generation_id"])
    if generation_id in completed:
        continue

    print(f"[{row['generation_order']}/{EXPECTED_TOTAL}] {generation_id}")

    try:
        text, response_id, usage = generate_response(
            row["transformed_text"], row["reasoning_effort"]
        )

        record = {
            **row.to_dict(),
            "response_text": text,
            "response_word_count": len(text.split()),
            "response_character_count": len(text),
            "response_id": response_id,
            "model": MODEL,
            "reasoning_mode": "standard",
            "response_instruction": RESPONSE_INSTRUCTION,
            **usage,
        }
        results.append(record)
        pd.DataFrame(results).to_csv(OUTPUT_PATH, index=False)
        completed.add(generation_id)

        print(
            f"   saved | {record['response_word_count']} words | "
            f"reasoning={row['reasoning_effort']}"
        )
        time.sleep(random.uniform(0.15, 0.35))

    except Exception as exc:
        print(f"Generation stopped at {generation_id}: {exc}")
        print("Completed responses are saved. Rerun to resume.")
        break

# Final check.
final = pd.read_csv(OUTPUT_PATH).drop_duplicates(
    subset=["generation_id"], keep="first"
)
n_complete = final["generation_id"].nunique()
n_duplicates = final.duplicated(subset=["generation_id"]).sum()

print(f"Completed responses: {n_complete}/{EXPECTED_TOTAL}")
print(f"Duplicate generation IDs: {n_duplicates}")

if n_complete == EXPECTED_TOTAL and n_duplicates == 0:
    print("Generation complete.")
    print(f"Output: {OUTPUT_PATH}")
