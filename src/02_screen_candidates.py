"""
Reproducibility pipeline for LMSYS-Chat-1M advice-scenario construction.

IMPORTANT
---------
- LMSYS-Chat-1M is gated. Do not redistribute source conversation text.
- Supply credentials through environment variables; never commit API keys/tokens.
- These scripts preserve the historical study output basenames used downstream.
"""

import os
import json
import time
from pathlib import Path

import pandas as pd
from openai import OpenAI
from tqdm.auto import tqdm

# ---------------- Configuration ----------------
PROJECT_DIR = Path(os.environ.get("FAACT_PROJECT_DIR", ".")).resolve()
RESTRICTED_DIR = PROJECT_DIR / "data" / "restricted"

INPUT_PATH = RESTRICTED_DIR / "lmsys_personal_advice_candidates.parquet"
CHECKPOINT_PATH = RESTRICTED_DIR / "lmsys_llm_screening_checkpoint.parquet"

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
SCREENING_MODEL = "gpt-5.6-luna"
CHECKPOINT_EVERY = 25
MAX_RETRIES = 4

if not OPENAI_API_KEY:
    raise RuntimeError("Set OPENAI_API_KEY in your environment before running.")

client = OpenAI(api_key=OPENAI_API_KEY)
candidates = pd.read_parquet(INPUT_PATH)

SCREENING_INSTRUCTIONS = """
Evaluate only the supplied first user message. Do not use outside knowledge and do not
invent missing facts.

Determine whether the message:
1. describes a real or apparently real personal situation involving the user, their
   relationships, work/career, or emotional wellbeing;
2. explicitly or implicitly seeks personal advice, decision support, interpretation,
   reflection, coping support, or guidance about what to think, feel, decide, or do;
3. has a primary support need in relationship, career, or emotional_wellbeing;
4. contains enough context for a context-sensitive response without inventing major facts; and
5. is not primarily a request for medical/psychiatric diagnosis, symptom diagnosis,
   medication instructions, or treatment recommendations.

Content generation/editing, factual/instrumental information retrieval, roleplay or
response-constraining persona instructions, and tasks outside the target domains are not
eligible merely because they contain personal context.

For multi-domain cases, assign the domain corresponding to the primary object of support.
"""

SCHEMA = {
    "type": "object",
    "properties": {
        "personal_situation": {"type": "boolean"},
        "advice_or_decision_support": {"type": "boolean"},
        "primary_domain": {
            "type": "string",
            "enum": [
                "relationship",
                "career",
                "emotional_wellbeing",
                "other",
                "ambiguous",
            ],
        },
        "sufficient_context": {"type": "boolean"},
        "clinical_diagnosis_or_treatment_request": {"type": "boolean"},
        "exclusion_reason": {
            "type": "string",
            "enum": [
                "none",
                "not_personal",
                "not_advice_or_decision_support",
                "outside_target_domains",
                "insufficient_context",
                "clinical_diagnosis_or_treatment",
                "factual_or_technical_request",
                "content_generation_or_roleplay",
                "other",
            ],
        },
    },
    "required": [
        "personal_situation",
        "advice_or_decision_support",
        "primary_domain",
        "sufficient_context",
        "clinical_diagnosis_or_treatment_request",
        "exclusion_reason",
    ],
    "additionalProperties": False,
}

TARGET_DOMAINS = {"relationship", "career", "emotional_wellbeing"}

def classify_message(text):
    response = client.responses.create(
        model=SCREENING_MODEL,
        reasoning={"effort": "low"},
        instructions=SCREENING_INSTRUCTIONS,
        input=text,
        text={
            "format": {
                "type": "json_schema",
                "name": "scenario_screening",
                "schema": SCHEMA,
                "strict": True,
            }
        },
    )
    result = json.loads(response.output_text)

    # Recompute overall eligibility from component judgments.
    result["eligible"] = bool(
        result["personal_situation"]
        and result["advice_or_decision_support"]
        and result["primary_domain"] in TARGET_DOMAINS
        and result["sufficient_context"]
        and not result["clinical_diagnosis_or_treatment_request"]
    )
    return result

# ---------------- Safe resume ----------------
candidate_hashes = set(candidates["source_text_hash"])

if CHECKPOINT_PATH.exists():
    prior = pd.read_parquet(CHECKPOINT_PATH)
    prior = (
        prior[prior["source_text_hash"].isin(candidate_hashes)]
        .drop_duplicates(subset="source_text_hash", keep="last")
        .reset_index(drop=True)
    )
    print(f"Valid checkpoint rows loaded: {len(prior):,}")
else:
    prior = pd.DataFrame()

already = set(prior["source_text_hash"]) if len(prior) else set()
results = prior.to_dict(orient="records")

for _, row in tqdm(candidates.iterrows(), total=len(candidates), desc="LLM screening"):
    source_hash = row["source_text_hash"]
    if source_hash in already:
        continue

    classification = None
    last_error = None

    for attempt in range(MAX_RETRIES):
        try:
            classification = classify_message(row["source_text"])
            break
        except Exception as exc:
            last_error = str(exc)
            time.sleep(2 ** attempt)

    result = row.to_dict()
    if classification is not None:
        result.update(classification)
        result["api_error"] = None
    else:
        result.update({
            "personal_situation": None,
            "advice_or_decision_support": None,
            "primary_domain": None,
            "sufficient_context": None,
            "clinical_diagnosis_or_treatment_request": None,
            "exclusion_reason": "other",
            "eligible": False,
            "api_error": last_error,
        })

    results.append(result)
    already.add(source_hash)

    if len(results) % CHECKPOINT_EVERY == 0:
        pd.DataFrame(results).to_parquet(CHECKPOINT_PATH, index=False)

screening = (
    pd.DataFrame(results)
    .drop_duplicates(subset="source_text_hash", keep="last")
    .reset_index(drop=True)
)

assert screening["source_text_hash"].isin(candidate_hashes).all()
screening.to_parquet(CHECKPOINT_PATH, index=False)

eligible = screening[screening["eligible"].eq(True)].copy()
print(f"Screened current candidates: {len(screening):,}")
print(f"Eligible cases: {len(eligible):,}")
print(eligible["primary_domain"].value_counts(dropna=False))
print(f"Saved checkpoint: {CHECKPOINT_PATH}")

historical = {
    "relationship": 256,
    "career": 257,
    "emotional_wellbeing": 111,
}
if len(eligible) != 624:
    print(
        f"WARNING: current run produced {len(eligible):,} eligible cases; "
        "the historical study run produced 624. Do not overwrite adjudicated "
        "historical study files until the difference is audited."
    )
for domain, expected in historical.items():
    observed = int((eligible["primary_domain"] == domain).sum())
    if observed != expected:
        print(f"WARNING: {domain}={observed}; historical study run={expected}.")
