from pathlib import Path
import json
import os
import time
from getpass import getpass

import pandas as pd
from openai import OpenAI
from tqdm.auto import tqdm

PROJECT_DIR = Path(__file__).resolve().parents[1]
RESTRICTED_DIR = PROJECT_DIR / "data" / "restricted"

CANDIDATE_PATH = (
    RESTRICTED_DIR
    / "lmsys_personal_advice_candidates.parquet"
)

CHECKPOINT_PATH = (
    RESTRICTED_DIR
    / "lmsys_llm_screening_checkpoint.parquet"
)

SCREENING_MODEL = "gpt-5.6-luna"
CHECKPOINT_EVERY = 25
MAX_RETRIES = 4

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    api_key = getpass("Paste OpenAI API key: ").strip()

client = OpenAI(api_key=api_key)

SCREENING_INSTRUCTIONS = """
You are screening user-authored messages from a conversational-AI
dataset for a research study of everyday advice-seeking and
decision support.

Evaluate ONLY the supplied first user message.

Do not use outside knowledge.
Do not diagnose the user.
Do not invent missing facts.

A message is ELIGIBLE only when all of the following are true:

1. PERSONAL SITUATION

The user describes a real or apparently real situation involving
themselves, their relationships, work/career, or emotional
wellbeing.

2. ADVICE OR DECISION SUPPORT

The user explicitly OR implicitly seeks help with a personal
decision, response, course of action, coping problem, or
interpersonal problem.

Explicit phrases such as "What should I do?" are NOT required.

A message that merely asks for factual information does not qualify.

3. TARGET DOMAIN

Choose the ONE primary domain that best represents the user's
main decision or support need.

RELATIONSHIP:
Romantic relationships, family relationships, friendships,
roommates, interpersonal boundaries, trust, conflict,
communication, breakup decisions, relationship continuation, or
similar interpersonal situations.

CAREER:
Jobs, workplace issues, career direction, job changes, job
offers, employment, applications, professional development,
education-to-career decisions, promotions, resignation, salary,
workplace conflict, or similar career situations.

EMOTIONAL_WELLBEING:
Everyday emotional distress or wellbeing, loneliness, grief,
stress, burnout, anxiety, low mood, self-esteem, coping,
support-seeking, motivation, emotional boundaries, or similar
non-diagnostic support needs.

4. SUFFICIENT CONTEXT

The message contains enough substantive context to reconstruct
the underlying situation and decision/support problem without
inventing major facts.

EXCLUDE messages whose primary purpose is:

- factual information seeking
- coding or technical assistance
- summarization
- translation
- editing or rewriting
- content generation
- fictional writing
- roleplay
- benchmark or classification tasks
- homework without a genuine personal decision
- general discussion without a personal situation
- medical diagnosis
- symptom diagnosis
- medication instructions
- treatment recommendations
- requests primarily asking what medical or psychiatric condition
  the user has

IMPORTANT MENTAL-HEALTH RULE:

Do NOT exclude a message merely because it mentions depression,
anxiety, therapy, suicide, self-harm, or another mental-health
topic.

It may still be eligible as emotional wellbeing when the primary
request concerns everyday coping, decision-making, support,
relationships, or help-seeking rather than diagnosis or
treatment.

DOMAIN ASSIGNMENT:

Select ONE primary domain:

relationship
career
emotional_wellbeing
other
ambiguous

Use "ambiguous" only if two or more target domains are genuinely
equally central and no primary decision can reasonably be
identified.

Return only the requested structured fields.
"""

SCREENING_SCHEMA = {
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
        "clinical_diagnosis_or_treatment_request": {
            "type": "boolean"
        },
        "eligible": {"type": "boolean"},
        "exclusion_reason": {
            "type": "string",
            "enum": [
                "none",
                "not_personal",
                "not_advice_or_decision_support",
                "outside_target_domains",
                "insufficient_context",
                "clinical_diagnosis_or_treatment",
                "content_generation_or_roleplay",
                "factual_or_technical_request",
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
        "eligible",
        "exclusion_reason",
    ],
    "additionalProperties": False,
}


def classify_message(text):
    response = client.responses.create(
        model=SCREENING_MODEL,
        reasoning={"effort": "low"},
        instructions=SCREENING_INSTRUCTIONS,
        input=text,
        text={
            "format": {
                "type": "json_schema",
                "name": "advice_screening",
                "schema": SCREENING_SCHEMA,
                "strict": True,
            }
        },
    )

    result = json.loads(response.output_text)

    result["eligible"] = (
        result["personal_situation"]
        and result["advice_or_decision_support"]
        and result["primary_domain"] in {
            "relationship",
            "career",
            "emotional_wellbeing",
        }
        and result["sufficient_context"]
        and not result["clinical_diagnosis_or_treatment_request"]
    )

    return result


candidates = pd.read_parquet(CANDIDATE_PATH)

if CHECKPOINT_PATH.exists():
    screening_results = pd.read_parquet(CHECKPOINT_PATH)
else:
    screening_results = pd.DataFrame()

already_screened = (
    set(screening_results["source_text_hash"])
    if len(screening_results) > 0
    else set()
)

unscreened = candidates[
    ~candidates["source_text_hash"].isin(already_screened)
].copy()

new_results = []

for _, row in tqdm(
    unscreened.iterrows(),
    total=len(unscreened),
    desc="LLM screening",
):
    classification = None
    last_error = None

    for attempt in range(MAX_RETRIES):
        try:
            classification = classify_message(row["source_text"])
            break
        except Exception as exc:
            last_error = str(exc)
            time.sleep(2 ** attempt)

    result_row = {
        "source_text_hash": row["source_text_hash"],
        "source_text": row["source_text"],
        "random_priority": row["random_priority"],
        "word_count": row.get("word_count"),
        "relationship_signal": row.get("relationship_signal"),
        "career_signal": row.get("career_signal"),
        "wellbeing_signal": row.get("wellbeing_signal"),
        "decision_support_signal": row.get("decision_support_signal"),
    }

    if classification is not None:
        result_row.update(classification)
        result_row["api_error"] = None
    else:
        result_row["api_error"] = last_error
        result_row["eligible"] = False
        result_row["primary_domain"] = None

    new_results.append(result_row)

    if len(new_results) % CHECKPOINT_EVERY == 0:
        combined = pd.concat(
            [screening_results, pd.DataFrame(new_results)],
            ignore_index=True,
        )

        combined = (
            combined
            .drop_duplicates(subset="source_text_hash", keep="last")
            .reset_index(drop=True)
        )

        combined.to_parquet(CHECKPOINT_PATH, index=False)

if new_results:
    screening_results = pd.concat(
        [screening_results, pd.DataFrame(new_results)],
        ignore_index=True,
    )

    screening_results = (
        screening_results
        .drop_duplicates(subset="source_text_hash", keep="last")
        .reset_index(drop=True)
    )

    screening_results.to_parquet(CHECKPOINT_PATH, index=False)

eligible = screening_results[
    screening_results["eligible"].eq(True)
]

print("Total screened:", len(screening_results))
print()
print("Eligible by primary domain:")
print(eligible["primary_domain"].value_counts())
print()
print("Saved:", CHECKPOINT_PATH)
