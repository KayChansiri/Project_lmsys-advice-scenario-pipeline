"""
05_generate_communicative_styles.py

Generate matched System 1-oriented, System 2-oriented, and mixed
communicative-style versions of the final decision scenarios.

The script expects:
    data/processed/lmsys_final_150_scenarios.csv

Required input columns:
    scenario_id
    human_primary_domain
    source_text

Authentication:
    Set the Hugging Face token as an environment variable:

        export HF_TOKEN="your_token_here"

The script does not redistribute source LMSYS messages or generated
stimuli. These data files should remain excluded from the public
repository.

One safety-sensitive scenario (S117) is intentionally skipped because
the transformation model refused to transform it. Its matched variants
are handled separately under the study's documented exception procedure.
"""

from pathlib import Path
import json
import os
import re
import time

import pandas as pd
from huggingface_hub import InferenceClient


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "lmsys_final_150_scenarios.csv"
)

OUTPUT_DIR = PROJECT_DIR / "data" / "processed"

CHECKPOINT_PATH = (
    OUTPUT_DIR
    / "communicative_style_generation_checkpoint.csv"
)

MODEL = "meta-llama/Llama-3.3-70B-Instruct"

TEMPERATURE = 0.3
TOP_P = 1.0
MAX_TOKENS = 1500

REQUIRED_STYLES = ["system1", "system2", "mixed"]

# This scenario triggered a safety refusal from the transformation model
# and is handled separately using the documented exception procedure.
SAFETY_EXCEPTION_IDS = {"S117"}


# ---------------------------------------------------------------------
# Hugging Face client
# ---------------------------------------------------------------------

HF_TOKEN = os.environ.get("HF_TOKEN")

if not HF_TOKEN:
    raise RuntimeError(
        "HF_TOKEN environment variable is not set. "
        "Set it before running this script."
    )

client = InferenceClient(
    api_key=HF_TOKEN,
    provider="auto"
)


# ---------------------------------------------------------------------
# Common preservation requirements
# ---------------------------------------------------------------------

COMMON_RULES = """
These requirements apply to ALL communicative-style conditions.

CONTENT PRESERVATION

The transformation must preserve the underlying scenario. The three
versions should differ in communicative style, not in substantive
information.

1. Preserve all substantive facts in the original message, including
   people, relationships, events, circumstances, and constraints.

2. Preserve the same underlying problem, decision, uncertainty, or
   support need.

3. Preserve all safety-, risk-, or vulnerability-relevant information.

4. Do not add new facts, events, motivations, preferences, emotions,
   alternatives, constraints, consequences, probabilities, or
   contextual information that are not supported by the original.

5. Do not remove information that could meaningfully affect how an
   advisor would understand or respond to the situation.

6. Do not change the user's intended question or what they are seeking
   help with.

7. Do not resolve ambiguities that are unresolved in the original
   message. If the original user is uncertain about something, preserve
   that uncertainty.

COMMUNICATIVE PRESERVATION

8. Preserve the user's general level of informality and conversational
   style. Do not unnecessarily improve grammar, spelling, vocabulary,
   fluency, or sophistication.

9. Do not make the user appear more or less intelligent, competent,
   educated, rational, or emotionally stable across conditions.

10. Do not increase or decrease the seriousness, urgency, or risk of
    the situation.

11. Do not provide advice, answer the question, or introduce an
    assistant perspective.

12. Keep the message concise and reasonably close to the length of the
    original. Some expansion is allowed when needed to express the target
    communicative style naturally, especially for short source messages.
    Do not add unnecessary detail merely to increase the length.

13. The three communicative-style versions of the same scenario should
    be approximately comparable in length and informational content.
"""


# ---------------------------------------------------------------------
# Communicative-style definitions
# ---------------------------------------------------------------------

STYLE_INSTRUCTIONS = {

    "system1": """
SYSTEM 1-ORIENTED COMMUNICATION

Drawing on dual-process accounts of judgment and decision making,
System 1-oriented communication reflects relatively intuitive,
automatic, affective, and impression-based processing.

Express the SAME underlying scenario in a way that relatively
foregrounds:
- immediate reactions or intuitive impressions;
- feelings or affective concerns when supported by the source;
- what feels salient or important to the user;
- intuitive expressions of uncertainty; and
- relatively spontaneous, conversational reasoning.

Do not systematically organize the request around explicit criteria,
comparisons, probabilities, or structured tradeoffs unless doing so
is necessary to preserve information already contained in the source.

System 1-oriented communication does NOT mean irrational, impulsive,
less intelligent, more distressed, or more emotionally extreme.
""",

    "system2": """
SYSTEM 2-ORIENTED COMMUNICATION

Drawing on dual-process accounts of judgment and decision making,
System 2-oriented communication reflects relatively deliberative,
analytical, effortful, and explicitly reasoned processing.

Express the SAME underlying scenario in a way that relatively
foregrounds:
- deliberate consideration of the situation;
- relevant facts and constraints;
- explicit reasoning about considerations relevant to the decision;
- alternatives, consequences, or tradeoffs when supported by the
  source; and
- more structured articulation of uncertainty and the decision problem.

Do not invent alternatives, criteria, consequences, probabilities,
or facts merely to make the message appear analytical.

System 2-oriented communication does NOT mean more intelligent,
knowledgeable, formal, emotionally detached, or sophisticated.
""",

    "mixed": """
MIXED COMMUNICATION

Express the SAME underlying scenario using meaningful characteristics
of both System 1-oriented and System 2-oriented communication.

The message should:
- communicate immediate reactions, intuitions, or feelings when
  supported by the source;
- also make relevant reasoning, constraints, considerations, or
  tradeoffs explicit when supported by the source;
- move naturally between intuitive/affective and
  deliberative/analytical expression; and
- avoid having either orientation clearly dominate.

The result should sound like one natural user message, not two
different styles artificially combined.
"""
}


# ---------------------------------------------------------------------
# Transformation function
# ---------------------------------------------------------------------

def transform_matched_set(source_text):
    """
    Generate the three communicative-style versions jointly as a
    matched set.
    """

    prompt = f"""
You are transforming a naturally occurring user message for a
controlled research study.

Create THREE matched versions of the same user message:

1. system1
2. system2
3. mixed

All three versions must preserve the SAME underlying situation,
substantive information, and decision-support need.

REQUIREMENTS FOR ALL CONDITIONS:

{COMMON_RULES}

TARGET STYLE: SYSTEM 1

{STYLE_INSTRUCTIONS["system1"]}

TARGET STYLE: SYSTEM 2

{STYLE_INSTRUCTIONS["system2"]}

TARGET STYLE: MIXED

{STYLE_INSTRUCTIONS["mixed"]}

MATCHING REQUIREMENT:

Generate the three versions as a matched set.

Moderate expansion from the source is allowed when needed to express
the communicative styles naturally, especially for short messages.

Keep the three transformed versions reasonably comparable to one
another in length and informational content.

Do not make one condition substantially more detailed, informative,
or elaborate than the others.

Do not introduce new objective facts, events, achievements,
relationships, circumstances, constraints, alternatives, or
consequences that are not present in the source.

Do not introduce or strengthen the user's position on what outcome is
deserved, justified, correct, or preferred unless that position is
already present in the source.

ORIGINAL USER MESSAGE:

{source_text}

Return ONLY valid JSON in exactly this structure:

{{
  "system1": "rewritten message",
  "system2": "rewritten message",
  "mixed": "rewritten message"
}}
"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=TEMPERATURE,
        top_p=TOP_P,
        max_tokens=MAX_TOKENS
    )

    text = response.choices[0].message.content

    if text is None:
        raise ValueError("Model returned no text.")

    text = text.strip()

    # Remove possible Markdown code fences.
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    # Extract the JSON object if surrounding text was returned.
    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1:
        raise ValueError(
            f"No JSON object found in model response: {text!r}"
        )

    transformed = json.loads(text[start:end + 1])

    if not all(style in transformed for style in REQUIRED_STYLES):
        raise ValueError(
            "Model response is missing one or more required styles."
        )

    return transformed


# ---------------------------------------------------------------------
# Main generation procedure
# ---------------------------------------------------------------------

def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_PATH}"
        )

    scenarios = pd.read_csv(INPUT_PATH)

    required_columns = {
        "scenario_id",
        "human_primary_domain",
        "source_text"
    }

    missing_columns = required_columns - set(scenarios.columns)

    if missing_columns:
        raise ValueError(
            f"Input file is missing columns: {sorted(missing_columns)}"
        )

    print(f"Loaded {len(scenarios)} scenarios.")

    # --------------------------------------------------------------
    # Resume from checkpoint
    # --------------------------------------------------------------

    if CHECKPOINT_PATH.exists():

        results_df = pd.read_csv(CHECKPOINT_PATH)
        results = results_df.to_dict("records")

        # A scenario counts as completed only if all three styles exist.
        completed_ids = set()

        for scenario_id, group in results_df.groupby("scenario_id"):
            if set(group["style"]) >= set(REQUIRED_STYLES):
                completed_ids.add(scenario_id)

        print(
            f"Resuming generation: "
            f"{len(completed_ids)} scenarios already completed."
        )

    else:

        results = []
        completed_ids = set()

        print("Starting generation from the beginning.")

    # --------------------------------------------------------------
    # Generate matched sets
    # --------------------------------------------------------------

    for position, (_, row) in enumerate(
        scenarios.iterrows(),
        start=1
    ):

        scenario_id = row["scenario_id"]

        if scenario_id in completed_ids:
            continue

        domain = row["human_primary_domain"]
        source_text = row["source_text"]

        print(
            f"[{position}/{len(scenarios)}] "
            f"{scenario_id} ({domain})..."
        )

        # ----------------------------------------------------------
        # Documented safety exception
        # ----------------------------------------------------------

        if scenario_id in SAFETY_EXCEPTION_IDS:
            print(
                f"   Skipping {scenario_id}: "
                "transformation-model refusal; "
                "handled separately under the documented "
                "exception procedure."
            )
            continue

        try:

            transformed = transform_matched_set(source_text)

            # Add all three conditions only after a complete matched
            # set has been successfully returned.
            new_rows = []

            for style in REQUIRED_STYLES:

                transformed_text = transformed[style].strip()

                if not transformed_text:
                    raise ValueError(
                        f"Empty transformation for style: {style}"
                    )

                new_rows.append({
                    "scenario_id": scenario_id,
                    "human_primary_domain": domain,
                    "style": style,
                    "source_text": source_text,
                    "transformed_text": transformed_text,
                    "source_word_count": len(source_text.split()),
                    "transformed_word_count":
                        len(transformed_text.split()),
                    "model": MODEL,
                    "temperature": TEMPERATURE,
                    "top_p": TOP_P,
                    "provider_requested": "auto",
                    "transformation_protocol": "standard"
                })

            results.extend(new_rows)

            # Save after every completed scenario.
            pd.DataFrame(results).to_csv(
                CHECKPOINT_PATH,
                index=False
            )

            completed_ids.add(scenario_id)

            print("   ✓ saved")

            # Brief pause between API calls.
            time.sleep(1)

        except Exception as exc:

            print(f"   ✗ ERROR: {exc}")
            print(
                "Progress has been preserved in the checkpoint. "
                "Inspect the failed scenario before continuing."
            )

            break

    # --------------------------------------------------------------
    # Summary
    # --------------------------------------------------------------

    if CHECKPOINT_PATH.exists():

        generated = pd.read_csv(CHECKPOINT_PATH)

        print("\nGeneration summary")
        print("-" * 50)

        print(
            "Generated rows:",
            len(generated)
        )

        print(
            "Unique generated scenarios:",
            generated["scenario_id"].nunique()
        )

        print("\nRows by style:")
        print(
            generated["style"]
            .value_counts()
            .reindex(REQUIRED_STYLES, fill_value=0)
        )

        print("\nUnique scenarios by domain:")
        print(
            generated
            .groupby("human_primary_domain")["scenario_id"]
            .nunique()
        )

        expected_ids = set(scenarios["scenario_id"])
        generated_ids = set(generated["scenario_id"])

        print("\nScenarios not generated by this script:")
        print(sorted(expected_ids - generated_ids))

        duplicates = generated.duplicated(
            subset=["scenario_id", "style"]
        ).sum()

        print(
            "\nDuplicate scenario × style rows:",
            duplicates
        )


if __name__ == "__main__":
    main()
