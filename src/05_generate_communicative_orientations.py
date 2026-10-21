"""
Generate CEST-grounded communicative-orientation variants for the 150 final scenarios.

Conditions:
- experiential: high experiential expression, low explicit analytical organization
- analytical: high analytical organization, low experiential foregrounding
- integrated: high experiential expression and high analytical organization

This public reproducibility script is derived from the executed study notebook.
It contains no service credentials. Set HF_TOKEN in the environment.

Input:
    data/processed/lmsys_final_150_scenarios.csv

Outputs:
    data/processed/communicative_style_generation.csv
    data/processed/communicative_style_generation_failures.csv (if failures occur)

Hosted inference behavior can change over time, so future reruns are not guaranteed
to reproduce identical generated text.
"""

import json
import os
import re
import time
from pathlib import Path

import pandas as pd
from huggingface_hub import InferenceClient

PROJECT_DIR = Path(os.environ.get("FAACT_PROJECT_DIR", ".")).resolve()
INPUT_PATH = PROJECT_DIR / "data" / "processed" / "lmsys_final_150_scenarios.csv"
OUTPUT_DIR = PROJECT_DIR / "data" / "processed"
OUTPUT_PATH = OUTPUT_DIR / "communicative_style_generation.csv"
FAILED_PATH = OUTPUT_DIR / "communicative_style_generation_failures.csv"

MODEL = "meta-llama/Llama-3.3-70B-Instruct"
TEMPERATURE = 0.3
TOP_P = 1.0
MAX_TOKENS = 10000
PROVIDER_REQUESTED = "auto"
TRANSFORMATION_PROTOCOL = "cest_communicative_orientation_v1"
REQUIRED_ORIENTATIONS = ["experiential", "analytical", "integrated"]

HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    raise RuntimeError("Set HF_TOKEN in your environment before running.")

client = InferenceClient(api_key=HF_TOKEN, provider=PROVIDER_REQUESTED)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


import json
import re


def transform_matched_set(source_text):

    prompt = f"""
You are creating three experimentally distinct versions of the SAME
naturally occurring advice-seeking message.

The manipulation is grounded in Cognitive-Experiential Self-Theory (CEST),
which distinguishes experiential and rational/analytical processing.

In this study, these theoretical characteristics are operationalized as
COMMUNICATIVE ORIENTATIONS: observable differences in how the same situation
and advice-seeking need are expressed.

The manipulation must be STRONG and CLEAR while preserving the same
underlying facts, uncertainty, stance, and decision problem.


============================================================
THE THREE CONDITIONS
============================================================

EXPERIENTIAL = HIGH experiential expression, LOW explicit analytical
organization.

The message should communicate the problem primarily through lived
experience: feelings, intuitions, impressions, immediate reactions,
subjective uncertainty, personal salience, and the way concerns arise in
the user's experience.

The user may mention facts and practical considerations, but should NOT
systematically decompose the problem into explicit criteria, alternatives,
pros and cons, consequences, or structured tradeoffs unless doing so is
necessary to preserve the source.

The message may unfold narratively or associatively rather than as an
explicitly organized decision analysis.


ANALYTICAL = HIGH analytical organization, LOW experiential foregrounding.

The message should communicate the problem primarily through explicit
deliberation. Organize the information around relevant facts, constraints,
priorities, alternatives, consequences, uncertainties, and tradeoffs that
are already supported by the source.

Make relationships among these considerations explicit. When the source
contains competing considerations, explicitly contrast or weigh them.

Preserve feelings and subjective experiences that are substantively part of
the source, but report them as relevant information rather than allowing
them to organize the message.


INTEGRATED = HIGH experiential expression AND HIGH analytical organization.

The message must simultaneously contain a strong experiential layer AND a
strong analytical layer.

The user's feelings, intuitions, subjective uncertainty, and lived
experience should be clearly foregrounded. At the same time, the user should
explicitly identify and deliberate over relevant constraints, alternatives,
consequences, priorities, or tradeoffs supported by the source.

These two modes should interact. For example, the user may explain what they
feel pulled toward while explicitly examining the practical considerations
that support or conflict with that feeling.

Integrated must NOT be:
- a midpoint between Experiential and Analytical;
- an Analytical message with a few emotional words added;
- an Experiential message with a list of facts added.

A reader should clearly perceive BOTH strong experiential expression AND
strong analytical deliberation.


============================================================
SOURCE MESSAGE
============================================================

{source_text}


============================================================
TRANSFORMATION TASK
============================================================

Rewrite the source into exactly:

1. experiential
2. analytical
3. integrated

The conditions should be sufficiently distinct that a human reviewer who
knows the definitions above can reliably distinguish them.

Use the SAME substantive information to create the contrast.

For EXPERIENTIAL, reorganize the information around how the situation is
being experienced.

For ANALYTICAL, reorganize the information around explicit consideration
of the decision and its relevant factors.

For INTEGRATED, make BOTH the lived experience and explicit deliberation
substantive organizing features of the message.


============================================================
INVARIANCE AND MATCHING REQUIREMENTS
============================================================

This is controlled rewriting, not summarization or creation of a new
scenario.

Across all three versions, preserve the source's:

- substantive facts and events;
- constraints and preferences;
- relevant feelings and concerns;
- uncertainty and ambiguity;
- degree of confidence or indecision;
- underlying stance;
- risks and consequences;
- decision problem;
- advice-seeking goal.

Do not replace several specific source details with a broad summary when
those details are relevant to the situation.

Do not invent facts, feelings, motivations, criteria, alternatives,
consequences, or preferences.

Most importantly, COMMUNICATIVE ORIENTATION must not change the user's
underlying position.



============================================================
LENGTH MATCHING
============================================================

The EXPERIENTIAL, ANALYTICAL, and INTEGRATED versions must be approximately
equal in length to ONE ANOTHER.

No condition should be systematically longer or shorter than the others.

In particular, do NOT make the Integrated version longer simply because it
contains both experiential and analytical characteristics. Express both
orientations through how the same information is organized, not by adding
more information or more words.

Aim for closely comparable word counts across the three versions. Small
natural differences are acceptable, but avoid substantial differences in
length.

The transformed versions do not need to match the source's exact word count.
However, preserve the source's substantive level of detail and do not
substantially summarize a detailed source merely to make the transformations
shorter.

Length must NOT be used as part of the experimental manipulation.


============================================================
CONTRAST AND MATCHING CHECK
============================================================

Before returning the answer, compare all three versions and revise them
internally if necessary.

EXPERIENTIAL:
Does subjective experience clearly organize the message, with explicit
decision analysis kept comparatively low?

ANALYTICAL:
Does explicit deliberation clearly organize the message, with experiential
material preserved but comparatively less foregrounded?

INTEGRATED:
Are BOTH experiential expression and explicit analytical deliberation
strong and substantive?

SEMANTIC MATCH:
Do all three versions preserve the same substantive information, stance,
uncertainty, and advice-seeking problem?

LENGTH MATCH:
Are the three versions approximately equal in length, with no condition
substantially longer or shorter than the others?

If Experiential and Analytical could easily be mistaken for one another,
increase the contrast in HOW the same information is organized.

If Integrated resembles Analytical with emotional adjectives added, revise
it so that lived experience genuinely organizes part of the message.

If one version is substantially longer or shorter than the other two,
revise it so the three versions have more comparable lengths WITHOUT
changing their substantive information or communicative orientations.

The contrast must come from communicative orientation, NOT from differences
in intelligence, grammar quality, vocabulary sophistication, formality,
politeness, factual content, or message length.


============================================================
OUTPUT FORMAT
============================================================

Return ONLY valid JSON:

{{
    "experiential": "...",
    "analytical": "...",
    "integrated": "..."
}}

Do not provide explanations.
Do not use Markdown.
Do not include text outside the JSON.
"""

    # --------------------------------------------------------
    # CALL LLAMA THROUGH HUGGING FACE INFERENCE CLIENT
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # EXTRACT RESPONSE TEXT
    # --------------------------------------------------------

    text = response.choices[0].message.content

    if text is None:
        raise ValueError(
            "Model returned no text."
        )

    text = text.strip()

    # --------------------------------------------------------
    # REMOVE ACCIDENTAL MARKDOWN CODE FENCES
    # --------------------------------------------------------

    text = re.sub(
        r"^```(?:json)?\s*",
        "",
        text
    )

    text = re.sub(
        r"\s*```$",
        "",
        text
    )

    # --------------------------------------------------------
    # EXTRACT JSON OBJECT
    # --------------------------------------------------------

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1:
        print("RAW MODEL RESPONSE:")
        print(repr(text))

        raise ValueError(
            "No JSON object found in model response."
        )

    text = text[start:end + 1]

    # --------------------------------------------------------
    # PARSE JSON
    # --------------------------------------------------------

    try:
        transformed = json.loads(text)

    except json.JSONDecodeError as e:

        print("RAW MODEL RESPONSE:")
        print(repr(text))

        raise ValueError(
            f"Model returned invalid JSON: {e}"
        )

    # --------------------------------------------------------
    # VERIFY REQUIRED OUTPUTS
    # --------------------------------------------------------

    required_styles = [
        "experiential",
        "analytical",
        "integrated"
    ]

    if not all(
        style in transformed
        for style in required_styles
    ):
        raise ValueError(
            "Model output is missing one or more required styles."
        )

    # --------------------------------------------------------
    # VERIFY OUTPUTS ARE NON-EMPTY STRINGS
    # --------------------------------------------------------

    for style in required_styles:

        if not isinstance(
            transformed[style],
            str
        ):
            raise ValueError(
                f"{style} output is not a string."
            )

        if not transformed[style].strip():
            raise ValueError(
                f"{style} output is empty."
            )

        transformed[style] = transformed[style].strip()

    return transformed


def validate_input(df):
    required = {
        "scenario_id", "review_id", "source_text_hash",
        "source_text", "human_primary_domain",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input is missing required columns: {sorted(missing)}")
    if len(df) != 150:
        print(f"WARNING: input contains {len(df)} scenarios; historical study input contained 150.")
    if df["scenario_id"].duplicated().any():
        raise ValueError("scenario_id must be unique.")


def main():
    df = pd.read_csv(INPUT_PATH)
    validate_input(df)
    results = []
    failed_scenarios = []

    if FAILED_PATH.exists():
        FAILED_PATH.unlink()

    print("=" * 70)
    print("STARTING CEST COMMUNICATIVE-ORIENTATION GENERATION")
    print("=" * 70)
    print(f"Scenarios to attempt: {len(df)}")
    print("Conditions: Experiential, Analytical, Integrated")
    print(f"Expected final rows: {len(df) * 3}")

    for i, row in df.iterrows():
        scenario_id = row["scenario_id"]
        review_id = row["review_id"]
        source_text_hash = row["source_text_hash"]
        source_text = row["source_text"]
        domain = row["human_primary_domain"]

        print(f"[{i + 1}/{len(df)}] {scenario_id} ({domain})...")

        try:
            transformed = transform_matched_set(source_text)

            if not all(style in transformed for style in REQUIRED_ORIENTATIONS):
                raise ValueError("Missing one or more communicative-orientation outputs.")

            scenario_results = []
            for style in REQUIRED_ORIENTATIONS:
                transformed_text = transformed[style].strip()
                if not transformed_text:
                    raise ValueError(f"Empty output for {style}.")

                scenario_results.append({
                    "scenario_id": scenario_id,
                    "review_id": review_id,
                    "source_text_hash": source_text_hash,
                    "human_primary_domain": domain,
                    "style": style,
                    "source_text": source_text,
                    "transformed_text": transformed_text,
                    "source_word_count": len(source_text.split()),
                    "transformed_word_count": len(transformed_text.split()),
                    "model": MODEL,
                    "temperature": TEMPERATURE,
                    "top_p": TOP_P,
                    "provider_requested": PROVIDER_REQUESTED,
                    "transformation_protocol": TRANSFORMATION_PROTOCOL,
                })

            # Save a scenario only after all three matched outputs succeed.
            results.extend(scenario_results)
            pd.DataFrame(results).to_csv(OUTPUT_PATH, index=False)
            print("   ✓ 3 transformations generated and saved")
            time.sleep(1)

        except Exception as exc:
            print(f"   ✗ ERROR: {exc}")
            failed_scenarios.append({
                "scenario_id": scenario_id,
                "review_id": review_id,
                "human_primary_domain": domain,
                "error": str(exc),
            })
            pd.DataFrame(failed_scenarios).to_csv(FAILED_PATH, index=False)
            print("   Failure recorded; continuing.")

    generated = pd.DataFrame(results)
    print("\n" + "=" * 70)
    print("GENERATION SUMMARY")
    print("=" * 70)
    print(f"Total transformed rows: {len(generated)}")

    if not generated.empty:
        print(f"Unique transformed scenarios: {generated['scenario_id'].nunique()}")
        print("\nRows by communicative orientation:")
        print(generated["style"].value_counts())
        print("\nUnique scenarios by domain:")
        print(generated.groupby("human_primary_domain")["scenario_id"].nunique())

        expected_ids = set(df["scenario_id"])
        generated_ids = set(generated["scenario_id"])
        missing_ids = sorted(expected_ids - generated_ids)
        duplicates = int(generated.duplicated(["scenario_id", "style"]).sum())

        print(f"\nMissing scenario IDs: {missing_ids if missing_ids else 'None'}")
        print(f"Duplicate scenario × orientation rows: {duplicates}")
        print(f"Failed generation attempts: {len(failed_scenarios)}")

        generated.to_csv(OUTPUT_PATH, index=False)

        complete = (
            len(generated) == len(df) * 3
            and generated["scenario_id"].nunique() == len(df)
            and all(
                int((generated["style"] == style).sum()) == len(df)
                for style in REQUIRED_ORIENTATIONS
            )
            and duplicates == 0
            and not missing_ids
            and not failed_scenarios
        )
        print(
            "\n✓ CEST COMMUNICATIVE-ORIENTATION GENERATION COMPLETE"
            if complete
            else "\n⚠ GENERATION IS NOT COMPLETE; inspect failures before downstream use."
        )

    print(f"\nMain generation file: {OUTPUT_PATH}")
    if failed_scenarios:
        print(f"Failure log: {FAILED_PATH}")


if __name__ == "__main__":
    main()
