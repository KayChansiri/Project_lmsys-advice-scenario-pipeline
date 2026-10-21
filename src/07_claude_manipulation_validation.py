"""Supplementary blinded Claude validation of the 135-item manipulation sample.

Claude uses the same four 1-5 dimensions and rating anchors as the human
validation codebook. The intended communicative-orientation condition is not
provided to the evaluator.
"""

from pathlib import Path
import json
import os
import time

import pandas as pd
from anthropic import Anthropic

PROJECT_DIR = Path(__file__).resolve().parents[1]
VALIDATION_DIR = PROJECT_DIR / "data" / "human_validation"
INPUT_PATH = VALIDATION_DIR / "communicative_orientation_validation_master.xlsx"
OUTPUT_PATH = VALIDATION_DIR / "claude_validation_135_decision_fidelity.csv"
MODEL = "claude-sonnet-5-5"

# Operational manipulation-check criteria. These are applied after blinded
# ratings and are not disclosed to the evaluator.
MANIPULATION_CRITERIA = {
    "experiential": {"experiential_min": 4, "analytical_max": 2},
    "analytical": {"analytical_min": 4, "experiential_max": 2},
    "integrated": {"experiential_min": 3, "analytical_min": 3, "max_abs_difference": 1},
}
QUALITY_MINIMUM = 3

api_key = os.environ.get("ANTHROPIC_API_KEY")
if not api_key:
    raise RuntimeError("Set ANTHROPIC_API_KEY in the environment before running this script.")
client = Anthropic(api_key=api_key)

RATING_SCHEMA = {
    "type": "object",
    "properties": {
        "experiential_orientation": {
            "type": "integer",
            "enum": [1, 2, 3, 4, 5]
        },
        "analytical_orientation": {
            "type": "integer",
            "enum": [1, 2, 3, 4, 5]
        },
        "naturalness": {
            "type": "integer",
            "enum": [1, 2, 3, 4, 5]
        },
        "decision_situation_fidelity": {
            "type": "integer",
            "enum": [1, 2, 3, 4, 5]
        }
    },
    "required": [
        "experiential_orientation",
        "analytical_orientation",
        "naturalness",
        "decision_situation_fidelity"
    ],
    "additionalProperties": False
}


# ============================================================
# VALIDATION CODEBOOK
# ============================================================

EVALUATION_INSTRUCTIONS = """
You are independently evaluating a transformed advice-seeking message
for a research study.

You will receive:

1. the original source message; and
2. a transformed version of that message.

You are NOT being told the intended communicative-orientation condition.

Evaluate the transformed message independently on FOUR dimensions.

Use integer scores from 1 to 5.


============================================================
1. EXPERIENTIAL ORIENTATION
============================================================

Rate the extent to which the transformed message communicates the
problem through lived experience and is organized around feelings,
intuitions, impressions, immediate reactions, subjective uncertainty,
personal salience, and the way concerns arise in the user's experience.

A message higher in experiential orientation tends to foreground how
the situation is being experienced. Facts and practical considerations
may still be present, but they are less likely to organize the message
through explicit criteria, alternatives, pros and cons, consequences,
or structured tradeoffs.

Rate EXPERIENTIAL ORIENTATION independently of analytical orientation.

1 = Very low:
    Lived experience, feelings, intuitions, impressions, and subjective
    concerns play little or no role in organizing the message.

2 = Low:
    Some experiential material is present, but it is mostly peripheral
    rather than an organizing feature.

3 = Moderate:
    Experiential material is clearly present and contributes to how the
    problem is communicated, but does not strongly organize the message.

4 = High:
    Lived experience, feelings, intuitions, impressions, or subjective
    concerns clearly organize substantial parts of the message.

5 = Very high:
    The message is strongly organized around the user's lived and
    subjective experience throughout.


============================================================
2. ANALYTICAL ORIENTATION
============================================================

Rate the extent to which the transformed message communicates the
problem through explicit deliberation and is organized around relevant
facts, constraints, priorities, alternatives, consequences,
uncertainties, and tradeoffs.

A message higher in analytical orientation tends to make relationships
among relevant considerations explicit. When competing considerations
are present, it may explicitly contrast or weigh them.

Feelings and subjective experiences may still be present, but they do
not necessarily organize the message.

Rate ANALYTICAL ORIENTATION independently of experiential orientation.

1 = Very low:
    Explicit deliberation or structured consideration of relevant
    factors, alternatives, consequences, or tradeoffs is largely absent.

2 = Low:
    Some analytical consideration is present, but it is mostly
    peripheral rather than an organizing feature.

3 = Moderate:
    Explicit deliberation is clearly present and contributes to how the
    problem is communicated, but does not strongly organize the message.

4 = High:
    Explicit consideration of relevant facts, constraints, priorities,
    alternatives, consequences, uncertainty, or tradeoffs clearly
    organizes substantial parts of the message.

5 = Very high:
    The message is strongly organized around explicit deliberation and
    relationships among relevant considerations throughout.


============================================================
3. NATURALNESS / PLAUSIBILITY
============================================================

Rate the extent to which the transformed message could plausibly have
been written by a real person seeking advice in an online conversational
setting.

Do NOT penalize a message simply because it is more emotionally
expressive, more structured, or organized differently from the source.

The question is whether the message remains plausible as human
advice-seeking communication.

1 = Very low:
    The message is clearly unnatural, artificial, or implausible as
    real advice-seeking communication.

2 = Low:
    The message contains substantial unnatural or constructed features
    that make it difficult to imagine as a real advice-seeking message.

3 = Acceptable:
    The message is generally plausible as real advice-seeking
    communication, although some wording or organization may feel
    somewhat constructed.

4 = High:
    The message reads naturally and plausibly as something a person
    could realistically write when seeking advice.

5 = Very high:
    The message reads highly naturally and convincingly as authentic
    advice-seeking communication.


============================================================
4. DECISION-SITUATION FIDELITY
============================================================

Rate the extent to which the transformed message represents the SAME
underlying advice-seeking or decision-making situation as the source.

Focus on whether the transformation preserves the underlying problem,
relevant considerations and constraints, uncertainty or stance, and
advice-seeking goal sufficiently for the source and transformed message
to be treated as the same substantive decision situation.

Exact wording, sentence structure, organization, and one-to-one
preservation of every minor detail are NOT required.

Reorganization, condensation, or changes in emphasis are acceptable
when they do not materially alter the underlying decision problem.

1 = Very low:
    The transformed message represents a substantially different
    decision situation, problem, stance, or advice-seeking goal.

2 = Low:
    Important changes substantially alter how the underlying decision
    or advice-seeking problem should be understood.

3 = Acceptable:
    Some details, emphasis, or organization differ, but the same
    underlying decision situation and advice-seeking problem are
    preserved sufficiently to be treated as the same scenario.

4 = High:
    The same underlying decision situation, relevant considerations,
    uncertainty or stance, and advice-seeking goal are clearly
    preserved, with only minor non-substantive differences.

5 = Very high:
    The underlying decision situation, relevant considerations,
    uncertainty or stance, and advice-seeking goal are preserved very
    closely despite the change in communicative orientation.


============================================================
IMPORTANT
============================================================

- Evaluate each dimension independently.

- Do not infer what score is expected.

- Do not attempt to identify the intended experimental condition.

- Do not reward polished or formal writing simply because it appears
  more sophisticated.

- Experiential orientation does NOT imply irrationality, lower
  intelligence, greater distress, impulsivity, or confusion.

- Analytical orientation does NOT imply greater intelligence,
  sophistication, certainty, decisiveness, or correctness.

- Experiential and analytical orientation are NOT opposite ends of a
  single scale.

- A message can be high in BOTH experiential and analytical
  orientation.

- A message can also be low in both.

- For naturalness, judge whether the transformed message is plausible
  as real advice-seeking communication rather than whether it resembles
  the source's writing style.

- For decision-situation fidelity, focus on preservation of the
  substantive decision or advice-seeking situation rather than exact
  wording or one-to-one preservation of every detail.
"""


def evaluate_stimulus(source_text: str, transformed_text: str) -> dict:
    prompt = f"""
{EVALUATION_INSTRUCTIONS}

============================================================
SOURCE MESSAGE
============================================================

{source_text}

============================================================
TRANSFORMED MESSAGE
============================================================

{transformed_text}

Provide the four requested ratings.
"""

    response = client.messages.create(
        model=MODEL,
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}],
        output_config={"format": {"type": "json_schema", "schema": RATING_SCHEMA}},
    )
    text_blocks = [block.text for block in response.content if getattr(block, "type", None) == "text"]
    if not text_blocks:
        raise ValueError("Claude returned no text output.")
    ratings = json.loads("".join(text_blocks).strip())
    for variable in RATING_SCHEMA["required"]:
        if ratings.get(variable) not in {1, 2, 3, 4, 5}:
            raise ValueError(f"Invalid rating for {variable}: {ratings.get(variable)!r}")
    return ratings


def matches_expected_orientation(row) -> bool:
    style = str(row["style"]).lower()
    e = int(row["experiential_orientation"])
    a = int(row["analytical_orientation"])
    if style == "experiential":
        return e >= 4 and a <= 2
    if style == "analytical":
        return a >= 4 and e <= 2
    if style == "integrated":
        return e >= 3 and a >= 3 and abs(e - a) <= 1
    raise ValueError(f"Unknown style: {style}")


def main() -> None:
    master = pd.read_excel(INPUT_PATH, sheet_name="master_key")
    assert len(master) == 135, f"Expected 135 validation stimuli, found {len(master)}."

    completed = []
    if OUTPUT_PATH.exists():
        prior = pd.read_csv(OUTPUT_PATH)
        completed = prior.to_dict("records")
    done_ids = {str(row["validation_id"]) for row in completed}

    rows = list(completed)
    for _, row in master.iterrows():
        validation_id = str(row["validation_id"])
        if validation_id in done_ids:
            continue
        ratings = evaluate_stimulus(str(row["source_text"]), str(row["transformed_text"]))
        rows.append({
            "validation_id": validation_id,
            "scenario_id": row.get("scenario_id"),
            "human_primary_domain": row.get("human_primary_domain"),
            "style": row.get("style"),
            **ratings,
            "evaluator_model": MODEL,
        })
        pd.DataFrame(rows).to_csv(OUTPUT_PATH, index=False)
        print(f"[{len(rows)}/135] {validation_id}: {ratings}")
        time.sleep(0.1)

    final = pd.DataFrame(rows)
    assert final["validation_id"].nunique() == 135

    final["expected_orientation_pattern"] = final.apply(matches_expected_orientation, axis=1)
    final["acceptable_naturalness"] = final["naturalness"] >= QUALITY_MINIMUM
    final["acceptable_decision_fidelity"] = (
        final["decision_situation_fidelity"] >= QUALITY_MINIMUM
    )
    final.to_csv(OUTPUT_PATH, index=False)

    print("Claude independently rated all 135 stimuli.")
    print("Orientation-pattern pass rates (%):")
    print((final.groupby("style")["expected_orientation_pattern"].mean() * 100).round(1))
    print("Overall orientation-pattern rate:", round(final["expected_orientation_pattern"].mean() * 100, 1))
    print("Overall acceptable-naturalness rate:", round(final["acceptable_naturalness"].mean() * 100, 1))
    print("Overall acceptable decision-situation fidelity rate:", round(final["acceptable_decision_fidelity"].mean() * 100, 1))
    print("Saved to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
