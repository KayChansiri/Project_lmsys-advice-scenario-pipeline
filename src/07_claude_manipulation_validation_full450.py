"""Blinded Claude validation of the communicative-orientation manipulation.

This script preserves the study's two-stage validation logic:

Stage 1: Claude evaluates the same stratified 135-item subset independently
evaluated by two human reviewers. This subset supports human-human reliability
and human-Claude alignment.

Stage 2: After preserving Stage 1, Claude evaluates the remaining 315
transformed stimuli using the exact same blinded four-dimension evaluator.

Final: The 135 + 315 ratings are combined into one 450-stimulus file for
supplementary model-based manipulation diagnostics.

The intended communicative-orientation condition, decision domain, human
ratings, validation-subset membership, and manipulation-check thresholds are
not provided to Claude.

API credentials are read only from the ANTHROPIC_API_KEY environment variable.
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
    """Apply operational manipulation-check criteria after blinded rating."""
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


# ============================================================
# TWO-STAGE VALIDATION PIPELINE
# ============================================================
#
# STAGE 1 — PRIMARY VALIDATION SUBSET (n = 135)
#   * selected upstream by 06_prepare_manipulation_validation.py
#   * stratified by communicative orientation x decision domain
#   * independently evaluated by two human reviewers
#   * independently evaluated by Claude with the blinded evaluator above
#   * supports human-human reliability and human-Claude alignment
#
# STAGE 2 — SECONDARY FULL-DATA SCREEN (n = 315)
#   * consists only of stimuli not included in Stage 1
#   * evaluated with the EXACT SAME Claude evaluator
#
# FINAL — 135 + 315 = 450
#
# Claude receives ONLY source_text and transformed_text in both stages.
# Condition, domain, scenario_id, human ratings, validation-subset status,
# and manipulation-check thresholds are restored/applied only after rating.
# ============================================================

FULL_STIMULI_PATH = (
    PROJECT_DIR / "data" / "processed" / "communicative_style_generation.csv"
)

# Historical and renamed Stage-1 master filenames are both supported.
HISTORICAL_MASTER_PATH = (
    VALIDATION_DIR / "communicative_style_validation_master.xlsx"
)
CURRENT_MASTER_PATH = (
    VALIDATION_DIR / "communicative_orientation_validation_master.xlsx"
)

STAGE1_OUTPUT_PATH = (
    VALIDATION_DIR / "claude_validation_135_decision_fidelity.csv"
)
STAGE2_OUTPUT_PATH = (
    VALIDATION_DIR / "claude_validation_remaining_315.csv"
)
ALL_OUTPUT_PATH = (
    VALIDATION_DIR / "claude_validation_all_450.csv"
)


def resolve_master_path() -> Path:
    """Prefer the historical master so an existing 135 run is preserved."""
    if HISTORICAL_MASTER_PATH.exists():
        return HISTORICAL_MASTER_PATH
    if CURRENT_MASTER_PATH.exists():
        return CURRENT_MASTER_PATH
    raise FileNotFoundError(
        "Could not find the 135-item validation master. Expected either:\n"
        f"  {HISTORICAL_MASTER_PATH}\n"
        f"  {CURRENT_MASTER_PATH}"
    )


def add_posthoc_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Apply interpretation criteria only after blinded ratings exist."""
    out = df.copy()
    out["expected_orientation_pattern"] = out.apply(
        matches_expected_orientation, axis=1
    )
    out["acceptable_naturalness"] = (
        out["naturalness"].astype(int) >= QUALITY_MINIMUM
    )
    out["acceptable_decision_fidelity"] = (
        out["decision_situation_fidelity"].astype(int) >= QUALITY_MINIMUM
    )
    return out


def run_stage1(master: pd.DataFrame) -> pd.DataFrame:
    """Preserve or reproduce the original 135-item Claude validation."""
    assert len(master) == 135, (
        f"Expected 135 Stage-1 stimuli, found {len(master)}."
    )
    assert master["validation_id"].astype(str).nunique() == 135

    if STAGE1_OUTPUT_PATH.exists():
        existing = pd.read_csv(STAGE1_OUTPUT_PATH)
        done_ids = set(existing["validation_id"].astype(str))
        rows = existing.to_dict("records")

        if len(done_ids) == 135:
            print("STAGE 1: completed historical 135-item file found.")
            print("No Stage-1 API calls will be repeated.")
            return existing

        print(f"STAGE 1: resuming checkpoint ({len(done_ids)}/135).")
    else:
        rows = []
        done_ids = set()
        print("STAGE 1: starting 135-item validation.")

    for _, row in master.iterrows():
        validation_id = str(row["validation_id"])
        if validation_id in done_ids:
            continue

        ratings = evaluate_stimulus(
            source_text=str(row["source_text"]),
            transformed_text=str(row["transformed_text"]),
        )

        rows.append(
            {
                "validation_id": validation_id,
                "experiential_orientation": ratings["experiential_orientation"],
                "analytical_orientation": ratings["analytical_orientation"],
                "naturalness": ratings["naturalness"],
                "decision_situation_fidelity": ratings[
                    "decision_situation_fidelity"
                ],
                "evaluator_model": MODEL,
                "sampling_parameters": "model_default",
            }
        )

        pd.DataFrame(rows).to_csv(STAGE1_OUTPUT_PATH, index=False)
        done_ids.add(validation_id)
        print(f"  [{len(done_ids)}/135] {validation_id}: {ratings}")
        time.sleep(0.5)

    final = pd.read_csv(STAGE1_OUTPUT_PATH)
    assert len(final) == 135
    assert final["validation_id"].astype(str).nunique() == 135
    return final


def prepare_stage2(
    full_stimuli: pd.DataFrame,
    master: pd.DataFrame,
) -> pd.DataFrame:
    """Identify exactly the 315 stimuli not included in Stage 1."""
    required_full = {
        "scenario_id", "style", "source_text", "transformed_text"
    }
    missing = required_full - set(full_stimuli.columns)
    if missing:
        raise ValueError(
            f"Full 450-stimulus file is missing: {sorted(missing)}"
        )

    required_master = {
        "validation_id", "scenario_id", "source_text", "transformed_text"
    }
    missing = required_master - set(master.columns)
    if missing:
        raise ValueError(
            f"Stage-1 master is missing: {sorted(missing)}"
        )

    assert len(full_stimuli) == 450, (
        f"Expected 450 transformed stimuli, found {len(full_stimuli)}."
    )

    full = full_stimuli.copy()
    stage1 = master.copy()

    # validation_id exists only in the 135 master, so match the actual
    # stimulus using scenario_id + transformed_text.
    full["stimulus_key"] = (
        full["scenario_id"].astype(str)
        + "|||"
        + full["transformed_text"].astype(str)
    )
    stage1["stimulus_key"] = (
        stage1["scenario_id"].astype(str)
        + "|||"
        + stage1["transformed_text"].astype(str)
    )

    assert full["stimulus_key"].nunique() == 450
    assert stage1["stimulus_key"].nunique() == 135

    stage1_keys = set(stage1["stimulus_key"])
    matched = full["stimulus_key"].isin(stage1_keys)

    assert int(matched.sum()) == 135, (
        f"Matched {int(matched.sum())}/135 Stage-1 stimuli."
    )

    remaining = full.loc[~matched].copy().reset_index(drop=True)
    assert len(remaining) == 315

    remaining["screen_id"] = [
        f"R{i:03d}" for i in range(1, 316)
    ]
    return remaining


def run_stage2(remaining: pd.DataFrame) -> pd.DataFrame:
    """Rate the remaining 315 with the exact same blinded evaluator."""
    if STAGE2_OUTPUT_PATH.exists():
        existing = pd.read_csv(STAGE2_OUTPUT_PATH)
        rows = existing.to_dict("records")
        done_keys = set(existing["stimulus_key"].astype(str))
        print(f"STAGE 2: resuming checkpoint ({len(done_keys)}/315).")
    else:
        rows = []
        done_keys = set()
        print("STAGE 2: starting remaining 315.")

    for _, row in remaining.iterrows():
        stimulus_key = str(row["stimulus_key"])
        if stimulus_key in done_keys:
            continue

        # EXACT SAME function/prompt/schema as Stage 1.
        ratings = evaluate_stimulus(
            source_text=str(row["source_text"]),
            transformed_text=str(row["transformed_text"]),
        )

        rows.append(
            {
                "screen_id": str(row["screen_id"]),
                "stimulus_key": stimulus_key,
                "experiential_orientation": ratings["experiential_orientation"],
                "analytical_orientation": ratings["analytical_orientation"],
                "naturalness": ratings["naturalness"],
                "decision_situation_fidelity": ratings[
                    "decision_situation_fidelity"
                ],
                "evaluator_model": MODEL,
                "sampling_parameters": "model_default",
            }
        )

        pd.DataFrame(rows).to_csv(STAGE2_OUTPUT_PATH, index=False)
        done_keys.add(stimulus_key)
        print(f"  [{len(done_keys)}/315] {row['screen_id']}: {ratings}")
        time.sleep(0.5)

    final = pd.read_csv(STAGE2_OUTPUT_PATH)
    assert len(final) == 315
    assert final["stimulus_key"].astype(str).nunique() == 315
    return final


def combine_all_450(
    full_stimuli: pd.DataFrame,
    master: pd.DataFrame,
    stage1_ratings: pd.DataFrame,
    stage2_ratings: pd.DataFrame,
) -> pd.DataFrame:
    """Combine Stage 1 and Stage 2, then restore experimental metadata."""
    full = full_stimuli.copy()
    full["stimulus_key"] = (
        full["scenario_id"].astype(str)
        + "|||"
        + full["transformed_text"].astype(str)
    )

    stage1_map = master[
        ["validation_id", "scenario_id", "transformed_text"]
    ].copy()
    stage1_map["validation_id"] = stage1_map["validation_id"].astype(str)
    stage1_map["stimulus_key"] = (
        stage1_map["scenario_id"].astype(str)
        + "|||"
        + stage1_map["transformed_text"].astype(str)
    )

    s1 = stage1_ratings.copy()
    s1["validation_id"] = s1["validation_id"].astype(str)

    keep = [
        "validation_id",
        "experiential_orientation",
        "analytical_orientation",
        "naturalness",
        "decision_situation_fidelity",
        "evaluator_model",
    ]
    if "sampling_parameters" in s1.columns:
        keep.append("sampling_parameters")

    s1 = s1[keep].merge(
        stage1_map[["validation_id", "stimulus_key"]],
        on="validation_id",
        how="left",
        validate="one_to_one",
    )
    if "sampling_parameters" not in s1.columns:
        s1["sampling_parameters"] = "model_default"

    s1["evaluation_stage"] = "stage1_human_validation_subset"

    s2 = stage2_ratings[
        [
            "stimulus_key",
            "experiential_orientation",
            "analytical_orientation",
            "naturalness",
            "decision_situation_fidelity",
            "evaluator_model",
            "sampling_parameters",
        ]
    ].copy()
    s2["evaluation_stage"] = "stage2_remaining_secondary_screen"

    rating_columns = [
        "stimulus_key",
        "experiential_orientation",
        "analytical_orientation",
        "naturalness",
        "decision_situation_fidelity",
        "evaluator_model",
        "sampling_parameters",
        "evaluation_stage",
    ]

    combined = pd.concat(
        [s1[rating_columns], s2[rating_columns]],
        ignore_index=True,
    )

    assert len(combined) == 450
    assert combined["stimulus_key"].nunique() == 450

    final = full.merge(
        combined,
        on="stimulus_key",
        how="left",
        validate="one_to_one",
    )
    assert len(final) == 450
    assert final[
        [
            "experiential_orientation",
            "analytical_orientation",
            "naturalness",
            "decision_situation_fidelity",
        ]
    ].notna().all().all()

    # Thresholds are applied only now, after blinded evaluation.
    final = add_posthoc_flags(final)
    final.to_csv(ALL_OUTPUT_PATH, index=False)
    return final


def print_summary(final: pd.DataFrame) -> None:
    print("\n" + "=" * 72)
    print("CLAUDE MANIPULATION-VALIDATION PIPELINE COMPLETE")
    print("=" * 72)

    print("\nEvaluation-stage counts:")
    print(final["evaluation_stage"].value_counts())

    print("\nMean ratings by generated condition:")
    print(
        final.groupby("style")[
            [
                "experiential_orientation",
                "analytical_orientation",
                "naturalness",
                "decision_situation_fidelity",
            ]
        ].mean().round(2)
    )

    print("\nOperational orientation-pattern pass rates (%):")
    print(
        (
            final.groupby("style")["expected_orientation_pattern"].mean()
            * 100
        ).round(1)
    )

    print(
        "\nOverall orientation-pattern pass rate:",
        round(final["expected_orientation_pattern"].mean() * 100, 1),
    )
    print(
        "Overall naturalness >= 3:",
        round(final["acceptable_naturalness"].mean() * 100, 1),
    )
    print(
        "Overall decision-situation fidelity >= 3:",
        round(final["acceptable_decision_fidelity"].mean() * 100, 1),
    )

    print("\nSaved:")
    print("  Stage 1:", STAGE1_OUTPUT_PATH)
    print("  Stage 2:", STAGE2_OUTPUT_PATH)
    print("  All 450:", ALL_OUTPUT_PATH)


def main() -> None:
    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)

    master_path = resolve_master_path()
    master = pd.read_excel(master_path, sheet_name="master_key")
    full_stimuli = pd.read_csv(FULL_STIMULI_PATH)

    print("=" * 72)
    print("BLINDED CLAUDE MANIPULATION VALIDATION")
    print("=" * 72)
    print("Model:", MODEL)
    print("Stage-1 master:", master_path)
    print("Full transformed stimuli:", FULL_STIMULI_PATH)

    # STAGE 1: preserve/reproduce the primary 135-item validation.
    stage1_ratings = run_stage1(master)

    # STAGE 2: identify and evaluate only the remaining 315.
    remaining = prepare_stage2(full_stimuli, master)
    print("\nStage-1 stimuli preserved: 135")
    print("Stage-2 stimuli identified:", len(remaining))
    stage2_ratings = run_stage2(remaining)

    # FINAL: combine 135 + 315 = 450.
    final = combine_all_450(
        full_stimuli,
        master,
        stage1_ratings,
        stage2_ratings,
    )
    print_summary(final)


if __name__ == "__main__":
    main()
