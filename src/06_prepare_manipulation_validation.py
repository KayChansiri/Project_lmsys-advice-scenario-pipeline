"""Prepare the blinded 30% manipulation-validation sample.

Creates a stratified sample of 135 transformed stimuli (15 per communicative
orientation x decision-domain cell) and two independently randomized reviewer
workbooks. Human reviewers use the same four 1-5 dimensions as the supplementary
Claude validation: experiential orientation, analytical orientation,
naturalness/plausibility, and decision-situation fidelity.
"""

from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_DIR / "data" / "processed" / "communicative_style_generation.csv"
VALIDATION_DIR = PROJECT_DIR / "data" / "human_validation"
VALIDATION_SEED = 20261015

MASTER_PATH = VALIDATION_DIR / "communicative_orientation_validation_master.xlsx"
REVIEWER1_PATH = VALIDATION_DIR / "communicative_orientation_validation_reviewer1.xlsx"
REVIEWER2_PATH = VALIDATION_DIR / "communicative_orientation_validation_reviewer2.xlsx"

# Operational manipulation-check criteria are applied only after blinded ratings
# are collected. They are intentionally not shown in the reviewer workbooks.
MANIPULATION_CRITERIA = {
    "experiential": {"experiential_min": 4, "analytical_max": 2},
    "analytical": {"analytical_min": 4, "experiential_max": 2},
    "integrated": {"experiential_min": 3, "analytical_min": 3, "max_abs_difference": 1},
}
QUALITY_MINIMUM = 3

RATING_COLUMNS = [
    "experiential_orientation",
    "analytical_orientation",
    "naturalness",
    "decision_situation_fidelity",
]

CODEBOOK = [
    ("Experiential orientation", "1-5", "Rate the extent to which the transformed message communicates the problem through lived experience and is organized around feelings, intuitions, impressions, immediate reactions, subjective uncertainty, personal salience, and the way concerns arise in the user's experience. Rate independently of analytical orientation."),
    ("Analytical orientation", "1-5", "Rate the extent to which the transformed message communicates the problem through explicit deliberation and is organized around relevant facts, constraints, priorities, alternatives, consequences, uncertainties, and tradeoffs. Rate independently of experiential orientation."),
    ("Naturalness / plausibility", "1-5", "Rate the extent to which the transformed message could plausibly have been written by a real person seeking advice in an online conversational setting. Do not penalize emotional expressiveness, structure, or organization simply because it differs from the source."),
    ("Decision-situation fidelity", "1-5", "Rate the extent to which the transformed message represents the same underlying advice-seeking or decision-making situation as the source, preserving the underlying problem, relevant considerations and constraints, uncertainty or stance, and advice-seeking goal. Exact wording and one-to-one preservation of every minor detail are not required."),
]

ANCHORS = {
    "experiential_orientation": [
        "1 = Very low: Lived experience, feelings, intuitions, impressions, and subjective concerns play little or no role in organizing the message.",
        "2 = Low: Some experiential material is present, but it is mostly peripheral rather than an organizing feature.",
        "3 = Moderate: Experiential material is clearly present and contributes to how the problem is communicated, but does not strongly organize the message.",
        "4 = High: Lived experience, feelings, intuitions, impressions, or subjective concerns clearly organize substantial parts of the message.",
        "5 = Very high: The message is strongly organized around the user's lived and subjective experience throughout.",
    ],
    "analytical_orientation": [
        "1 = Very low: Explicit deliberation or structured consideration of relevant factors, alternatives, consequences, or tradeoffs is largely absent.",
        "2 = Low: Some analytical consideration is present, but it is mostly peripheral rather than an organizing feature.",
        "3 = Moderate: Explicit deliberation is clearly present and contributes to how the problem is communicated, but does not strongly organize the message.",
        "4 = High: Explicit consideration of relevant facts, constraints, priorities, alternatives, consequences, uncertainty, or tradeoffs clearly organizes substantial parts of the message.",
        "5 = Very high: The message is strongly organized around explicit deliberation and relationships among relevant considerations throughout.",
    ],
    "naturalness": [
        "1 = Very low: The message is clearly unnatural, artificial, or implausible as real advice-seeking communication.",
        "2 = Low: The message contains substantial unnatural or constructed features that make it difficult to imagine as a real advice-seeking message.",
        "3 = Acceptable: The message is generally plausible as real advice-seeking communication, although some wording or organization may feel somewhat constructed.",
        "4 = High: The message reads naturally and plausibly as something a person could realistically write when seeking advice.",
        "5 = Very high: The message reads highly naturally and convincingly as authentic advice-seeking communication.",
    ],
    "decision_situation_fidelity": [
        "1 = Very low: The transformed message represents a substantially different decision situation, problem, stance, or advice-seeking goal.",
        "2 = Low: Important changes substantially alter how the underlying decision or advice-seeking problem should be understood.",
        "3 = Acceptable: Some details, emphasis, or organization differ, but the same underlying decision situation and advice-seeking problem are preserved sufficiently to be treated as the same scenario.",
        "4 = High: The same underlying decision situation, relevant considerations, uncertainty or stance, and advice-seeking goal are clearly preserved, with only minor non-substantive differences.",
        "5 = Very high: The underlying decision situation, relevant considerations, uncertainty or stance, and advice-seeking goal are preserved very closely despite the change in communicative orientation.",
    ],
}

IMPORTANT = [
    "Evaluate each dimension independently.",
    "Do not infer what score is expected or attempt to identify the intended experimental condition.",
    "Do not reward polished or formal writing simply because it appears more sophisticated.",
    "Experiential orientation does not imply irrationality, lower intelligence, greater distress, impulsivity, or confusion.",
    "Analytical orientation does not imply greater intelligence, sophistication, certainty, decisiveness, or correctness.",
    "Experiential and analytical orientation are not opposite ends of a single scale; a message can be high in both or low in both.",
    "For naturalness, judge plausibility as real advice-seeking communication rather than similarity to the source's writing style.",
    "For decision-situation fidelity, focus on preservation of the substantive decision or advice-seeking situation rather than exact wording or one-to-one preservation of every detail.",
]


def build_instructions() -> pd.DataFrame:
    rows = []
    for construct, scale, instruction in CODEBOOK:
        rows.append({"Section": construct, "Scale": scale, "Instruction": instruction})
    for variable, anchors in ANCHORS.items():
        label = variable.replace("_", " ").title()
        for anchor in anchors:
            rows.append({"Section": f"{label} anchors", "Scale": "", "Instruction": anchor})
    for item in IMPORTANT:
        rows.append({"Section": "Important", "Scale": "", "Instruction": item})
    return pd.DataFrame(rows)


def main() -> None:
    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    stimuli = pd.read_csv(INPUT_PATH)

    assert len(stimuli) == 450, f"Expected 450 stimuli, found {len(stimuli)}."
    assert stimuli["scenario_id"].nunique() == 150, "Expected 150 unique scenarios."
    assert stimuli.duplicated(subset=["scenario_id", "style"]).sum() == 0, "Duplicate scenario x orientation rows detected."

    validation_sample = (
        stimuli.groupby(["style", "human_primary_domain"], group_keys=False)
        .sample(n=15, random_state=VALIDATION_SEED)
        .reset_index(drop=True)
    )
    assert len(validation_sample) == 135

    validation_sample = validation_sample.sample(frac=1, random_state=VALIDATION_SEED).reset_index(drop=True)
    validation_sample["validation_id"] = [f"V{i:03d}" for i in range(1, 136)]

    counts = pd.crosstab(validation_sample["style"], validation_sample["human_primary_domain"])
    assert (counts == 15).all().all(), "Expected 15 stimuli in every orientation x domain cell."

    master_columns = [
        "validation_id", "scenario_id", "human_primary_domain", "style",
        "source_text", "transformed_text", "source_word_count",
        "transformed_word_count", "transformation_protocol",
    ]
    master_columns = [c for c in master_columns if c in validation_sample.columns]
    master = validation_sample[master_columns].copy()

    reviewer = validation_sample[["validation_id", "source_text", "transformed_text"]].copy()
    for column in RATING_COLUMNS:
        reviewer[column] = ""
    reviewer["reviewer_notes"] = ""

    reviewer1 = reviewer.sample(frac=1, random_state=VALIDATION_SEED + 1).reset_index(drop=True)
    reviewer2 = reviewer.sample(frac=1, random_state=VALIDATION_SEED + 2).reset_index(drop=True)
    instructions = build_instructions()

    for path, ratings in [(MASTER_PATH, master), (REVIEWER1_PATH, reviewer1), (REVIEWER2_PATH, reviewer2)]:
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            ratings.to_excel(writer, sheet_name="master_key" if path == MASTER_PATH else "ratings", index=False)
            instructions.to_excel(writer, sheet_name="instructions", index=False)

    print("Validation stimuli:", len(validation_sample))
    print(counts)
    print("Saved:", MASTER_PATH)
    print("Saved:", REVIEWER1_PATH)
    print("Saved:", REVIEWER2_PATH)


if __name__ == "__main__":
    main()
