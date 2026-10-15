"""
07_sample_human_validation.py

Create the stratified human-validation sample for the communicative-
style manipulation check.

A 30% sample of the 450 transformed stimuli is selected (n = 135).
Sampling is stratified by communicative-style condition and decision
domain:

    3 communicative styles
    × 3 decision domains
    × 15 stimuli per stratum
    = 135 stimuli

This produces:
    - 45 System 1-oriented stimuli
    - 45 System 2-oriented stimuli
    - 45 mixed stimuli

Each of the two reviewers receives the SAME 135 stimuli in a
different randomized order. The intended communicative-style
condition is omitted from reviewer files.

The master key and reviewer files contain study data and should not
be committed to the public repository.
"""

from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "communicative_style_stimuli_final.csv"
)

OUTPUT_DIR = (
    PROJECT_DIR
    / "data"
    / "human_validation"
)

MASTER_PATH = (
    OUTPUT_DIR
    / "communicative_style_validation_master.xlsx"
)

REVIEWER1_PATH = (
    OUTPUT_DIR
    / "communicative_style_validation_reviewer1.xlsx"
)

REVIEWER2_PATH = (
    OUTPUT_DIR
    / "communicative_style_validation_reviewer2.xlsx"
)

VALIDATION_SEED = 20261015

REQUIRED_STYLES = [
    "system1",
    "system2",
    "mixed"
]

REQUIRED_DOMAINS = [
    "career",
    "emotional_wellbeing",
    "relationship"
]


# ---------------------------------------------------------------------
# Load final stimuli
# ---------------------------------------------------------------------

def load_stimuli():

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Final stimulus file not found: {INPUT_PATH}"
        )

    stimuli = pd.read_csv(INPUT_PATH)

    required_columns = {
        "scenario_id",
        "human_primary_domain",
        "style",
        "source_text",
        "transformed_text"
    }

    missing_columns = required_columns - set(stimuli.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    # Integrity checks
    if len(stimuli) != 450:
        raise ValueError(
            f"Expected 450 stimuli, found {len(stimuli)}."
        )

    if stimuli["scenario_id"].nunique() != 150:
        raise ValueError(
            "Expected 150 unique underlying scenarios."
        )

    duplicate_count = stimuli.duplicated(
        subset=["scenario_id", "style"]
    ).sum()

    if duplicate_count != 0:
        raise ValueError(
            f"Found {duplicate_count} duplicate "
            "scenario × style rows."
        )

    return stimuli


# ---------------------------------------------------------------------
# Draw stratified validation sample
# ---------------------------------------------------------------------

def sample_validation_stimuli(stimuli):

    # 15 observations from every style × domain combination:
    #
    # 3 styles × 3 domains × 15 = 135 stimuli
    validation_sample = (
        stimuli
        .groupby(
            ["style", "human_primary_domain"],
            group_keys=False
        )
        .sample(
            n=15,
            random_state=VALIDATION_SEED
        )
        .reset_index(drop=True)
    )

    if len(validation_sample) != 135:
        raise ValueError(
            f"Expected 135 validation stimuli, "
            f"found {len(validation_sample)}."
        )

    # Randomize before assigning validation IDs so that the IDs
    # themselves do not convey experimental condition.
    validation_sample = (
        validation_sample
        .sample(
            frac=1,
            random_state=VALIDATION_SEED
        )
        .reset_index(drop=True)
    )

    validation_sample["validation_id"] = [
        f"V{i:03d}"
        for i in range(1, len(validation_sample) + 1)
    ]

    return validation_sample


# ---------------------------------------------------------------------
# Verify stratification
# ---------------------------------------------------------------------

def verify_sample(validation_sample):

    style_counts = (
        validation_sample["style"]
        .value_counts()
        .reindex(REQUIRED_STYLES, fill_value=0)
    )

    for style in REQUIRED_STYLES:
        if style_counts[style] != 45:
            raise ValueError(
                f"{style} has {style_counts[style]} stimuli; "
                "expected 45."
            )

    cross_tab = pd.crosstab(
        validation_sample["style"],
        validation_sample["human_primary_domain"]
    )

    for style in REQUIRED_STYLES:
        for domain in REQUIRED_DOMAINS:

            if cross_tab.loc[style, domain] != 15:
                raise ValueError(
                    f"{style} × {domain} does not contain "
                    "15 stimuli."
                )

    print("\nValidation sample by style:")
    print(style_counts)

    print("\nValidation sample by style × domain:")
    print(cross_tab)


# ---------------------------------------------------------------------
# Create blinded reviewer template
# ---------------------------------------------------------------------

def create_reviewer_template(validation_sample):

    # IMPORTANT:
    # style, domain, and scenario_id are deliberately excluded
    # from the reviewer-facing file.

    reviewer = validation_sample[
        [
            "validation_id",
            "source_text",
            "transformed_text"
        ]
    ].copy()

    reviewer["system1_orientation"] = ""
    reviewer["system2_orientation"] = ""
    reviewer["naturalness"] = ""
    reviewer["semantic_fidelity"] = ""
    reviewer["information_preservation"] = ""
    reviewer["reviewer_notes"] = ""

    return reviewer


# ---------------------------------------------------------------------
# Reviewer instructions
# ---------------------------------------------------------------------

def create_instructions():

    return pd.DataFrame({
        "Criterion": [
            "System 1 orientation",
            "System 2 orientation",
            "Naturalness",
            "Semantic fidelity",
            "Information preservation",
            "Rating scale",
            "Blinding"
        ],

        "Instruction": [
            (
                "Rate the extent to which the transformed message "
                "reflects intuitive, affective, impression-based, "
                "immediate, or relatively spontaneous communication."
            ),
            (
                "Rate the extent to which the transformed message "
                "reflects deliberate, analytical, structured, or "
                "explicitly reasoned communication."
            ),
            (
                "Rate how naturally the transformed message reads "
                "as an online advice-seeking message."
            ),
            (
                "Rate the extent to which the transformed message "
                "preserves the substantive meaning and underlying "
                "decision or support need of the source."
            ),
            (
                "Rate the extent to which substantive information "
                "from the source is preserved without unsupported "
                "addition, omission, or material alteration."
            ),
            (
                "Rate each criterion from 1 to 5 using the detailed "
                "anchors provided in the validation codebook."
            ),
            (
                "The intended communicative-style condition is "
                "hidden. Evaluate each stimulus independently "
                "without attempting to recover its assigned condition."
            )
        ]
    })


# ---------------------------------------------------------------------
# Save files
# ---------------------------------------------------------------------

def save_validation_files(validation_sample):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    instructions = create_instructions()

    # --------------------------------------------------------------
    # Master key
    # --------------------------------------------------------------

    master_columns = [
        "validation_id",
        "scenario_id",
        "human_primary_domain",
        "style",
        "source_text",
        "transformed_text"
    ]

    optional_columns = [
        "source_word_count",
        "transformed_word_count",
        "transformation_protocol"
    ]

    for column in optional_columns:
        if column in validation_sample.columns:
            master_columns.append(column)

    master = validation_sample[
        master_columns
    ].copy()

    # --------------------------------------------------------------
    # Reviewer templates
    # --------------------------------------------------------------

    reviewer_template = create_reviewer_template(
        validation_sample
    )

    # Same 135 stimuli, independently randomized order.
    reviewer1 = (
        reviewer_template
        .sample(
            frac=1,
            random_state=VALIDATION_SEED + 1
        )
        .reset_index(drop=True)
    )

    reviewer2 = (
        reviewer_template
        .sample(
            frac=1,
            random_state=VALIDATION_SEED + 2
        )
        .reset_index(drop=True)
    )

    # --------------------------------------------------------------
    # Save master
    # --------------------------------------------------------------

    with pd.ExcelWriter(
        MASTER_PATH,
        engine="openpyxl"
    ) as writer:

        master.to_excel(
            writer,
            sheet_name="master_key",
            index=False
        )

        instructions.to_excel(
            writer,
            sheet_name="instructions",
            index=False
        )

    # --------------------------------------------------------------
    # Save Reviewer 1
    # --------------------------------------------------------------

    with pd.ExcelWriter(
        REVIEWER1_PATH,
        engine="openpyxl"
    ) as writer:

        reviewer1.to_excel(
            writer,
            sheet_name="ratings",
            index=False
        )

        instructions.to_excel(
            writer,
            sheet_name="instructions",
            index=False
        )

    # --------------------------------------------------------------
    # Save Reviewer 2
    # --------------------------------------------------------------

    with pd.ExcelWriter(
        REVIEWER2_PATH,
        engine="openpyxl"
    ) as writer:

        reviewer2.to_excel(
            writer,
            sheet_name="ratings",
            index=False
        )

        instructions.to_excel(
            writer,
            sheet_name="instructions",
            index=False
        )

    print("\nFiles created:")
    print(MASTER_PATH)
    print(REVIEWER1_PATH)
    print(REVIEWER2_PATH)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    stimuli = load_stimuli()

    print(
        f"Loaded {len(stimuli)} stimuli "
        f"from {stimuli['scenario_id'].nunique()} scenarios."
    )

    validation_sample = sample_validation_stimuli(
        stimuli
    )

    verify_sample(
        validation_sample
    )

    save_validation_files(
        validation_sample
    )

    print(
        "\n✓ Human-validation sample successfully created."
    )


if __name__ == "__main__":
    main()
