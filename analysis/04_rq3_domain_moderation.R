# 04_rq3_domain_moderation.R
# RQ3: Decision-Domain Moderation
# ============================================================

# 1. PACKAGES
# ============================================================

library(tidyverse)
library(lme4)
library(lmerTest)
library(emmeans)
library(car)


# 2. LOAD RESPONSE DATA
# ============================================================

response_data <- read_csv(
  "data/annotation/claude_sonnet55_high_context_full_ratings_6750.csv",
  show_col_types = FALSE
)


# 3. PREPARE ANALYSIS DATA
# ============================================================

analysis_df <- response_data %>%
  mutate(
    scenario_id = factor(scenario_id),

    style = factor(
      style,
      levels = c(
        "experiential",
        "analytical",
        "integrated"
      )
    ),

    reasoning_effort = factor(
      reasoning_effort,
      levels = c(
        "none",
        "medium",
        "high"
      )
    ),

    domain = factor(
      domain,
      levels = c(
        "career",
        "relationship",
        "emotional_wellbeing"
      )
    )
  )

# Sum-to-zero contrasts for Type III omnibus tests.

contrasts(analysis_df$style) <- contr.sum(3)
contrasts(analysis_df$reasoning_effort) <- contr.sum(3)
contrasts(analysis_df$domain) <- contr.sum(3)


# 4. MODELED OUTCOMES
# ============================================================

outcomes <- c(
  "collaborative_exploration",
  "directiveness",
  "tradeoff_analysis",
  "uncertainty_acknowledgment",
  "emotional_validation",
  "autonomy_support",
  "stance_endorsement",
  "professional_support_recommendation",
  "safety_escalation",
  "decision_support_utility",
  "respectfulness",
  "contextual_tailoring",
  "sycophancy"
)


# 5. FIT PRIMARY RQ3 MODELS
# ============================================================

# Each outcome is modeled as:
#
# outcome ~ style * reasoning_effort * domain +
#           (1 | scenario_id)
#
# The three focal moderation effects are:
#
#   style × domain
#   reasoning_effort × domain
#   style × reasoning_effort × domain

rq3_models <- list()

for (outcome in outcomes) {

  formula <- as.formula(
    paste0(
      outcome,
      " ~ style * reasoning_effort * domain + ",
      "(1 | scenario_id)"
    )
  )

  rq3_models[[outcome]] <- lmer(
    formula,
    data = analysis_df,
    REML = FALSE
  )
}

cat("\nAll 13 RQ3 models fitted.\n")


# 6. OMNIBUS MODERATION TESTS
# ============================================================

# Three moderation tests are extracted for each of the 13 outcomes:
#
#   13 × 3 = 39 prespecified tests.
#
# BH correction is applied jointly across all 39 tests.

rq3_tests <- list()

for (outcome in outcomes) {

  model <- rq3_models[[outcome]]

  omnibus <- as.data.frame(
    anova(
      model,
      type = 3,
      ddf = "Satterthwaite"
    )
  ) %>%
    rownames_to_column("term")

  keep_terms <- c(
    "style:domain",
    "reasoning_effort:domain",
    "style:reasoning_effort:domain"
  )

  rq3_tests[[outcome]] <- omnibus %>%
    filter(term %in% keep_terms) %>%
    transmute(
      outcome = outcome,
      term = term,
      numerator_df = NumDF,
      denominator_df = DenDF,
      F = `F value`,
      p = `Pr(>F)`
    )
}

rq3_omnibus <- bind_rows(rq3_tests) %>%
  mutate(
    p_BH = p.adjust(
      p,
      method = "BH"
    ),

    eta2_partial =
      (F * numerator_df) /
      (F * numerator_df + denominator_df),

    moderation_test = case_when(
      term == "style:domain" ~
        "Orientation × domain",

      term == "reasoning_effort:domain" ~
        "Reasoning × domain",

      term == "style:reasoning_effort:domain" ~
        "Orientation × reasoning × domain"
    )
  )

cat("\n")
cat("============================================================\n")
cat("RQ3 OMNIBUS MODERATION TESTS\n")
cat("============================================================\n")

print(
  rq3_omnibus %>%
    select(
      outcome,
      moderation_test,
      F,
      numerator_df,
      denominator_df,
      p,
      p_BH,
      eta2_partial
    ),
  n = Inf
)


# 7. IDENTIFY SIGNIFICANT MODERATION EFFECTS
# ============================================================

significant_rq3 <- rq3_omnibus %>%
  filter(
    p_BH < .05
  )

cat("\n")
cat("============================================================\n")
cat("BH-SIGNIFICANT RQ3 MODERATION EFFECTS\n")
cat("============================================================\n")

print(
  significant_rq3 %>%
    select(
      outcome,
      moderation_test,
      F,
      p_BH,
      eta2_partial
    ),
  n = Inf
)


# 8. ORIENTATION × DOMAIN EMMS
# ============================================================

# Follow up all significant orientation × domain interactions.
# EMMs estimate the three communicative orientations separately within career, relationship, and emotional-wellbeing domains.

significant_orientation_outcomes <- rq3_omnibus %>%
  filter(
    moderation_test == "Orientation × domain",
    p_BH < .05
  ) %>%
  pull(outcome)

rq3_orientation_emms <- list()
rq3_orientation_posthocs <- list()

for (outcome in significant_orientation_outcomes) {

  model <- rq3_models[[outcome]]

  emm <- emmeans(
    model,
    ~ style | domain,
    lmer.df = "satterthwaite"
  )

  rq3_orientation_emms[[outcome]] <-
    as.data.frame(emm) %>%
    mutate(
      outcome = outcome
    )

  rq3_orientation_posthocs[[outcome]] <-
    as.data.frame(
      pairs(
        emm,
        adjust = "tukey"
      )
    ) %>%
    mutate(
      outcome = outcome
    )
}

rq3_orientation_emm_table <- bind_rows(
  rq3_orientation_emms
)

rq3_orientation_posthoc_table <- bind_rows(
  rq3_orientation_posthocs
)

cat("\n")
cat("============================================================\n")
cat("RQ3 ORIENTATION × DOMAIN EMMS\n")
cat("============================================================\n")

print(
  rq3_orientation_emm_table,
  n = Inf
)

cat("\n")
cat("============================================================\n")
cat("RQ3 ORIENTATION × DOMAIN TUKEY COMPARISONS\n")
cat("============================================================\n")

print(
  rq3_orientation_posthoc_table,
  n = Inf
)


# 9. REASONING × DOMAIN EMMS
# ============================================================

# Follow up any significant reasoning-effort × domain interactions.
# In the current analysis, safety escalation is the only outcome with a BH-significant reasoning × domain interaction.

significant_reasoning_outcomes <- rq3_omnibus %>%
  filter(
    moderation_test == "Reasoning × domain",
    p_BH < .05
  ) %>%
  pull(outcome)

rq3_reasoning_emms <- list()
rq3_reasoning_posthocs <- list()

for (outcome in significant_reasoning_outcomes) {

  model <- rq3_models[[outcome]]

  emm <- emmeans(
    model,
    ~ reasoning_effort | domain,
    lmer.df = "satterthwaite"
  )

  rq3_reasoning_emms[[outcome]] <-
    as.data.frame(emm) %>%
    mutate(
      outcome = outcome
    )

  rq3_reasoning_posthocs[[outcome]] <-
    as.data.frame(
      pairs(
        emm,
        adjust = "tukey"
      )
    ) %>%
    mutate(
      outcome = outcome
    )
}

rq3_reasoning_emm_table <- bind_rows(
  rq3_reasoning_emms
)

rq3_reasoning_posthoc_table <- bind_rows(
  rq3_reasoning_posthocs
)

cat("\n")
cat("============================================================\n")
cat("RQ3 REASONING × DOMAIN EMMS\n")
cat("============================================================\n")

print(
  rq3_reasoning_emm_table,
  n = Inf
)

cat("\n")
cat("============================================================\n")
cat("RQ3 REASONING × DOMAIN TUKEY COMPARISONS\n")
cat("============================================================\n")

print(
  rq3_reasoning_posthoc_table,
  n = Inf
)


# 10. RANDOM EFFECTS AND ICCS
# ============================================================

rq3_random_effects <- purrr::imap_dfr(
  rq3_models,
  function(model, outcome) {

    vc <- as.data.frame(
      VarCorr(model)
    )

    scenario_var <- vc %>%
      filter(
        grp == "scenario_id"
      ) %>%
      pull(vcov)

    residual_var <- sigma(model)^2

    tibble(
      outcome = outcome,
      scenario_variance = scenario_var,
      residual_variance = residual_var,
      ICC =
        scenario_var /
        (scenario_var + residual_var)
    )
  }
)

cat("\n")
cat("============================================================\n")
cat("RQ3 RANDOM EFFECTS AND ICCS\n")
cat("============================================================\n")

print(
  rq3_random_effects,
  n = Inf
)


# 11. SAVE ANALYSIS TABLES
# ============================================================

dir.create(
  "results/rq3",
  recursive = TRUE,
  showWarnings = FALSE
)

write_csv(
  rq3_omnibus,
  "results/rq3/rq3_primary_moderation_tests.csv"
)

write_csv(
  rq3_orientation_emm_table,
  "results/rq3/rq3_orientation_domain_emms.csv"
)

write_csv(
  rq3_orientation_posthoc_table,
  "results/rq3/rq3_orientation_domain_posthocs.csv"
)

write_csv(
  rq3_reasoning_emm_table,
  "results/rq3/rq3_reasoning_domain_emms.csv"
)

write_csv(
  rq3_reasoning_posthoc_table,
  "results/rq3/rq3_reasoning_domain_posthocs.csv"
)

write_csv(
  rq3_random_effects,
  "results/rq3/rq3_random_effects_iccs.csv"
)
