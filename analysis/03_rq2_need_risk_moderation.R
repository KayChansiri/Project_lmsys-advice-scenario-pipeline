# 03_rq2_need_risk_moderation.R
# RQ2: Scenario-Level Need/Risk Moderation
# ============================================================

# 1. PACKAGES
# ============================================================

library(tidyverse)
library(readxl)
library(lme4)
library(lmerTest)
library(emmeans)
library(car)


# 2. LOAD RESPONSE DATA AND SCENARIO NEED/RISK RATINGS
# ============================================================

# Update paths as needed.
response_data <- read_csv(
  "data/annotation/claude_sonnet55_high_context_full_ratings_6750.csv",
  show_col_types = FALSE
)

need_risk <- read_excel(
  "data/human_validation/scenario_need_risk_reviewer1.xlsx",
  sheet = "ratings"
)


# 3. PREPARE ANALYSIS DATA
# ============================================================

# Merge the scenario-level need/risk indicators onto the response-level data.

analysis_df <- response_data %>%
  left_join(need_risk, by = "scenario_id") %>%
  mutate(
    scenario_id = factor(scenario_id),
    style = factor(
      style,
      levels = c("experiential", "analytical", "integrated")
    ),
    reasoning_effort = factor(
      reasoning_effort,
      levels = c("none", "medium", "high")
    )
  )

# Sum-to-zero contrasts for omnibus tests.
contrasts(analysis_df$style) <- contr.sum(3)
contrasts(analysis_df$reasoning_effort) <- contr.sum(3)


# 4. PRESPECIFIED NEED/RISK–OUTCOME PAIRS
# ============================================================

rq2_pairs <- tribble(
  ~outcome,                              ~indicator,
  "professional_support_recommendation", "professional_support_indicated",
  "safety_escalation",                   "safety_escalation_indicated",
  "directiveness",                       "direct_guidance_appropriate",
  "collaborative_exploration",           "reflective_exploration_beneficial",
  "emotional_validation",                "emotional_validation_warranted",
  "stance_endorsement",                  "stance_endorsement_risky",
  "sycophancy",                          "stance_endorsement_risky"
)


# 5. FIT PRIMARY RQ2 MODELS
# ============================================================

# Each outcome is modeled as:
#
# outcome ~ style * reasoning_effort * need_indicator +
#           (1 | scenario_id)

rq2_models <- list()

for (i in seq_len(nrow(rq2_pairs))) {

  outcome <- rq2_pairs$outcome[i]
  indicator <- rq2_pairs$indicator[i]

  analysis_df[[indicator]] <- factor(
    analysis_df[[indicator]],
    levels = c(0, 1)
  )
  contrasts(analysis_df[[indicator]]) <- contr.sum(2)

  formula <- as.formula(
    paste0(
      outcome,
      " ~ style * reasoning_effort * ",
      indicator,
      " + (1 | scenario_id)"
    )
  )

  model_name <- paste(outcome, indicator, sep = "__")

  rq2_models[[model_name]] <- lmer(
    formula,
    data = analysis_df,
    REML = FALSE
  )
}


# 6. OMNIBUS MODERATION TESTS
# ============================================================

# Extract the three prespecified moderation effects:
#   style × need
#   reasoning × need
#   style × reasoning × need
#
# BH correction is applied jointly across all 21 tests.

rq2_tests <- list()

for (i in seq_len(nrow(rq2_pairs))) {

  outcome <- rq2_pairs$outcome[i]
  indicator <- rq2_pairs$indicator[i]
  model_name <- paste(outcome, indicator, sep = "__")

  model <- rq2_models[[model_name]]

  omnibus <- as.data.frame(
    anova(model, type = 3, ddf = "Satterthwaite")
  ) %>%
    rownames_to_column("term")

  keep_terms <- c(
    paste0("style:", indicator),
    paste0("reasoning_effort:", indicator),
    paste0("style:reasoning_effort:", indicator)
  )

  tmp <- omnibus %>%
    filter(term %in% keep_terms) %>%
    transmute(
      outcome = outcome,
      indicator = indicator,
      term = term,
      numerator_df = NumDF,
      denominator_df = DenDF,
      F = `F value`,
      p = `Pr(>F)`
    )

  rq2_tests[[model_name]] <- tmp
}

rq2_omnibus <- bind_rows(rq2_tests) %>%
  mutate(
    p_BH = p.adjust(p, method = "BH"),

    # Partial eta squared from the F statistic.
    eta2_partial =
      (F * numerator_df) /
      (F * numerator_df + denominator_df),

    moderation_test = case_when(
      str_starts(term, "style:reasoning") ~
        "Orientation × reasoning × need",
      str_starts(term, "style:") ~
        "Orientation × need",
      str_starts(term, "reasoning_effort:") ~
        "Reasoning × need"
    )
  )

print(
  rq2_omnibus %>%
    select(
      outcome,
      indicator,
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


# 7. EMMS FOR SIGNIFICANT ORIENTATION × NEED EFFECTS
# ============================================================

significant_rq2_pairs <- tribble(
  ~outcome,                              ~indicator,
  "professional_support_recommendation", "professional_support_indicated",
  "safety_escalation",                   "safety_escalation_indicated",
  "emotional_validation",                "emotional_validation_warranted",
  "stance_endorsement",                  "stance_endorsement_risky"
)

rq2_emms <- list()
rq2_posthocs <- list()

for (i in seq_len(nrow(significant_rq2_pairs))) {

  outcome <- significant_rq2_pairs$outcome[i]
  indicator <- significant_rq2_pairs$indicator[i]
  model_name <- paste(outcome, indicator, sep = "__")

  model <- rq2_models[[model_name]]

  emm_formula <- as.formula(
    paste0("~ style | ", indicator)
  )

  emm <- emmeans(
    model,
    emm_formula,
    lmer.df = "satterthwaite"
  )

  rq2_emms[[model_name]] <- as.data.frame(emm) %>%
    mutate(
      outcome = outcome,
      indicator = indicator
    )

  rq2_posthocs[[model_name]] <- as.data.frame(
    pairs(emm, adjust = "tukey")
  ) %>%
    mutate(
      outcome = outcome,
      indicator = indicator
    )
}

rq2_emm_table <- bind_rows(rq2_emms)
rq2_posthoc_table <- bind_rows(rq2_posthocs)

cat("\nRQ2 ESTIMATED MARGINAL MEANS\n")
print(rq2_emm_table, n = Inf)

cat("\nRQ2 TUKEY-ADJUSTED POST-HOC COMPARISONS\n")
print(rq2_posthoc_table, n = Inf)


# 8. RANDOM EFFECTS AND ICCS
# ============================================================

rq2_random_effects <- purrr::imap_dfr(
  rq2_models,
  function(model, model_name) {

    vc <- as.data.frame(VarCorr(model))

    scenario_var <- vc %>%
      filter(grp == "scenario_id") %>%
      pull(vcov)

    residual_var <- sigma(model)^2

    tibble(
      model = model_name,
      scenario_variance = scenario_var,
      residual_variance = residual_var,
      ICC = scenario_var /
        (scenario_var + residual_var)
    )
  }
)

cat("\nRQ2 RANDOM EFFECTS AND ICCS\n")
print(rq2_random_effects, n = Inf)


# 9. SPARSE-CATEGORY SENSITIVITY ANALYSES
# ============================================================

# Reduced two-way models are estimated for indicators with fewer than
# 30 scenarios in the minority category.

sparse_pairs <- tribble(
  ~outcome,              ~indicator,
  "safety_escalation",   "safety_escalation_indicated",
  "directiveness",       "direct_guidance_appropriate",
  "stance_endorsement",  "stance_endorsement_risky",
  "sycophancy",          "stance_endorsement_risky"
)

sparse_models <- list()
sparse_tests <- list()

for (i in seq_len(nrow(sparse_pairs))) {

  outcome <- sparse_pairs$outcome[i]
  indicator <- sparse_pairs$indicator[i]

  formula <- as.formula(
    paste0(
      outcome,
      " ~ style * reasoning_effort + ",
      "style * ", indicator, " + ",
      "reasoning_effort * ", indicator,
      " + (1 | scenario_id)"
    )
  )

  model_name <- paste(outcome, indicator, sep = "__")

  model <- lmer(
    formula,
    data = analysis_df,
    REML = FALSE
  )

  sparse_models[[model_name]] <- model

  omnibus <- as.data.frame(
    anova(model, type = 3, ddf = "Satterthwaite")
  ) %>%
    rownames_to_column("term")

  keep_terms <- c(
    paste0("style:", indicator),
    paste0("reasoning_effort:", indicator)
  )

  sparse_tests[[model_name]] <- omnibus %>%
    filter(term %in% keep_terms) %>%
    transmute(
      outcome = outcome,
      indicator = indicator,
      term = term,
      numerator_df = NumDF,
      denominator_df = DenDF,
      F = `F value`,
      p = `Pr(>F)`
    )
}

sparse_results <- bind_rows(sparse_tests) %>%
  mutate(
    p_BH = p.adjust(p, method = "BH"),
    eta2_partial =
      (F * numerator_df) /
      (F * numerator_df + denominator_df)
  )

cat("\nRQ2 SPARSE-CATEGORY SENSITIVITY ANALYSES\n")

print(
  sparse_results,
  n = Inf
)


# 10. SAVE ANALYSIS TABLES
# ============================================================

dir.create(
  "results/rq2",
  recursive = TRUE,
  showWarnings = FALSE
)

write_csv(
  rq2_omnibus,
  "results/rq2/rq2_primary_moderation_tests.csv"
)

write_csv(
  rq2_emm_table,
  "results/rq2/rq2_significant_interaction_emms.csv"
)

write_csv(
  rq2_posthoc_table,
  "results/rq2/rq2_significant_interaction_posthocs.csv"
)

write_csv(
  rq2_random_effects,
  "results/rq2/rq2_random_effects_iccs.csv"
)

write_csv(
  sparse_results,
  "results/rq2/rq2_sparse_sensitivity.csv"
)
