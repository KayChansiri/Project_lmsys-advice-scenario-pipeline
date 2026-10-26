# RQ1: COMMUNICATIVE ORIENTATION × REASONING EFFORT
# Primary analysis: Linear mixed-effects models (LMMs) with a random intercept for underlying advice-seeking scenario.
# Clarity is excluded from inferential analysis because all 6,750   responses received the maximum score.


# 1. PACKAGES
# ============================================================

library(dplyr)
library(tibble)
library(lme4)
library(lmerTest)
library(emmeans)


# 2. OUTCOMES
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


# 3. FACTOR CODING
# ============================================================

# Sum-to-zero contrasts are used so that Type III omnibus tests evaluate each main effect averaged across levels of the other experimental factor.

analysis_df$style <- factor(
  analysis_df$style,
  levels = c(
    "experiential",
    "analytical",
    "integrated"
  )
)

analysis_df$reasoning_effort <- factor(
  analysis_df$reasoning_effort,
  levels = c(
    "none",
    "medium",
    "high"
  )
)

contrasts(analysis_df$style) <- contr.sum(3)
contrasts(analysis_df$reasoning_effort) <- contr.sum(3)

cat("\nSTYLE CONTRASTS\n")
print(contrasts(analysis_df$style))

cat("\nREASONING CONTRASTS\n")
print(contrasts(analysis_df$reasoning_effort))


# 4. PRIMARY RQ1 MODELS
# ============================================================

# Model:
# outcome ~ style * reasoning_effort + (1 | scenario_id)

models <- list()

for (outcome in outcomes) {

  formula_i <- as.formula(
    paste0(
      outcome,
      " ~ style * reasoning_effort + (1 | scenario_id)"
    )
  )

  models[[outcome]] <- lmer(
    formula_i,
    data = analysis_df,
    REML = FALSE
  )
}

cat("\nAll 13 RQ1 models fitted.\n")


# 5. MODEL DIAGNOSTICS
# ============================================================

diagnostics <- data.frame(
  outcome = outcomes,
  singular = sapply(
    models,
    function(x) isSingular(x, tol = 1e-4)
  )
)

cat("\nMODEL DIAGNOSTICS\n")
print(diagnostics)


# 6. TYPE III OMNIBUS TESTS
# ============================================================

omnibus_list <- list()

for (outcome in outcomes) {

  model <- models[[outcome]]

  a <- anova(
    model,
    type = 3,
    ddf = "Satterthwaite"
  )

  a <- as.data.frame(a)
  a$effect_raw <- rownames(a)

  a <- a[
    a$effect_raw %in%
      c(
        "style",
        "reasoning_effort",
        "style:reasoning_effort"
      ),
  ]

  a$partial_eta2 <- with(
    a,
    (`F value` * NumDF) /
      ((`F value` * NumDF) + DenDF)
  )

  omnibus_list[[outcome]] <- data.frame(
    outcome = outcome,
    effect = a$effect_raw,
    df_num = a$NumDF,
    df_den = a$DenDF,
    F = a$`F value`,
    p_raw = a$`Pr(>F)`,
    partial_eta2 = a$partial_eta2
  )
}

omnibus_table <- bind_rows(omnibus_list)

stopifnot(nrow(omnibus_table) == 39)


# 7. BENJAMINI-HOCHBERG CORRECTION
# ============================================================

# BH correction is applied jointly across the 39 primary RQ1
# omnibus tests: 13 outcomes × 3 experimental effects.

omnibus_table$p_BH <- p.adjust(
  omnibus_table$p_raw,
  method = "BH"
)

omnibus_table$effect <- recode(
  omnibus_table$effect,
  "style" = "Communicative orientation",
  "reasoning_effort" = "Reasoning effort",
  "style:reasoning_effort" = "Orientation × reasoning"
)

cat("\nFULL RQ1 OMNIBUS RESULTS\n")

print(
  as_tibble(
    omnibus_table %>%
      arrange(outcome, effect)
  ),
  n = 39,
  width = Inf
)


# 8. PUBLICATION TABLE
# ============================================================

publication_table <- omnibus_table %>%
  mutate(
    F = round(F, 2),
    df_num = round(df_num, 0),
    df_den = round(df_den, 2),
    partial_eta2 = round(partial_eta2, 4),

    p_BH_display = case_when(
      p_BH < .001 ~ "< .001",
      TRUE ~ sprintf("%.3f", p_BH)
    )
  ) %>%
  select(
    outcome,
    effect,
    F,
    df_num,
    df_den,
    p_BH_display,
    partial_eta2
  ) %>%
  arrange(outcome, effect)

cat("\nPUBLICATION-READY OMNIBUS RESULTS\n")

print(
  as_tibble(publication_table),
  n = 39,
  width = Inf
)


# 9. ESTIMATED MARGINAL MEANS: COMMUNICATIVE ORIENTATION
# ============================================================

orientation_emm_list <- list()

for (outcome in outcomes) {

  emm <- emmeans(
    models[[outcome]],
    ~ style,
    lmer.df = "satterthwaite"
  )

  emm_df <- as.data.frame(
    confint(
      emm,
      level = .95
    )
  )

  orientation_emm_list[[outcome]] <- data.frame(
    outcome = outcome,
    orientation = emm_df$style,
    EMM = emm_df$emmean,
    SE = emm_df$SE,
    df = emm_df$df,
    CI_lower = emm_df$lower.CL,
    CI_upper = emm_df$upper.CL
  )
}

orientation_emms <- bind_rows(
  orientation_emm_list
)

cat("\nORIENTATION EMMs\n")

print(
  as_tibble(orientation_emms),
  n = nrow(orientation_emms),
  width = Inf
)


# 10. ESTIMATED MARGINAL MEANS: REASONING EFFORT
# ============================================================

reasoning_emm_list <- list()

for (outcome in outcomes) {

  emm <- emmeans(
    models[[outcome]],
    ~ reasoning_effort,
    lmer.df = "satterthwaite"
  )

  emm_df <- as.data.frame(
    confint(
      emm,
      level = .95
    )
  )

  reasoning_emm_list[[outcome]] <- data.frame(
    outcome = outcome,
    reasoning_effort = emm_df$reasoning_effort,
    EMM = emm_df$emmean,
    SE = emm_df$SE,
    df = emm_df$df,
    CI_lower = emm_df$lower.CL,
    CI_upper = emm_df$upper.CL
  )
}

reasoning_emms <- bind_rows(
  reasoning_emm_list
)

cat("\nREASONING-EFFORT EMMs\n")

print(
  as_tibble(reasoning_emms),
  n = nrow(reasoning_emms),
  width = Inf
)


# 11. TUKEY-ADJUSTED PAIRWISE COMPARISONS
# ============================================================

# Pairwise comparisons are calculated from the same primary models.
# Tukey adjustment is applied within each outcome/factor family.

orientation_pairwise <- list()
reasoning_pairwise <- list()

for (outcome in outcomes) {

  orientation_pairwise[[outcome]] <-
    as.data.frame(
      pairs(
        emmeans(
          models[[outcome]],
          ~ style,
          lmer.df = "satterthwaite"
        ),
        adjust = "tukey"
      )
    )

  orientation_pairwise[[outcome]]$outcome <-
    outcome

  reasoning_pairwise[[outcome]] <-
    as.data.frame(
      pairs(
        emmeans(
          models[[outcome]],
          ~ reasoning_effort,
          lmer.df = "satterthwaite"
        ),
        adjust = "tukey"
      )
    )

  reasoning_pairwise[[outcome]]$outcome <-
    outcome
}

orientation_pairwise_table <-
  bind_rows(orientation_pairwise) %>%
  select(
    outcome,
    contrast,
    estimate,
    SE,
    df,
    t.ratio,
    p.value
  )

reasoning_pairwise_table <-
  bind_rows(reasoning_pairwise) %>%
  select(
    outcome,
    contrast,
    estimate,
    SE,
    df,
    t.ratio,
    p.value
  )

cat("\nORIENTATION PAIRWISE COMPARISONS\n")

print(
  as_tibble(orientation_pairwise_table),
  n = nrow(orientation_pairwise_table),
  width = Inf
)

cat("\nREASONING PAIRWISE COMPARISONS\n")

print(
  as_tibble(reasoning_pairwise_table),
  n = nrow(reasoning_pairwise_table),
  width = Inf
)


# 12. MANUSCRIPT-REPORTED PAIRWISE COMPARISONS
# ============================================================

# These subsets provide a direct reproducibility check for the  pairwise comparisons discussed in the RQ1 results text.

manuscript_orientation_pairwise <-
  orientation_pairwise_table %>%
  filter(
    outcome %in%
      c(
        "emotional_validation",
        "tradeoff_analysis"
      )
  )

manuscript_reasoning_pairwise <-
  reasoning_pairwise_table %>%
  filter(
    outcome %in%
      c(
        "tradeoff_analysis",
        "uncertainty_acknowledgment",
        "emotional_validation",
        "autonomy_support"
      )
  )

cat("\nMANUSCRIPT CHECK: ORIENTATION PAIRWISE COMPARISONS\n")

print(
  as_tibble(manuscript_orientation_pairwise),
  n = nrow(manuscript_orientation_pairwise),
  width = Inf
)

cat("\nMANUSCRIPT CHECK: REASONING PAIRWISE COMPARISONS\n")

print(
  as_tibble(manuscript_reasoning_pairwise),
  n = nrow(manuscript_reasoning_pairwise),
  width = Inf
)


# 13. RANDOM EFFECTS AND ICC
# ============================================================

random_effects_list <- list()

for (outcome in outcomes) {

  model <- models[[outcome]]

  vc <- as.data.frame(
    VarCorr(model)
  )

  scenario_variance <- vc$vcov[
    vc$grp == "scenario_id" &
      is.na(vc$var2)
  ][1]

  residual_variance <- vc$vcov[
    vc$grp == "Residual"
  ][1]

  icc <- scenario_variance /
    (
      scenario_variance +
        residual_variance
    )

  random_effects_list[[outcome]] <-
    data.frame(
      outcome = outcome,
      scenario_variance = scenario_variance,
      residual_variance = residual_variance,
      ICC = icc
    )
}

random_effects_table <- bind_rows(
  random_effects_list
)

cat("\nRANDOM EFFECTS AND ICC\n")

print(
  as_tibble(random_effects_table),
  n = 13,
  width = Inf
)


# 14. ICC SUMMARY
# ============================================================

icc_summary <- random_effects_table %>%
  summarise(
    min_ICC = min(
      ICC,
      na.rm = TRUE
    ),
    median_ICC = median(
      ICC,
      na.rm = TRUE
    ),
    max_ICC = max(
      ICC,
      na.rm = TRUE
    )
  )

cat("\nICC SUMMARY\n")
print(icc_summary)


# 15. MESSAGE-LENGTH SENSITIVITY ANALYSIS
# ============================================================

# Message length is treated as a sensitivity covariate rather than part of the primary specification because it is a post-transformation characteristic and may partly reflect the communicative-orientation manipulation itself.

analysis_df <- analysis_df %>%
  mutate(
    transformed_word_count_z =
      as.numeric(
        scale(transformed_word_count)
      )
  )

length_models <- list()

for (outcome in outcomes) {

  formula_i <- as.formula(
    paste0(
      outcome,
      " ~ style * reasoning_effort + ",
      "transformed_word_count_z + ",
      "(1 | scenario_id)"
    )
  )

  length_models[[outcome]] <- lmer(
    formula_i,
    data = analysis_df,
    REML = FALSE
  )
}


# 16. LENGTH-COVARIATE EFFECTS
# ============================================================

length_effects <- list()

for (outcome in outcomes) {

  coef_table <- summary(
    length_models[[outcome]]
  )$coefficients

  length_effects[[outcome]] <-
    data.frame(
      outcome = outcome,

      estimate =
        coef_table[
          "transformed_word_count_z",
          "Estimate"
        ],

      SE =
        coef_table[
          "transformed_word_count_z",
          "Std. Error"
        ],

      p_raw =
        coef_table[
          "transformed_word_count_z",
          "Pr(>|t|)"
        ]
    )
}

length_effects_table <-
  bind_rows(length_effects)

length_effects_table$p_BH <-
  p.adjust(
    length_effects_table$p_raw,
    method = "BH"
  )

cat("\nMESSAGE-LENGTH EFFECTS\n")

print(
  as_tibble(length_effects_table),
  n = 13,
  width = Inf
)


# 17. SENSITIVITY OMNIBUS TESTS
# ============================================================

sensitivity_omnibus_list <- list()

for (outcome in outcomes) {

  a <- anova(
    length_models[[outcome]],
    type = 3,
    ddf = "Satterthwaite"
  )

  a <- as.data.frame(a)
  a$effect <- rownames(a)

  a <- a[
    a$effect %in%
      c(
        "style",
        "reasoning_effort",
        "style:reasoning_effort"
      ),
  ]

  sensitivity_omnibus_list[[outcome]] <-
    data.frame(
      outcome = outcome,
      effect = a$effect,
      df_num = a$NumDF,
      df_den = a$DenDF,
      F = a$`F value`,
      p_raw = a$`Pr(>F)`
    )
}

sensitivity_omnibus <-
  bind_rows(
    sensitivity_omnibus_list
  )

sensitivity_omnibus$p_BH <-
  p.adjust(
    sensitivity_omnibus$p_raw,
    method = "BH"
  )

cat("\nLENGTH-ADJUSTED OMNIBUS TESTS\n")

print(
  as_tibble(sensitivity_omnibus),
  n = 39,
  width = Inf
)


# 18. FINAL SANITY CHECKS
# ============================================================

cat("\nRQ1 SANITY CHECKS\n")

cat(
  "Observations:",
  nrow(analysis_df),
  "\n"
)

cat(
  "Scenarios:",
  length(
    unique(analysis_df$scenario_id)
  ),
  "\n"
)

cat(
  "Primary omnibus tests:",
  nrow(omnibus_table),
  "\n"
)

cat("\nSignificant primary tests after BH correction:\n")

print(
  omnibus_table %>%
    mutate(
      significant = p_BH < .05
    ) %>%
    group_by(effect) %>%
    summarise(
      significant_tests = sum(significant),
      total_tests = n(),
      .groups = "drop"
    )
)

cat("\nICC range:\n")
print(icc_summary)

cat("\nRQ1 analysis complete.\n")
