
# Descriptive Characteristics of Scenarios, Stimuli, and Model Responses
# Reproduces descriptive statistics reported in: Section 4.1, Appendix 9, Appendix 10
# This script assumes `analysis_df` has already been created by merging the model-generation data with the response-level annotation data by generation_id.
-------------
library(dplyr)
# 1. ORIGINAL ADVICE-SEEKING SCENARIOS
original_df <- analysis_df %>%
  distinct(
    scenario_id,
    human_primary_domain,
    source_text
  )

original_df$source_word_count <- sapply(
  strsplit(trimws(original_df$source_text), "\\s+"),
  length
)

# Overall descriptive statistics
original_descriptives <- original_df %>%
  summarise(
    n_scenarios = n(),
    mean_words = mean(source_word_count, na.rm = TRUE),
    sd_words = sd(source_word_count, na.rm = TRUE),
    median_words = median(source_word_count, na.rm = TRUE),
    min_words = min(source_word_count, na.rm = TRUE),
    max_words = max(source_word_count, na.rm = TRUE)
  )

original_descriptives


# By decision domain
original_descriptives_domain <- original_df %>%
  group_by(human_primary_domain) %>%
  summarise(
    n = n(),
    mean_words = mean(source_word_count, na.rm = TRUE),
    sd_words = sd(source_word_count, na.rm = TRUE),
    median_words = median(source_word_count, na.rm = TRUE),
    min_words = min(source_word_count, na.rm = TRUE),
    max_words = max(source_word_count, na.rm = TRUE),
    .groups = "drop"
  )

original_descriptives_domain

# 2. TRANSFORMED EXPERIMENTAL STIMULI

analysis_df$transformed_word_count <- sapply(
  strsplit(trimws(analysis_df$transformed_text), "\\s+"),
  length
)

# Use one row per unique transformed stimulus.
stimulus_descriptives <- analysis_df %>%
  distinct(
    stimulus_id,
    scenario_id,
    human_primary_domain,
    style,
    transformed_word_count
  ) %>%
  group_by(style) %>%
  summarise(
    n_stimuli = n(),
    mean_words = mean(transformed_word_count, na.rm = TRUE),
    sd_words = sd(transformed_word_count, na.rm = TRUE),
    median_words = median(transformed_word_count, na.rm = TRUE),
    min_words = min(transformed_word_count, na.rm = TRUE),
    max_words = max(transformed_word_count, na.rm = TRUE),
    .groups = "drop"
  )

stimulus_descriptives

# 3. GENERATED MODEL RESPONSES — OVERALL

response_descriptives_overall <- analysis_df %>%
  summarise(
    n = n(),

    mean_words = mean(response_word_count, na.rm = TRUE),
    sd_words = sd(response_word_count, na.rm = TRUE),
    median_words = median(response_word_count, na.rm = TRUE),
    min_words = min(response_word_count, na.rm = TRUE),
    max_words = max(response_word_count, na.rm = TRUE),

    mean_input_tokens = mean(input_tokens, na.rm = TRUE),
    sd_input_tokens = sd(input_tokens, na.rm = TRUE),

    mean_output_tokens = mean(output_tokens, na.rm = TRUE),
    sd_output_tokens = sd(output_tokens, na.rm = TRUE),

    mean_reasoning_tokens = mean(reasoning_tokens, na.rm = TRUE),
    sd_reasoning_tokens = sd(reasoning_tokens, na.rm = TRUE),

    mean_total_tokens = mean(total_tokens, na.rm = TRUE),
    sd_total_tokens = sd(total_tokens, na.rm = TRUE)
  )

response_descriptives_overall


# 4. GENERATED RESPONSES BY EXPERIMENTAL CONDITION
# Appendix 9

response_descriptives_condition <- analysis_df %>%
  group_by(style, reasoning_effort) %>%
  summarise(
    n = n(),

    mean_words = mean(response_word_count, na.rm = TRUE),
    sd_words = sd(response_word_count, na.rm = TRUE),

    mean_output_tokens = mean(output_tokens, na.rm = TRUE),
    sd_output_tokens = sd(output_tokens, na.rm = TRUE),

    mean_reasoning_tokens = mean(reasoning_tokens, na.rm = TRUE),
    sd_reasoning_tokens = sd(reasoning_tokens, na.rm = TRUE),

    mean_total_tokens = mean(total_tokens, na.rm = TRUE),
    sd_total_tokens = sd(total_tokens, na.rm = TRUE),

    .groups = "drop"
  )

response_descriptives_condition

# 5. RESPONSE-LEVEL BEHAVIOR DISTRIBUTIONS
#  Appendix 10
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
  "clarity",
  "respectfulness",
  "contextual_tailoring",
  "sycophancy"
)

distribution_table <- do.call(
  rbind,
  lapply(outcomes, function(x) {

    counts <- table(
      factor(
        analysis_df[[x]],
        levels = 0:2
      )
    )

    percentages <- prop.table(counts) * 100

    data.frame(
      outcome = x,

      n_0 = as.integer(counts[1]),
      pct_0 = round(as.numeric(percentages[1]), 1),

      n_1 = as.integer(counts[2]),
      pct_1 = round(as.numeric(percentages[2]), 1),

      n_2 = as.integer(counts[3]),
      pct_2 = round(as.numeric(percentages[3]), 1)
    )
  })
)

rownames(distribution_table) <- NULL

distribution_table

# 6. BASIC REPRODUCIBILITY CHECKS

stopifnot(nrow(analysis_df) == 6750)
stopifnot(length(unique(analysis_df$scenario_id)) == 150)

# 150 scenarios × 3 communicative orientations = 450 stimuli
stopifnot(
  nrow(
    analysis_df %>%
      distinct(stimulus_id)
  ) == 450
)

# Balanced 3 × 3 design: 750 responses per cell
cell_counts <- analysis_df %>%
  count(style, reasoning_effort)

stopifnot(all(cell_counts$n == 750))

# Clarity should be invariant in the analyzed dataset.
stopifnot(
  all(analysis_df$clarity == 2, na.rm = TRUE)
)

cat(
  "\nDescriptive-analysis checks completed successfully.\n"
)

# 7. DOMAIN-SPECIFIC DESCRIPTIVE STATISTICS (Appendix 11)

# Appendix 11 reports raw means and standard deviations for the 13 modeled advice behaviors within each decision domain.
# Clarity is excluded here because it was invariant across all  responses and therefore excluded from the inferential models.

modeled_outcomes <- c(
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

# Calculate raw M and SD for each behavior within each domain
domain_behavior_descriptives <- analysis_df %>%
  group_by(human_primary_domain) %>%
  summarise(
    across(
      all_of(modeled_outcomes),
      list(
        mean = ~ mean(.x, na.rm = TRUE),
        sd = ~ sd(.x, na.rm = TRUE)
      ),
      .names = "{.col}__{.fn}"
    ),
    .groups = "drop"
  ) %>%
  pivot_longer(
    cols = -human_primary_domain,
    names_to = c("outcome", ".value"),
    names_sep = "__"
  ) %>%
  arrange(
    human_primary_domain,
    match(outcome, modeled_outcomes)
  )


cat("\n")
cat("============================================================\n")
cat("APPENDIX 11: DOMAIN-SPECIFIC DESCRIPTIVE STATISTICS\n")
cat("============================================================\n")

print(
  domain_behavior_descriptives,
  n = Inf,
  width = Inf
)


# Create manuscript-ready M (SD) values
appendix11_table <- domain_behavior_descriptives %>%
  mutate(
    statistic = sprintf(
      "%.2f (%.2f)",
      mean,
      sd
    )
  ) %>%
  select(
    outcome,
    human_primary_domain,
    statistic
  ) %>%
  pivot_wider(
    names_from = human_primary_domain,
    values_from = statistic
  ) %>%
  arrange(
    match(outcome, modeled_outcomes)
  )


cat("\n")
cat("============================================================\n")
cat("APPENDIX 11: MANUSCRIPT-READY TABLE — M (SD)\n")
cat("============================================================\n")

print(
  appendix11_table,
  n = Inf,
  width = Inf
)


# Identify the three highest raw means within each domain.
# This is retained as a descriptive reproducibility output.
domain_top3_behaviors <- domain_behavior_descriptives %>%
  group_by(human_primary_domain) %>%
  slice_max(
    order_by = mean,
    n = 3,
    with_ties = FALSE
  ) %>%
  arrange(
    human_primary_domain,
    desc(mean)
  ) %>%
  ungroup()


cat("\n")
cat("============================================================\n")
cat("TOP 3 MODELED BEHAVIORS WITHIN EACH DOMAIN\n")
cat("============================================================\n")

print(
  domain_top3_behaviors,
  n = Inf,
  width = Inf
)


# 8. SAVE DESCRIPTIVE RESULTS

dir.create(
  "results/descriptives",
  recursive = TRUE,
  showWarnings = FALSE
)


write.csv(
  original_descriptives,
  "results/descriptives/original_scenario_descriptives.csv",
  row.names = FALSE
)

write.csv(
  original_descriptives_domain,
  "results/descriptives/original_scenario_descriptives_by_domain.csv",
  row.names = FALSE
)

write.csv(
  stimulus_descriptives,
  "results/descriptives/transformed_stimulus_descriptives.csv",
  row.names = FALSE
)

write.csv(
  response_descriptives_overall,
  "results/descriptives/response_descriptives_overall.csv",
  row.names = FALSE
)

write.csv(
  response_descriptives_condition,
  "results/descriptives/appendix9_response_descriptives_by_condition.csv",
  row.names = FALSE
)

write.csv(
  distribution_table,
  "results/descriptives/appendix10_behavior_distributions.csv",
  row.names = FALSE
)

write.csv(
  domain_behavior_descriptives,
  "results/descriptives/appendix11_domain_behavior_descriptives.csv",
  row.names = FALSE
)

write.csv(
  appendix11_table,
  "results/descriptives/appendix11_manuscript_table.csv",
  row.names = FALSE
)

write.csv(
  domain_top3_behaviors,
  "results/descriptives/domain_top3_behaviors.csv",
  row.names = FALSE
)


cat("\n")
cat("============================================================\n")
cat("DESCRIPTIVE RESULTS SAVED\n")
cat("============================================================\n")

cat(
  "\nFiles written to results/descriptives/:\n",
  "\n  original_scenario_descriptives.csv",
  "\n  original_scenario_descriptives_by_domain.csv",
  "\n  transformed_stimulus_descriptives.csv",
  "\n  response_descriptives_overall.csv",
  "\n  appendix9_response_descriptives_by_condition.csv",
  "\n  appendix10_behavior_distributions.csv",
  "\n  appendix11_domain_behavior_descriptives.csv",
  "\n  appendix11_manuscript_table.csv",
  "\n  domain_top3_behaviors.csv",
  "\n"
)
