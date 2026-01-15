# Feature selection

# Packages ---------------------------------------------------------------

library(tidymodels)
library(filtro)
library(desirability2)


# Load data --------------------------------------------------------------

dat <- readr::read_rds("full_data_analysis_7Dec2025/data/dat_train.rds")


# Filtro -----------------------------------------------------------------

# Multicriteria approach - Anova F-test, information gain, AUC-ROC

# Information gain
info_gain_res  <- 
  score_info_gain %>% 
  fit(cesd_cat20 ~ ., data = dat)

# Gain ratio
gain_ratio_res  <- 
  score_gain_ratio %>% 
  fit(cesd_cat20 ~ ., data = dat)

# Symmetrical uncertainty
sym_uncert_res <- 
  score_sym_uncert %>% 
  fit(cesd_cat20 ~ ., data = dat)

# Create a list
class_score_list <- list(
  info_gain_res,
  gain_ratio_res,
  sym_uncert_res
)

# Fill safe values
scores_results <- 
  class_score_list %>% 
  fill_safe_values() %>% 
  select(-outcome)

# Optimise for all 3 approaches
top30_variables <- 
  scores_results %>% 
  show_best_desirability_num(
    maximize(infogain), maximize(gainratio), maximize(symuncert), num_terms = 30
  ) 

# Save
# readr::write_rds(top30_variables, "full_data_analysis_7Dec2025/data/filtro/top30_filtro.rds")

