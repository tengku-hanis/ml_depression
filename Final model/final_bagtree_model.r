# Final bagged tree model 

# Packages ---------------------------------------------------------------

library(tidymodels)
library(themis)
library(baguette)


# Final parameters -------------------------------------------------------

final_params <- readr::read_rds("full_data_analysis_7Dec2025/final_param/final_params.rds")

# Data -------------------------------------------------------------------

dat <- readr::read_rds("data_full.rds")

# Read selected features
feature_selected <- readr::read_rds("full_data_analysis_7Dec2025/data/filtro/top30_filtro.rds")

# Reduced data
dat2 <- 
  dat %>% 
  select(feature_selected$predictor, cesd_cat20)

# Split
set.seed(123)
split <- initial_split(dat2, prop = 0.8, strata = cesd_cat20)
dat_train <- training(split)
dat_test <- testing(split)


# Recipes ----------------------------------------------------------------

set.seed(123)
bagTree_rec <- 
  recipe(cesd_cat20 ~., data = dat_train) %>% 
  step_other(all_factor_predictors()) %>% 
  step_novel(all_factor_predictors()) %>% 
  step_rose(cesd_cat20, skip = T, seed = 123) %>% 
  step_zv(all_nominal_predictors()) %>%
  step_YeoJohnson(all_numeric_predictors()) %>% 
  step_normalize(all_numeric_predictors())


# Model specs ------------------------------------------------------------

bagtree_final_spec <- 
  bag_tree(
    cost_complexity = final_params$cost_complexity[1],
    tree_depth = final_params$tree_depth[1],
    min_n = final_params$min_n[1]
  ) %>% 
  set_engine("rpart", times = 25) %>% 
  set_mode("classification")

# Metrics -----------------------------------------------------------------

# Custom F2 score
f_meas_two_vec <- function(truth, estimate, estimator = NULL, na_rm = TRUE, ...) {
  f_meas_vec(
    truth = truth, 
    estimate = estimate, 
    beta = 2, 
    estimator = estimator, 
    na_rm = na_rm,
    ...
  )
}

f_meas_two <- function(data, truth, estimate, estimator = NULL, na_rm = TRUE, ...) {
  class_metric_summarizer(
    "f_meas_two",
    f_meas_two_vec,
    data = data,
    truth = {{truth}},
    estimate = {{estimate}},
    estimator = estimator,
    na_rm = na_rm,
    ...
  )
}

# Formalize Custom F2 score
f_meas_two <- new_class_metric(f_meas_two, "maximize")

# Metrics
my_metrics <- metric_set(f_meas_two, sens, spec, precision, roc_auc)

# Fit final model --------------------------------------------------------

# Workflow
bagtree_final_wf <- 
  workflow() %>% 
  add_recipe(bagTree_rec) %>% 
  add_model(bagtree_final_spec)
  
# Final model
bagtree_final_model <- 
  bagtree_final_wf %>%
  last_fit(
    split = split,
    metrics = my_metrics
  )


# Performance on the testing dataset -------------------------------------

bagtree_final_model %>% 
  collect_metrics()


# Save model -------------------------------------------------------------

# readr::write_rds(bagtree_final_model, "full_data_analysis_7Dec2025/final_models/ml_mod_depression.rds")
