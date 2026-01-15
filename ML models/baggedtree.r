# ML models - bagged tree

# Packages ---------------------------------------------------------------

library(tidymodels)
library(themis)
library(baguette)
library(finetune)
library(furrr)


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

dat_train_baked <- bagTree_rec %>% prep() %>% juice()

# 10-fold CV
set.seed(123)
dat_cv <- vfold_cv(dat_train, v = 10)


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

## Formalize Custom F2 score
f_meas_two <- new_class_metric(f_meas_two, "maximize")


# Model spec -------------------------------------------------------------

bagTree_mod <- 
  bag_tree(tree_depth = tune(), min_n = tune(), cost_complexity = tune()) %>% 
  set_engine("rpart", times = 25) %>% # 25 ensemble
  set_mode("classification")


# Workflow ---------------------------------------------------------------

bagTree_wf <- 
  workflow() %>% 
  add_recipe(bagTree_rec) %>% 
  add_model(bagTree_mod)


# Tuning -----------------------------------------------------------------

cesd_metrics <- 
  metric_set(accuracy, sens, spec, precision, kap, mcc, roc_auc, pr_auc, f_meas_two, j_index)

ctrl <- 
  control_race(verbose = T,
               save_pred = T,
               save_workflow = T)


# Parallel processing
plan(multicore) # for linux system, multisession for windows

tictoc::tic()
set.seed(123)
bagTree_tuned <- 
  tune_race_anova(
    bagTree_wf,
    resamples = dat_cv,
    grid = 1000,
    metrics = cesd_metrics,
    control = ctrl
  )
tictoc::toc() 
beepr::beep(3)

# Save CV result
# readr::write_rds(bagTree_tuned, "full_data_analysis_7Dec2025/cv_results/bagTree_cv_result.rds")


# Expore tuned results ---------------------------------------------------

# Max value for each metric
bagTree_tuned %>% 
  collect_metrics() %>% 
  group_by(.metric) %>% 
  summarise(mean = max(mean, na.rm = T))

# Plot
autoplot(bagTree_tuned)
autoplot(bagTree_tuned, metric = "sens")
autoplot(bagTree_tuned, metric = "spec")
autoplot(bagTree_tuned, metric = "pr_auc")
autoplot(bagTree_tuned, metric = "f_meas_two")

# Show best
bagTree_tuned %>% 
  show_best(metric = "sens")
bagTree_tuned %>% 
  show_best(metric = "pr_auc")
bagTree_tuned %>% 
  show_best(metric = "spec")
bagTree_tuned %>% 
  show_best(metric = "f_meas_two")


# Finalise workflow ------------------------------------------------------

# Finalise workflow
bagTree_final_wf <- 
  bagTree_tuned %>% 
  select_best(metric = "sens") %>% 
  finalize_workflow(bagTree_wf, .)

# Finalise model
bagTree_final_model <- 
  bagTree_final_wf %>% 
  last_fit(split)

# Save final model
# readr::write_rds(bagTree_final_model, "full_data_analysis_7Dec2025/final_models/bagTree_mod.rds")


# Testing performance ----------------------------------------------------

bagTree_final_model %>% 
  collect_predictions() %>% 
  conf_mat(cesd_cat20, estimate = .pred_class) 

bagTree_final_model %>% 
  collect_metrics()

class_metrics <- metric_set(accuracy, sens, spec, precision, kap, mcc, f_meas_two, j_index)
bagTree_final_model %>% 
  collect_predictions() %>% 
  class_metrics(cesd_cat20, estimate = .pred_class)

prob_metrics <- metric_set(roc_auc, pr_auc)
bagTree_final_model %>% 
  collect_predictions() %>% 
  prob_metrics(cesd_cat20, .pred_high)

