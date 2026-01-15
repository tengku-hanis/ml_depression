# ML models - boosted tree @ XGBoost

# Packages ---------------------------------------------------------------

library(tidymodels)
library(themis)
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
xgboost_rec <- 
  recipe(cesd_cat20 ~., data = dat_train) %>% 
  step_other(all_factor_predictors()) %>% 
  step_novel(all_factor_predictors()) %>% 
  step_rose(cesd_cat20, skip = T, seed = 123) %>% 
  step_zv(all_nominal_predictors()) %>%
  step_YeoJohnson(all_numeric_predictors()) %>% 
  step_normalize(all_numeric_predictors()) %>% 
  step_dummy(all_factor_predictors())

# 10-fold CV
set.seed(123)
dat_cv <- vfold_cv(dat_train, v = 10)


# Metrics -----------------------------------------------------------------

# Custom F2 score
f_meas_two_vec <- function(truth, estimate, estimator = NULL, na_rm = TRUE, ...) {
  yardstick::f_meas_vec(
    truth = truth, 
    estimate = estimate, 
    beta = 2, 
    estimator = estimator, 
    na_rm = na_rm,
    ...
  )
}

f_meas_two <- function(data, truth, estimate, estimator = NULL, na_rm = TRUE, ...) {
  yardstick::class_metric_summarizer(
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
f_meas_two <- yardstick::new_class_metric(f_meas_two, "maximize")


# Model spec -------------------------------------------------------------

xgboost_mod <- 
  boost_tree(
    tree_depth = tune(), 
    tree = tune(), 
    learn_rate = tune(),
    mtry = tune(), 
    min_n = tune()
  ) %>% 
  set_engine("xgboost") %>% 
  set_mode("classification")


# Workflow ---------------------------------------------------------------

xgboost_wf <- 
  workflow() %>% 
  add_recipe(xgboost_rec) %>% 
  add_model(xgboost_mod)


# Tuning -----------------------------------------------------------------

cesd_metrics <- 
  metric_set(accuracy, sens, spec, precision, kap, mcc, roc_auc, pr_auc, f_meas_two, j_index)

ctrl <- 
  control_race(verbose = T,
               save_pred = T,
               save_workflow = T)


# Parallel processing
# Force use 12 cores (leaving 4 for the OS)
# Use 'multicore' because on Ubuntu (it's faster/stable than multisession)
plan(multicore, workers = 12)

tictoc::tic()
set.seed(123)
xgboost_tuned <- 
  tune_race_anova(
    xgboost_wf,
    resamples = dat_cv,
    grid = 1000,
    metrics = cesd_metrics,
    control = ctrl
  )
tictoc::toc() # 2120.371 sec elapsed
beepr::beep(3) 

# Save CV result
# readr::write_rds(xgboost_tuned, "full_data_analysis_7Dec2025/cv_results/30features/xgboost_cv_result.rds")


# Expore tuned results ---------------------------------------------------

# Max value for each metric
xgboost_tuned %>% 
  collect_metrics() %>% 
  group_by(.metric) %>% 
  summarise(mean = max(mean, na.rm = T))

# Plot
autoplot(xgboost_tuned)
autoplot(xgboost_tuned, metric = "sens")
autoplot(xgboost_tuned, metric = "spec")
autoplot(xgboost_tuned, metric = "pr_auc")
autoplot(xgboost_tuned, metric = "f_meas_two")

# Show best
xgboost_tuned %>% 
  show_best(metric = "sens")
xgboost_tuned %>% 
  show_best(metric = "pr_auc")
xgboost_tuned %>% 
  show_best(metric = "spec")
xgboost_tuned %>% 
  show_best(metric = "f_meas_two")


# Finalise workflow ------------------------------------------------------

# Finalise workflow
xgboost_final_wf <- 
  xgboost_tuned %>% 
  select_best(metric = "sens") %>% 
  finalize_workflow(xgboost_wf, .)

# Finalise model
xgboost_final_model <- 
  xgboost_final_wf %>% 
  last_fit(split)

# Save final model
# readr::write_rds(xgboost_final_model, "full_data_analysis_7Dec2025/final_models/30features/xgboost_mod.rds")


# Testing performance ----------------------------------------------------

xgboost_final_model %>% 
  collect_predictions() %>% 
  conf_mat(cesd_cat20, estimate = .pred_class) 

xgboost_final_model %>% 
  collect_metrics()

class_metrics <- metric_set(accuracy, sens, spec, precision, kap, mcc, f_meas_two, j_index)
xgboost_final_model %>% 
  collect_predictions() %>% 
  class_metrics(cesd_cat20, estimate = .pred_class)

prob_metrics <- metric_set(roc_auc, pr_auc)
xgboost_final_model %>% 
  collect_predictions() %>% 
  prob_metrics(cesd_cat20, .pred_high)

