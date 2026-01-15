# Explainable ML

# Packages ---------------------------------------------------------------

library(tidyverse)
library(tidymodels)
library(themis)
library(baguette)
library(DALEXtra)


# Data -------------------------------------------------------------------

dat <- readr::read_rds("data_full.rds")

# Read selected features
feature_selected <- readr::read_rds("full_data_analysis_7Dec2025/data/filtro/top30_filtro.rds")

# Reduced data
dat2 <- 
  dat %>% 
  select(feature_selected$predictor, cesd_cat20)


# Load model -------------------------------------------------------------

final_mod <- readr::read_rds("full_data_analysis_7Dec2025/final_models/ml_mod_depression.rds")


# tidymodels explainer ---------------------------------------------------

# Extract the actual trained model/workflow from the last_fit object
actual_model <- final_mod %>% extract_workflow()

# Create the explainer
ml_exp <- 
  explain_tidymodels(
    actual_model, 
    data = dat2 %>% select(-cesd_cat20), 
    y = dat2$cesd_cat20 %>% as.numeric(), 
    predict_function_target_column = "high",       
    label = "Bagged tree",
    verbose = TRUE
  )


# Global explainer -------------------------------------------------------

# Custom Loss Function for F2 score
loss_one_minus_f2 <- function(observed, predicted) {
  # 1. Handle OBSERVED (Truth)
  observed_class <- ifelse(observed == 1, "high", "low") 
  observed_factor <- factor(observed_class, levels = c("high", "low"))
  
  # 2. Handle PREDICTED (Estimate)
  # Convert probabilities to class labels
  predicted_class <- ifelse(predicted >= 0.5, "high", "low")
  predicted_factor <- factor(predicted_class, levels = c("high", "low"))
  
  # 3. Safe F2 Calculation
  # We use suppressWarnings to stop the console from spamming if
  # a permutation results in 0 "high" predictions (rare but possible)
  f2 <- suppressWarnings(
    yardstick::f_meas_vec(truth = observed_factor, estimate = predicted_factor, beta = 2)
  )
  
  # 4. Handle Division by Zero (if f2 is NA, it means score is 0)
  if (is.na(f2)) {
    return(1) # Loss is 1 (Max error) because Score was 0
  } else {
    return(1 - f2)
  }
}
attr(loss_one_minus_f2, "loss_name") <- "One minus F2 Score"

# Variable importance (permutation-based) 
tictoc::tic()
set.seed(123)

ml_vp <- model_parts(
    ml_exp,
    loss_function = loss_one_minus_f2,
    B = 50, 
    type = "variable_importance"
) 

tictoc::toc() # 281.413 sec elapsed
beepr::beep(3)

# Save explainer - vi
# write_rds(ml_vp, "full_data_analysis_7Dec2025/explainable_ml/ml_vp.rds")
# ml_vp <- readr::read_rds("full_data_analysis_7Dec2025/explainable_ml/ml_vp.rds")

# Change variable names
ml_vp$variable[ml_vp$variable == "media_use_internet"] <- "How often do you use the internet"
ml_vp$variable[ml_vp$variable == "health_perception"] <- "How do you feel about your current physical health"
ml_vp$variable[ml_vp$variable == "media_use_radio"] <- "How often do you use radio"
ml_vp$variable[ml_vp$variable == "attitude_ageing_excluded"] <- "Do you feel excluded because of your age"
ml_vp$variable[ml_vp$variable == "attitude_social_impact"] <- "Social changes are increasingly unfavorable to the elderly"
ml_vp$variable[ml_vp$variable == "community_services_available_hotline"] <- "Does your community provide elderly service hotline"
ml_vp$variable[ml_vp$variable == "internet_impact_leisure"] <- "How the internet technologies impact your leisure and entertainment"
ml_vp$variable[ml_vp$variable == "sleep_past_1month"] <- "How does your sleep in the past 1 month"
ml_vp$variable[ml_vp$variable == "internet_help_the_most"] <- "Who do you prefer to ask when encounter difficulties in using the internet"
ml_vp$variable[ml_vp$variable == "internet_impact_public_service"] <- "How the iternnet technologies affect your your payment for public services"
       
# Plot
plot(ml_vp %>% mutate(label = ""), max_vars = 10) +
  ggtitle("", "") +
  ylab("One minus F2 score loss after permutations")


