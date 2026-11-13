library(glmmTMB)
library(dplyr)
library(tidyr)
library(ggplot2)
library(broom.mixed)
library(performance)
library(gt)
library(readr)
library(DHARMa)
library(ggeffects)
library(titanic)
library(psych)
library(tibble)

## Pulling built-in R dataset
titanic <- titanic_train

titanic <- titanic |>
  select(Survived, Sex, Age, Pclass) |> ## Selecting variables of interest
  drop_na() |>
  mutate(Sex.Binary = case_when(
    Sex == "male" ~ 0,
    Sex == "female" ~ 1)) |> ## Giving numerical values to sex categories
  mutate(Survived.Char = case_when(
    Survived == 0 ~ "Died",
    Survived == 1 ~ "Survived")) |> ## Giving informative labels to numerically-assigned categories
  mutate(Pclass.Char = case_when(
    Pclass == "1" ~ "First Class",
    Pclass == "2" ~ "Second Class",
    Pclass == "3" ~ "Third Class")) |>
  mutate(Pclass = factor(Pclass, levels = c(1, 2, 3))) |> ## Re-classifying integer vector as factor
  select(Survived, Survived.Char, Sex, Sex.Binary, Pclass, Pclass.Char, Age) ## Expanding dataset to include newly created vectors

n <- nrow(titanic)

## Exploring the Data

summary.table <- summary(titanic) ## Looking at basic descriptives

head(titanic) ## Looking at the first few rows of the dataset
str(titanic) ## Looking at the structure of the dataset

descriptives <- titanic |>
  summarise(across(
    c(Age),
    list(mean = mean, sd = sd, min = min, max = max))) ## Exploring specific descriptive statistics for Age
  
table(titanic$Survived.Char) ## Exploring counts of the categorical variables' factors
table(titanic$Sex)
table(titanic$Pclass.Char)

## Cross-tabbing categorical predictors by outcome to see whether sub-samples are particularly low for combinations of predictor factors with outcomes
cross_tab_sex <- table(titanic$Sex, titanic$Survived.Char) |> 
  as.data.frame.matrix() |> 
  rownames_to_column(var = "Sex") |> 
  gt() |> 
  tab_header(title = "Survival by Sex")

cross_tab_class <- table(titanic$Pclass.Char, titanic$Survived.Char) |> 
  as.data.frame.matrix() |> 
  rownames_to_column(var = "Passenger Class") |> 
  gt() |> 
  tab_header(title = "Survival by Passenger Class")

##Specifying Models
mod0 <- glmmTMB(Survived ~ 1, 
                data = titanic,
                family = binomial(link = "logit")) ## Intercept-only

mod1 <- glmmTMB(Survived ~ Sex.Binary,
                data = titanic,
                family = binomial(link = "logit")) ## Evaluate sex

mod2 <- glmmTMB(Survived ~ Sex.Binary + Pclass,
                data = titanic,
                family = binomial(link = "logit")) ## Evaluate the additional contribution of ticket class

mod3 <- glmmTMB(Survived ~ Sex.Binary + Pclass + Age,
                data = titanic,
                family = binomial(link = "logit")) ## Evaluate the additional contribution of age

summary(mod0) ## Results for intercept-only model
summary(mod1) ## Results for Model 1
summary(mod2) ## Results for Model 2
summary(mod3) ## Results for Model 3

confint(mod3) ## 95% confidence intervals for Model 3

## Nested Model Comparisons
tests <- anova(mod0, mod1, mod2, mod3) ## Comparing models

tests <- tests |>
  select(Chisq, `Chi Df`, `Pr(>Chisq)`) |>
  rename(Nested.df = `Chi Df`,
    Nested.Chisq.stat = Chisq,
    Nested.p.value = `Pr(>Chisq)`) ## Retaining and renaming relevant statistics

glance0 <- broom.mixed::glance(mod0) ## Extract overall model performance for intercept-only model
glance1 <- broom.mixed::glance(mod1) ## Extract overall model performance for Model 1
glance2 <- broom.mixed::glance(mod2) ## Extract overall model performance for Model 2
glance3 <- broom.mixed::glance(mod3) ## Extract overall model performance for Model 3

mods <- c("mod0: Null",
  "mod1: Sex",
  "mod2: + Ticket Class",
  "mod3: + Age") ## Creating and combining labels of the four models

glances <- rbind(glance0,
  glance1,
  glance2,
  glance3) ## Combining the individual model results

comparison_table <- cbind(Model = mods,
  glances,
  tests) ## Combining the model labels with the two statistical outputs

comparison_table |> ## Producing a combined model comparison table with relevant statistics and formatting
  select(Model,
    AIC,
    BIC,
    logLik,
    deviance,
    df.residual,
    Nested.df,
    Nested.Chisq.stat,
    Nested.p.value) |> ## Selecting relevant stats
  gt() |> ## Formatted as gt table
  cols_label(Model = "Model",
    AIC = "AIC",
    BIC = "BIC",
    logLik = "Log-likelihood",
    deviance = "Deviance",
    df.residual = "Residual.df",
    Nested.df = "Nested df",
    Nested.Chisq.stat = "Nested χ²",
    Nested.p.value = "Nested p") |> ## Providing informative labels for columns
  fmt_number(columns = c(
      AIC,
      BIC,
      logLik,
      deviance,
      Nested.Chisq.stat),
    decimals = 3) |> ## Specifying values at three decimal places
  fmt(columns = Nested.p.value,
    fns = function(x) {
      ifelse(is.na(x),
        "—",
        ifelse(x < .001, "< .001", sprintf("%.3f", x)))}) |> ## Formatting display of p values
  sub_missing(columns = c(Nested.df, Nested.Chisq.stat),
    missing_text = "—") ## Formatting display for missing values

## Evaluating binary logistic regression diagnostics
performance_comparison <- compare_performance(
  mod0,
  mod1,
  mod2,
  mod3,
  rank = TRUE) ## Providing performance-score for each model, alongside other metrics

plot(performance_comparison) ## Visual depiction of the above metrics

selected_row <- comparison_table |> ## Filtering to select most complex model with nested p-value < .05
  filter(Nested.p.value < .05) |>
  slice_tail(n = 1)

selected_name <- selected_row$Model ## Creating an object with the selected model's name for dynamic reporting

selected_model <- switch(selected_name,
  "mod0: Null" = mod0,
  "mod1: Sex" = mod1,
  "mod2: + Ticket Class" = mod2,
  "mod3: + Age" = mod3) ## Case matching for dynamic reporting

## Evaluating binary logistic regression diagnostics

set.seed(9041) ## Selecting the course number as the arbitrary seed for simulation of residuals
res_mod3 <- simulateResiduals(mod3, n = 500, plot = FALSE)
plot(res_mod3) #Plotting the above simulation to create a default DHARMa plot

plot(res_mod3) #Plotting the above simulation to create a default DHARMa plot

check_collinearity(mod3) ## Checking for collinearity of predictors
vif_plot <- plot(check_collinearity(mod3)) ## Plotting the VIF values calculated above

testOutliers(res_mod3) ## Testing for outliers

fitted_probs <- predict(mod3, type = "response") ## Checking for indicators of separation (i.e., fitted probabilities off essentially 0 or 1)

summary(fitted_probs) ## Generating summary table with minimum and maximum probability values

n_event <- sum(titanic$Survived == 1) ## Generating number of survivors
n_nonevent <- sum(titanic$Survived == 0) ## Generating number of non-survivors
n_coef <- length(fixef(mod3)$cond) ## Generating number of coefficients in the model

## Reporting results
  ## Coefficient Table
coefficient_results <- broom.mixed::tidy(selected_model,
  effects = "fixed",
  component = "cond",
  conf.int = TRUE,
  conf.level = 0.95) ## Extracting coefficient-level results from the fitted binary logistic regression model

coefficient_results |>
  select(term, ## Selecting relevant values
    estimate,
    std.error,
    conf.low,
    conf.high,
    p.value) |>
  gt() |> ## Formatting as gt table
  cols_label(term = "Predictor",
    estimate = "Estimate",
    std.error = "SE",
    conf.low = "95% CI lower",
    conf.high = "95% CI upper",
    p.value = "p") |> ## Renaming columns
  fmt_number(columns = c(estimate,
      std.error,
      conf.low,
      conf.high),
    decimals = 2) |> ## Formatting values to two decimal places

  ## Model Performance Table
selected_row |>
  select(Model,
    AIC,
    BIC,
    logLik,
    deviance,
    df.residual,
    Nested.df,
    Nested.Chisq.stat,
    Nested.p.value) |> ## Selecting relevant statistics
  gt() |> ## Formatting as gt table
  cols_label(Model = "Model",
    AIC = "AIC",
    BIC = "BIC",
    logLik = "Log-likelihood",
    deviance = "Deviance",
    df.residual = "Residual df",
    Nested.df = "Nested df",
    Nested.Chisq.stat = "Nested χ²",
    Nested.p.value = "Nested p") |> ## Adding informative column labels
  fmt_number(columns = c(AIC,
                         BIC,
                         logLik,
                         deviance,
                         df.residual,
                         Nested.df,
                         Nested.Chisq.stat,
                         Nested.p.value),
                decimals = 2) |> ## Formatting numbers to two decimal places
  fmt(columns = Nested.p.value,
    fns = function(x) {
      ifelse(x < .001,
        "< .001",
        sprintf("%.3f", x))}) ## Formatting p-values

fixef(mod3) ## Extracting fixed-effect coefficient estimates for each predictor
round(fixef(mod3)$cond[["Sex.Binary"]], 2)
round(fixef(mod3)$cond[["Pclass2"]], 2)
round(fixef(mod3)$cond[["Pclass3"]], 2)
round(fixef(mod3)$cond[["Age"]], 2)

exp(fixef(mod3)$cond) ## Exponentiating odds ratios
or_ci <- exp(confint(mod3, parm = "beta_")) ## ... and their 95% confidence intervals
or_ci

pred_sex <- ggpredict(mod3, terms = "Sex.Binary [0:1 by=0.05]")

ggplot(pred_sex, aes(x, predicted)) +
  geom_ribbon(aes(ymin = conf.low, ymax = conf.high), alpha = 0.18) +
  geom_line(linewidth = 1.2) +
  labs(x = "Passenger Sex",
    y = "Predicted Probability of Surviving the Titanic Sinking") +
  theme_classic()

pred_age <- ggpredict(mod3, terms = "Age")

ggplot(pred_age, aes(x, predicted)) +
  geom_ribbon(aes(ymin = conf.low, ymax = conf.high), alpha = 0.18) +
  geom_line(linewidth = 1.2) +
  labs(x = "Passenger Age",
       y = "Predicted Probability of Surviving the Titanic Sinking") +
  theme_classic()