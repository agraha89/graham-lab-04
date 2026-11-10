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
    Survived == 1 ~ "Survived")) |> # Giving informative labels to numerically-assigned categories
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

## Descriptive Exploration

descriptives <- titanic |>
  summarise(across(
    c(Age),
    list(mean = mean, sd = sd, min = min, max = max))) ## Exploring specific descriptive statistics for Age
  
table(titanic$Survived.Char)
table(titanic$Sex)
table(titanic$Pclass.Char)