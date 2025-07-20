library(readr)
library(caret)
library(tidyverse)
library(leaps)
library(MASS)
library(splines)
library(mgcv)

# load data
data <- read_csv("/Users/siheonjung/Desktop/psu/summer 2024/stat380/2/data/Stat_380_train.csv")
test <- read_csv("/Users/siheonjung/Desktop/psu/summer 2024/stat380/2/data/Stat_380_test.csv")

# replace null to average value if the variable is numeric
na_mean <- function(df) {
  df %>%
    mutate(across(where(is.numeric), ~ ifelse(is.na(.), mean(., na.rm = TRUE), .)))
}

data <- na_mean(data)
test <- na_mean(test)

# Convert categorical variables to factors
data <- data %>% mutate(across(where(is.character), as.factor))
test <- test %>% mutate(across(where(is.character), as.factor))

# Log-transform SalePrice
data$SalePrice <- log(data$SalePrice)

# normalization
normalize <- function(df, excol) {
  numeric_columns <- colnames(df)[sapply(df, is.numeric) & !colnames(df) %in% excol]
  df[, numeric_columns] <- scale(df[, numeric_columns])
  df
}

norm_data <- normalize(data, excol = "Id")
norm_test <- normalize(test, excol = "Id")

# RMSE function
calc_rmse <- function(actual, predicted) {
  sqrt(mean((actual - predicted)^2))
}

# split data
set.seed(9091)
trainIndex <- createDataPartition(norm_data$SalePrice, p = 0.8, list = FALSE, times = 1)
train <- norm_data[trainIndex, ] # train dataset
validation <- norm_data[-trainIndex, ] # validation dataset

# multiple linear regression
model_lm <- lm(SalePrice ~ ., data = train)
predictions_lm <- predict(model_lm, newdata = validation)
rmse_lm <- calc_rmse(validation$SalePrice, predictions_lm)
print(paste("rmse_lm: ", rmse_lm)) # 0.593680939406237

# best subsets regression
subsets <- regsubsets(SalePrice ~ ., data = train, nvmax = 13)
summary <- summary(subsets)
vars <- names(coef(subsets, which.max(summary$adjr2)))[-1]
real_vars <- vars[vars %in% colnames(train)]
model_bsr <- lm(as.formula(paste("SalePrice ~", paste(real_vars, collapse = " + "))), data = train)
predictions_bsr <- predict(model_bsr, newdata = validation)
rmse_bsr <- calc_rmse(validation$SalePrice, predictions_bsr)
print(paste("rmse_bsr: ", rmse_bsr)) # 0.59504310857517

# step regression
model_sr <- stepAIC(lm(SalePrice ~ ., data = train), direction = "both", trace = FALSE)
predictions_sr <- predict(model_sr, newdata = validation)
rmse_sr <- calc_rmse(validation$SalePrice, predictions_sr)
print(paste("rmse_sr: ", rmse_sr)) # 0.593581576851409

# regression splines
numeric <- names(train)[sapply(train, is.numeric)]
numeric <- setdiff(numeric, c("SalePrice", "Id"))
categorical <- names(train)[sapply(train, is.factor)]
spline_terms <- paste0("bs(", numeric, ", degree = 5)")
formula <- as.formula(paste("SalePrice ~", paste(c(spline_terms, categorical), collapse = " + ")))
model_rs <- lm(formula, data = train)
predictions_rs <- predict(model_rs, newdata = validation)
rmse_rs <- calc_rmse(validation$SalePrice, predictions_rs)
print(paste("rmse_rs: ", rmse_rs)) # 0.551726329701176

# smoothing splines
formula <- as.formula(paste("SalePrice ~", paste(colnames(train)[!colnames(train) %in% c("SalePrice", "Id")], collapse = " + ")))
model_ss <- gam(formula, data = train, family = gaussian(), method = "REML")
predictions_ss <- predict(model_ss, newdata = validation)
rmse_ss <- calc_rmse(validation$SalePrice, predictions_ss)
print(paste("rmse_ss: ", rmse_ss)) # 0.593721957145594


# denormalization
saleprice_mean <- mean(data$SalePrice)
saleprice_sd <- sd(data$SalePrice)
denorm_predictions <- exp(predictions * saleprice_sd + saleprice_mean)


# prediction using test dataset
final_predictions <- predict(model_rs, newdata = norm_test)

# denormalization
denorm_final_predictions <- exp(final_predictions * saleprice_sd + saleprice_mean)

# save file
write_csv(data.frame(Id = test_id, SalePrice = denorm_final_predictions), "/Users/siheonjung/Desktop/psu/summer 2024/stat380/2/data/final_predictions.csv")
