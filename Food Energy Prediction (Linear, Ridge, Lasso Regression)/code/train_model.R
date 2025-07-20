library(readr)
library(tidyverse)
library(caret)
library(glmnet)
library(pls)
library(Metrics)

# load data
remove_bom <- function(file_path) {
  content <- readLines(file_path, encoding = "UTF-8")
  if (startsWith(content[1], "\ufeff")) {
    content[1] <- substring(content[1], 2)
  }
  writeLines(content, file_path)
}

remove_bom("../volume/data/raw/train.csv")
remove_bom("../volume/data/raw/test.csv")

read_clean_csv <- function(file_path) {
  lines <- readLines(file_path, warn = FALSE)
  lines <- iconv(lines, from = "UTF-8", to = "UTF-8", sub = "") 
  data <- read_csv(I(lines), locale = locale(encoding = "UTF-8"))
  return(data)
}

data <- read_clean_csv("../volume/data/raw/train.csv")
test <- read_clean_csv("../volume/data/raw/test.csv")

# remove the first column
data <- data %>% select(-1)
test <- test %>% select(-1)

# set Id as identifier
data <- data %>% column_to_rownames("Id")
test <- test %>% column_to_rownames("Id")

# replace null to average value if the variable is numeric
na_mean <- function(df) {
  df %>%
    mutate(across(where(is.numeric), ~ ifelse(is.na(.), mean(., na.rm = TRUE), .)))
}

data <- na_mean(data)
test <- na_mean(test)

# set target and predictor variables
target <- "Energ_Kcal"
predictors <- setdiff(names(data), target)

# split the dataset
set.seed(9091)
trainIndex <- createDataPartition(data[[target]], p = 0.8, list = FALSE, times = 1)
train <- data[trainIndex,]
validation <- data[-trainIndex,]

# normalization
preprocess_params <- preProcess(train[, predictors], method = c("center", "scale"))
train[, predictors] <- predict(preprocess_params, train[, predictors])
validation[, predictors] <- predict(preprocess_params, validation[, predictors])
test[, predictors] <- predict(preprocess_params, test[, predictors])

# linear regression
linear_model <- lm(as.formula(paste(target, "~ .")), data = train)
linear_pred <- predict(linear_model, validation)
linear_rmse <- rmse(validation[[target]], linear_pred)
print(linear_rmse)

# ridge regression
x_train <- model.matrix(as.formula(paste(target, "~ .")), train)[,-1]
y_train <- train[[target]]
x_val <- model.matrix(as.formula(paste(target, "~ .")), validation)[,-1]
y_val <- validation[[target]]
ridge_model <- cv.glmnet(x_train, y_train, alpha = 0)
ridge_pred <- predict(ridge_model, x_val, s = "lambda.min")
ridge_rmse <- rmse(y_val, ridge_pred)
print(ridge_rmse)

# lasso regression
lasso_model <- cv.glmnet(x_train, y_train, alpha = 1)
lasso_pred <- predict(lasso_model, x_val, s = "lambda.min")
lasso_rmse <- rmse(y_val, lasso_pred)
print(lasso_rmse)

# PCA
pca_model <- prcomp(x_train, scale. = TRUE)
pca_train <- as.data.frame(predict(pca_model, x_train))
pca_val <- as.data.frame(predict(pca_model, x_val))
pca_model <- lm(y_train ~ ., data = pca_train)
pca_pred <- predict(pca_model, pca_val)
pca_rmse <- rmse(y_val, pca_pred)
print(pca_rmse)

# PCR
pcr_model <- pcr(as.formula(paste(target, "~ .")), data = train, validation = "CV")
pcr_rmsep <- RMSEP(pcr_model)
if (is.matrix(pcr_rmsep$val)) {
  pcr_ncomp <- which.min(pcr_rmsep$val[1, ])
} else if (is.array(pcr_rmsep$val)) {
  pcr_ncomp <- which.min(pcr_rmsep$val[1, , 1]) 
} else {
}
pcr_pred <- predict(pcr_model, validation, ncomp = pcr_ncomp)
pcr_rmse <- rmse(validation[[target]], pcr_pred)
print(pcr_rmse)

# PLS
pls_model <- plsr(as.formula(paste(target, "~ .")), data = train, validation = "CV")
pls_rmsep <- RMSEP(pls_model)
pls_ncomp <- which.min(pls_rmsep$val[2, , 1])
pls_pred <- predict(pls_model, validation, ncomp = pls_ncomp)
pls_rmse <- rmse(validation[[target]], pls_pred)
print(pls_rmse)



# final prediction
full_train <- rbind(train, validation)

x_full_train <- model.matrix(as.formula(paste(target, "~ .")), full_train)[,-1]
y_full_train <- full_train[[target]]
x_test <- model.matrix(~ ., test)[,-1]

final_pca_model <- prcomp(x_full_train, scale. = TRUE)

final_pca_train <- as.data.frame(predict(final_pca_model, x_full_train))
final_pca_test <- as.data.frame(predict(final_pca_model, x_test))

final_lm_model <- lm(y_full_train ~ ., data = final_pca_train)

final_pred <- predict(final_lm_model, final_pca_test)

final_predictions <- data.frame(Id = rownames(test), Energ_Kcal = final_pred)
write.csv(final_predictions, file = "/Users/siheonjung/Desktop/psu/summer 2024/stat380/3/Week03/final_predictions2.csv", row.names = FALSE)

