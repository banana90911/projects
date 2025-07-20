library(tidyverse)
library(caret)
library(xgboost)
library(randomForest)
library(rpart)

# load  datasets
train_red <- read.csv("../volume/data/raw/trainRed.csv")
train_white <- read.csv("../volume/data/raw/trainWhite.csv")
test_red <- read.csv("../volume/data/raw/testRed.csv")
test_white <- read.csv("../volume/data/raw/testWhite.csv")

str(train_red)
str(train_white)

# ensure quality is a factor 
train_red$quality <- as.factor(train_red$quality)
train_white$quality <- as.factor(train_white$quality)

# separate features and labels for red wine
features_red <- train_red %>% select(-quality)
labels_red <- train_red$quality

# separate features and labels for white wine
features_white <- train_white %>% select(-quality)
labels_white <- train_white$quality

# standardization
preprocess_params_red <- preProcess(features_red, method = c("center", "scale"))
features_red_scaled <- predict(preprocess_params_red, features_red)

preprocess_params_white <- preProcess(features_white, method = c("center", "scale"))
features_white_scaled <- predict(preprocess_params_white, features_white)

# convert features to matrix format for XGBoost
x_train_red <- as.matrix(features_red_scaled)
y_train_red <- as.numeric(as.character(labels_red))

x_train_white <- as.matrix(features_white_scaled)
y_train_white <- as.numeric(as.character(labels_white))

# split data
set.seed(9091)

# red wine data
train_index_red <- createDataPartition(y_train_red, p = 0.8, list = FALSE)
x_train_red_train <- x_train_red[train_index_red, ]
x_train_red_valid <- x_train_red[-train_index_red, ]
y_train_red_train <- y_train_red[train_index_red]
y_train_red_valid <- y_train_red[-train_index_red]

# white wine data
train_index_white <- createDataPartition(y_train_white, p = 0.8, list = FALSE)
x_train_white_train <- x_train_white[train_index_white, ]
x_train_white_valid <- x_train_white[-train_index_white, ]
y_train_white_train <- y_train_white[train_index_white]
y_train_white_valid <- y_train_white[-train_index_white]

# parameter grid
param_grid <- expand.grid(
  nrounds = seq(50, 200, by = 50),
  max_depth = seq(3, 9, by = 2),
  eta = c(0.01, 0.05, 0.1),
  gamma = c(0, 0.1, 0.2),
  colsample_bytree = seq(0.6, 1, by = 0.1),
  min_child_weight = c(1, 3, 5),
  subsample = seq(0.6, 1, by = 0.1)
)

# RMSE
rmse <- function(true, predicted) {
  sqrt(mean((true - predicted)^2))
}

# train and tune XGBoost model for red wine
xgb_grid_red <- train(
  x = x_train_red_train,
  y = y_train_red_train,
  method = "xgbTree",
  trControl = trainControl(method = "cv", number = 5, verboseIter = TRUE),
  tuneGrid = param_grid,
  metric = "RMSE"
)

# train and tune XGBoost model for white wine
xgb_grid_white <- train(
  x = x_train_white_train,
  y = y_train_white_train,
  method = "xgbTree",
  trControl = trainControl(method = "cv", number = 5, verboseIter = TRUE),
  tuneGrid = param_grid,
  metric = "RMSE"
)

# best hyperparameters
best_params_red <- xgb_grid_red$bestTune
best_params_white <- xgb_grid_white$bestTune

cat("Best parameters for red wine:\n")
print(best_params_red)
cat("Best parameters for white wine:\n")
print(best_params_white)

# retrain XGBoost for red wine
xgb_model_red <- xgboost(data = x_train_red_train, label = y_train_red_train, 
                         max_depth = best_params_red$max_depth,
                         eta = best_params_red$eta,
                         gamma = best_params_red$gamma,
                         colsample_bytree = best_params_red$colsample_bytree,
                         min_child_weight = best_params_red$min_child_weight,
                         subsample = best_params_red$subsample,
                         nrounds = best_params_red$nrounds,
                         objective = "reg:squarederror", eval_metric = "rmse", verbose = 0)

# retrain XGBoost for white wine
xgb_model_white <- xgboost(data = x_train_white_train, label = y_train_white_train, 
                           max_depth = best_params_white$max_depth,
                           eta = best_params_white$eta,
                           gamma = best_params_white$gamma,
                           colsample_bytree = best_params_white$colsample_bytree,
                           min_child_weight = best_params_white$min_child_weight,
                           subsample = best_params_white$subsample,
                           nrounds = best_params_white$nrounds,
                           objective = "reg:squarederror", eval_metric = "rmse", verbose = 0)

# calculate RMSE
pred_xgb_red_valid <- predict(xgb_model_red, x_train_red_valid)
pred_xgb_white_valid <- predict(xgb_model_white, x_train_white_valid)

rmse_red_valid <- rmse(y_train_red_valid, pred_xgb_red_valid)
rmse_white_valid <- rmse(y_train_white_valid, pred_xgb_white_valid)

cat("RMSE for red wine:", rmse_red_valid, "\n")
cat("RMSE for white wine:", rmse_white_valid, "\n")

# prepare test data
x_test_red <- as.matrix(predict(preprocess_params_red, test_red %>% select(-Id)))
x_test_white <- as.matrix(predict(preprocess_params_white, test_white %>% select(-Id)))

# prediction
pred_xgb_red <- predict(xgb_model_red, x_test_red)
pred_xgb_white <- predict(xgb_model_white, x_test_white)

# combine predictions
final_predictions_red <- data.frame(Id = test_red$Id, quality = round(pred_xgb_red))
final_predictions_white <- data.frame(Id = test_white$Id, quality = round(pred_xgb_white))

final_predictions <- rbind(final_predictions_red, final_predictions_white)

# save file
write.csv(final_predictions, "../volume/data/raw/submission.csv", row.names = FALSE)
