library(dplyr)
library(caret)
library(glmnet)

# load the dataset
data <- read.csv("../volume/data/raw/train.csv")
test <- read.csv("../volume/data/raw/test.csv")

# split dataset
set.seed(9091)
train_index <- createDataPartition(data$Signal, p = 0.8, list = FALSE)
train <- data[train_index, ]
validation <- data[-train_index, ]

# signal to factor
train$Signal <- factor(train$Signal, levels = c(0, 1), labels = c("No", "Yes"))
validation$Signal <- factor(validation$Signal, levels = c(0, 1), labels = c("No", "Yes"))

# standardization
preProcValues <- preProcess(train[, -which(names(train) == "Signal")], method = c("center", "scale"))
train_scaled <- predict(preProcValues, train)
validation_scaled <- predict(preProcValues, validation)
test_scaled <- predict(preProcValues, test)

# cross-validation
control <- trainControl(method = "cv", number = 10, classProbs = TRUE, summaryFunction = twoClassSummary)

# generalized linear regression (Lasso) hyperparameter tuning
glm_grid <- expand.grid(alpha = 1, lambda = seq(0.001, 1, length = 100))
lasso_tuned <- train(Signal ~ ., data = train_scaled, method = "glmnet", trControl = control, tuneGrid = glm_grid, metric = "ROC")

# ridge logistic regression hyperparameter tuning
ridge_grid <- expand.grid(alpha = 0, lambda = seq(0.001, 1, length = 100))
ridge_tuned <- train(Signal ~ ., data = train_scaled, method = "glmnet", trControl = control, tuneGrid = ridge_grid, metric = "ROC")

# evaluation
lasso_predictions <- predict(lasso_tuned, newdata = validation_scaled)
ridge_predictions <- predict(ridge_tuned, newdata = validation_scaled)

lasso_accuracy <- mean(lasso_predictions == validation_scaled$Signal)
ridge_accuracy <- mean(ridge_predictions == validation_scaled$Signal)
ensemble_probabilities <- (predict(lasso_tuned, newdata = validation_scaled, type = "prob")[,2] +
                             predict(ridge_tuned, newdata = validation_scaled, type = "prob")[,2]) / 2

ensemble_predictions <- ifelse(ensemble_probabilities > 0.5, "Yes", "No")
ensemble_accuracy <- mean(ensemble_predictions == validation_scaled$Signal)
print(paste("Ensemble Accuracy:", ensemble_accuracy))

print(paste( "Lasso Accuracy:", lasso_accuracy))
print(paste("Ridge Accuracy:", ridge_accuracy))



# final prediction
# full datasets
full_train_set <- bind_rows(train_scaled, validation_scaled)

# generalized linear regression (Lasso) hyperparameter tuning
lasso_full_model <- train(Signal ~ ., data = full_train_set, method = "glmnet", trControl = control, tuneGrid = glm_grid, metric = "ROC")

# prediction
test_probabilities <- predict(lasso_full_model, newdata = test_scaled, type = "prob")
test_predictions <- ifelse(test_probabilities[, 2] > 0.5, 1, 0)

# create a dataframe
final_results <- data.frame(ID = seq_len(length(test_predictions)), Signal = test_predictions)

# save file
write.csv(final_results, "/Users/siheonjung/Desktop/psu/summer 2024/stat380/4/Week04/project/volume/data/raw/final_predictions.csv", row.names = FALSE)
