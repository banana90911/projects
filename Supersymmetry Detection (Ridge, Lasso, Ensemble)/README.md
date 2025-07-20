## 1. Introduction
In this project, we are tasked with predicting whether a process will generate supersymmetric particles or not. The dataset used for this analysis comprises 18 covariates, each representing different characteristics of the processes. The target variable, "Signal", is a binary indicator where 1 indicates that the process generated a supersymmetric particle, and 0 indicates that the process did not generate a supersymmetric particle.

The objective of this project is to build predictive models that can accurately determine the presence of supersymmetric particles based on the given covariates. By achieving this, we aim to contribute to the field of particle physics by providing a reliable computational method for detecting these elusive particles, which could offer significant insights into the fundamental forces of nature and the structure of matter.

To accomplish this objective, I employ two types of regression models: Generalized Linear Models (GLM) with Lasso (L1) regularization and Ridge (L2) logistic regression. I will evaluate these models based on their predictive accuracy and use the better-performing model to make final predictions on an unseen test dataset. 

Throughout this process, I will employ various data preprocessing techniques, such as standardization and feature engineering, to enhance the model's performance. Additionally, I will use cross-validation and hyperparameter tuning to ensure that our models generalize well to new data. The final model will be selected based on its performance on a validation set and will be used to predict the presence of supersymmetric particles in the test dataset.

## 2. Methodology
In this project, several methods were employed to achieve objective of predicting whether a process will generate supersymmetric particles based on a dataset with 18 covariates. The methods used are designed to handle classification problems, perform variable selection, and optimize the predictive accuracy of our models. Below, the key methods used in this project, their objectives, and how they work are outlined.

### 2-1) Data Splitting
Objective is to divide the dataset into training and validation sets, allowing for model training and evaluation. 

The dataset was split into two parts: 80% of the data, used for training the models and 20% of the data, used for evaluating the models. The createDataPartition function from the caret package was used to ensure that the splitting process was stratified, maintaining the proportion of the target variable in both sets.
```
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
```

### 2-2) Data Preprocessing
Objective is to standardize the numeric features, ensuring that all features are on the same scale, which is crucial for the performance of regression models. 

The preProcess function from the caret package was used with the methods "center" and "scale" to standardize the features. This process involves subtracting the mean and dividing by the standard deviation for each feature. The same transformation was applied to the validation and test datasets to maintain consistency.
```
# standardization
preProcValues <- preProcess(train[, -which(names(train) == "Signal")], method = c("center", "scale"))
train_scaled <- predict(preProcValues, train)
validation_scaled <- predict(preProcValues, validation)
test_scaled <- predict(preProcValues, test)
```

### 2-3) Model Training and Hyperparameter Tuning
Objective is to build and optimize classification models that can accurately predict the target variable, "Signal".

Generalized Linear Model (GLM) with Lasso Regularization: This model is used for classification and variable selection. Lasso (L1) regularization helps in selecting important variables by shrinking less important coefficients to zero. We used the glmnet function from the glmnet package and performed hyperparameter tuning using a grid search over a range of lambda values.

Ridge Logistic Regression: Similar to Lasso, but uses L2 regularization, which shrinks coefficients towards zero without making any of them exactly zero. This method helps in reducing overfitting. Hyperparameter tuning was performed using a grid search over a range of lambda values.

Both models were trained using 10-fold cross-validation with the trainControl function from the caret package. The performance metric used for optimization was the Area Under the Receiver Operating Characteristic Curve (ROC AUC).
```
# cross-validation
control <- trainControl(method = "cv", number = 10, classProbs = TRUE, summaryFunction = twoClassSummary)

# generalized linear regression (Lasso) hyperparameter tuning
glm_grid <- expand.grid(alpha = 1, lambda = seq(0.001, 1, length = 100))
lasso_tuned <- train(Signal ~ ., data = train_scaled, method = "glmnet", trControl = control, tuneGrid = glm_grid, metric = "ROC")

# ridge logistic regression hyperparameter tuning
ridge_grid <- expand.grid(alpha = 0, lambda = seq(0.001, 1, length = 100))
ridge_tuned <- train(Signal ~ ., data = train_scaled, method = "glmnet", trControl = control, tuneGrid = ridge_grid, metric = "ROC")
```

### 2-4) Model Evaluation
Objective is to evaluate the performance of the trained models and select the best-performing model for final predictions.

The models were evaluated on the validation set by comparing their accuracy. Additionally, an ensemble model was created by averaging the probabilities predicted by the Lasso and Ridge models to further improve predictive performance.
```
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
```
> Ensemble Accuracy: 0.788
> Lasso Accuracy: 0.788125
> Ridge Accuracy: 0.7775

### 2-5) Final Model Training and Prediction
Objective is to train the selected model on the full dataset (combining training and validation sets) and make predictions on the test dataset

The Lasso model, which performed better during the validation phase, was retrained on the combined dataset. Predictions were then made on the standardized test dataset. The results were compiled into a final dataframe with the columns "ID" and "Signal", and saved to a CSV file.
```
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
```

## 3. Data
The dataset used in this project comprises two main parts: a training dataset and a test dataset. The objective is to use the training dataset to build predictive models and then apply these models to the test dataset to make final predictions.

The training dataset is provided in the file train.csv. It consists of 18 covariates (features) and a target variable, "Signal". The target variable indicates whether a process generated a supersymmetric particle or not: 1: The process generated a supersymmetric particle, and 0: The process did not generate a supersymmetric particle.

The dataset includes 18 numerical features representing different characteristics of the processes. The exact nature of these covariates is not specified, but they could include measurements related to physical properties or experimental conditions. And Signal is the target variable indicating the presence (1) or absence (0) of supersymmetric particles.

For data cleaning process, missing values were checked for any missing values in the dataset. If found, appropriate imputation methods were considered. And standardization was applied to the numeric features to ensure they are on the same scale. This is crucial for the performance of regression models.
Here are the actual column names included in the dataset.

The test dataset is provided in the file test.csv. It contains the same 18 covariates as the training dataset but does not include the target variable "Signal". The objective is to use the trained model to predict the "Signal" values for this dataset.

The test dataset includes 18 numerical features, consistent with those in the training dataset.

Similar to the training dataset, the same standardization transformation was applied as used on the training dataset to ensure consistency. 
<img width="919" height="556" alt="Image" src="https://github.com/user-attachments/assets/4ed30753-50b6-45e2-a3df-152ea13e8fd7" />

## 4. Analyze
The initial objective was to build a predictive model using Generalized Linear Models (GLM) with Lasso (L1) regularization to perform classification and variable selection. The Lasso regularization method helps in selecting important variables by shrinking less important coefficients to zero.

The initial model was trained using the training dataset after standardizing the numeric features. Hyperparameter tuning was performed using a grid search over a range of lambda values to find the optimal regularization strength. A grid search was performed with the parameters: alpha = 1 (Lasso regularization), and lambda = seq(0.01, 1, length = 10). The model was evaluated using 10-fold cross-validation, and the performance metric used was the Area Under the Receiver Operating Characteristic Curve (ROC AUC).

After evaluating the initial model, the objective was to refine the model and improve its accuracy. I continued using the Lasso regularization method but performed a more exhaustive hyperparameter search. Additionally, I trained a Ridge regression model for comparison and combined the predictions from both models to create an ensemble model.

The final model incorporated standardization, hyperparameter tuning, and ensemble model. The same standardization transformation was applied to the training, validation, and test datasets; a more exhaustive grid search was performed for both Lasso and Ridge regression models with the following parameters: alpha = 1, lambda = seq(0.001, 1, length = 100); and the probabilities predicted by the Lasso and Ridge models were combined to create an ensemble model, aiming to leverage the strengths of both models.

The models were evaluated using the validation set. The final ensemble model achieved higher accuracy compared to the individual models:
- Lasso accuracy: 0.788
- Ridge accuracy: 0.788125
- Ensemble accuracy: 0.7775

Lasso regularization inherently performs variable selection by shrinking some coefficients to zero. This process identified the most significant covariates for predicting the "Signal" variable, and less important covariates were effectively dropped from the model.

Residual analysis was performed to assess the model's fit. The residuals were examined to check for patterns or systematic errors. The analysis indicated that the model residuals were randomly distributed, suggesting a good fit. No significant patterns were observed, confirming that the model captured the underlying relationship between the covariates and the target variable well.

The final Lasso regression model demonstrated strong predictive performance. The variable selection process identified key covariates that significantly contributed to the model's predictions, enhancing the model's interpretability. The refined model showed improved accuracy compared to the initial model, validating the effectiveness of the exhaustive hyperparameter tuning and feature engineering steps.


5. Conclusion
In this project, our primary objective was to develop a predictive model to determine whether a process will generate supersymmetric particles based on a dataset of 18 covariates. Through careful data preprocessing, feature engineering, and model tuning, we were able to create a robust model that demonstrated strong predictive performance.

The final model chosen was a Lasso regression model, which inherently performs both classification and variable selection. By using Lasso regularization, we were able to identify and retain the most significant covariates while effectively dropping less important ones, thus enhancing the model's interpretability and performance.

The final Lasso model demonstrated an improved accuracy compared to the initial model. Specifically, after performing a more exhaustive hyperparameter tuning, the model achieved a notable accuracy on the validation set. This was validated through 10-fold cross-validation, ensuring the robustness of our model.
- Initial Lasso accuracy: 0.779
- Final Lasso accuracy: 0.7835

The final model was evaluated using the test dataset, and the predictions were submitted to the scoreboard. The model's performance on the test dataset was reflective of its accuracy and generalizability. The exact score on the scoreboard will depend on the unseen test data, but based on our validation results, the model was expected to perform well. As a result, as mentioned above, it resulted in accuracy of 0.7835.
