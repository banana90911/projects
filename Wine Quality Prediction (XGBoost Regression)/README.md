## 1. Introduction
The dataset consists of two primary datasets for red and white wines, each containing a set of features that represent the physicochemical properties of the wines. These features include measurements such as fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, sulphates, and alcohol content. The target variable, quality, is a categorical score that ranges from 0 to 10, indicating the perceived quality of the wine as assessed by wine experts. The test datasets do not contain the quality column, which serves as the target variable for prediction. Red wine dataset contains records of various red wine samples with features representing their chemical properties, and white wine dataset contains records of white wine samples with similar features as those of the red wine dataset.

The primary objective of this project is to develop predictive models that can accurately assess wine quality based on the given chemical properties. This involves training machine learning models using the training datasets and evaluating their performance on the validation datasets. The models developed in this project aim to minimize the Root Mean Squared Error (RMSE) between predicted and actual quality scores, thereby ensuring high predictive accuracy.

To achieve these goals, XGBoost, which is known for its robust performance in regression tasks, was used. Dara preprocessing techniques, including standardization, were applied to enhance the performance of these models. Hyperparameter tuning is conducted to optimize model parameters, further improving prediction accuracy.

## 2. Methodology
### 2-1) Data Preprocessing
**Objective:** Prepare the dataset for analysis by ensuring that the features are appropriately formatted and the data is clean.

**Description:** The datasets for red and white wines were loaded and examined to understand their structure and characteristics. The target variable, quality, was converted into a factor for classification purposes. 
```
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
```

### 2-2) Standardization
**Objective:** Normalize the features to have a mean of 0 and a standard deviation of 1, ensuring that all features contribute equally to the model's performance.

**Description:** Standardization was applied to all numeric features using z-score normalization. 
```
# standardization
preprocess_params_red <- preProcess(features_red, method = c("center", "scale"))
features_red_scaled <- predict(preprocess_params_red, features_red)

preprocess_params_white <- preProcess(features_white, method = c("center", "scale"))
features_white_scaled <- predict(preprocess_params_white, features_white)
```

### 2-3) Model Tuning and Tuning with XGBoost
**Objective:** Develop robust predictive models for wine quality using the XGBoost algorithm, which is renowned for its high performance in regression and classification tasks.

**Description:** XGBoost (Extreme Gradient Boosting) is an advanced ensemble learning algorithm that combines the predictions of multiple weak learners (decision trees) to produce a strong learner. 

**Training:** The model was trained on 80% of the data, with the remaining 20% used for validation. The training process involved fitting the model to the scaled features and quality labels, optimizing the internal parameters to minimize prediction error.
- Tuning: A comprehensive hyperparameter tuning process was conducted using grid search to identify the optimal combination of parameters that minimize RMSE. Parameters such as nrounds, max_depth, eta, gamma, colsample_bytree, min_child_weight, and subsample were explored.
```
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
```

```
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
```

```
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
```

### 2-4) Model Evaluation
**Objective:** Assess the performance of the predictive models to ensure accuracy and reliability in predicting wine quality.

**Description:** Model evaluation was performed using the Root Mean Squared Error (RMSE) metric, which measures the average magnitude of the error between predicted and actual values. Evaluation was conducted on the validation set, allowing for an unbiased assessment of the model's ability to generalize to unseen data.
```
# RMSE
rmse <- function(true, predicted) {
  sqrt(mean((true - predicted)^2))
}

# calculate RMSE
pred_xgb_red_valid <- predict(xgb_model_red, x_train_red_valid)
pred_xgb_white_valid <- predict(xgb_model_white, x_train_white_valid)

rmse_red_valid <- rmse(y_train_red_valid, pred_xgb_red_valid)
rmse_white_valid <- rmse(y_train_white_valid, pred_xgb_white_valid)

cat("RMSE for red wine:", rmse_red_valid, "\n")
cat("RMSE for white wine:", rmse_white_valid, "\n")\
```
> RMSE for red wine: 0.5473912
> RMSE for white wine: 0.6201782

### 2-5) Prediction
**Objective:** Generate predictions for the wine quality scores of the test dataset, leveraging the trained and optimized models.

**Description:** After training and tuning the models, predictions were made on the standardized test dataset, which lacked quality scores. The predictions were rounded to the nearest integer to match the format of the quality scores in the training dataset.
```
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
```

## 3. Data
The dataset used in this project consists of physicochemical properties of red and white wines, along with quality ratings. The data is sourced from the wine industry, where these chemical measurements are used to assess the quality and characteristics of wine. This section provides an overview of the dataset, the variables included, and any data cleaning processes applied.

The dataset is divided into two separate files: one for red wines and one for white wines. Each dataset comprises various chemical properties that influence wine quality, alongside a quality score assigned by human tasters. The datasets are structured in a tabular format with each row representing a unique wine sample and each column representing a specific feature or attribute.

The datasets contain the following variables:
1. Fixed Acidity: The concentration of fixed acids (e.g., tartaric) that do not evaporate easily.
2. Volatile Acidity: The amount of volatile acids (e.g., acetic acid) that evaporate and contribute to a vinegar-like taste.
3. Citric Acid: A weak organic acid present in wines, which can add freshness and flavor.
4. Residual Sugar: The amount of sugar remaining after fermentation; influences sweetness.
5. Chlorides: The concentration of chloride ions, contributing to the saltiness.
6. Free Sulfur Dioxide: The amount of SO2 that is free and available to protect the wine from oxidation
7. Total Sulfur Dioxide: The total amount of SO2 present, including both bound and free forms.
8. Density: The density of the wine, which can be related to alcohol and sugar content.
9. pH: A measure of acidity or alkalinity.
10. Sulphates: Salts of sulfuric acid that can contribute to the wine's aroma and flavor.
11. Alcohol: The alcohol content by volume.
12. Quality: The quality score ranging from 0 to 10, as determined by wine experts.

For data cleaning, the datasets were examined for missing values, and the target variable, quality, was converted into a factor variable for classification purposes. Also, all numeric features were standardized to have a mean of 0 and a standard deviation of 1. The data was checked for outliers that could potentially skew the analysis. Given the nature of the data and domain expertise, no extreme outliers were found to warrant exclusion.
<img width="940" height="698" alt="Image" src="https://github.com/user-attachments/assets/bd824e00-cd48-4578-9250-f93afdee28d0" />

## 4. Analyze
### 4-1) Initial Model
**Objective:** Develop a baseline model to predict wine quality and identify potential areas for improvement.

**Description:** The initial model was built using a decision tree algorithm. 

**Process:** 
- Data preparation: The data was preprocessed to handle categorical and continuous variables appropriately. The target variable quality was set as a factor for classification purposes.
- Model training: A decision tree model was trained on the entire dataset without any feature selection or transformation. The goal was to establish a baseline performance metric (RMSE).

**Results:** The initial decision tree model provided a basic understanding of feature importance but lacked accuracy due to its simplicity and tendency to overfit the training data.

### 4-2) Variable Selection
**Objective:** Identify and retain the most predictive features to enhance model accuracy and reduce complexity.

**Description:** Feature selection was performed to refine the model by identifying which variables contribute the most to predicting wine quality. 

**Process:** 
- Feature importance analysis: The initial decision tree model and feature importance scores from Random Forests were used to identify key predictors. Features such as volatile acidity, alcohol content, and sulphates were found to be strong indicators of wine quality.
- Dropped variables: Features with low importance scores, such as citric acid and chlorides, were considered for exclusion in the final model. However, due to the small number of features and the risk of losing potentially useful information, all features were retained for the final model to avoid any unintended loss of predictive power.

### 4-3) Final Model
**Objective:** Develop a robust predictive model using XGBoost to accurately predict wine quality.

**Description:** The final model was constructed using the XGBoost algorithm.

**Process:** 
- Hyperparameter tuning: A grid search was conducted to optimize hyperparameters such as max_depth, eta (learning rate), nrounds (number of boosting rounds), subsample, and colsample_bytree. This tuning aimed to minimize the RMSE and enhance model accuracy.
- Model training: The model was trained on the standardized training data, using the best hyperparameters identified during tuning. The training data was split into 80% training and 20% validation to ensure the model generalized well to unseen data.

**Results:** The final model showed a significant improvement in prediction accuracy over the initial model. The tuned XGBoost model achieved lower RMSE on the validation dataset.

### 4-4) Residual Analysis
Objective: Evaluate model predictions by analyzing residuals to identify potential areas for improvement.
Description: Residual analysis involves examining the differences between predicted and actual values to detect patterns or biases that the model may have missed.
Results: The absence of patterns in the residual plots confirmed the adequacy of the final model.


## 5. Conclusion
In this project, predictive models were successfully developed to assess wine quality using physicochemical properties and advanced machine learning techniques. By leveraging data preprocessing, feature engineering, and the powerful XGBoost algorithm, models were achieved with low Root Mean Squared Error (RMSE) scores for both red and white wines. The final models demonstrated superior performance over baseline models, accurately capturing the complex relationships between chemical properties and quality scores.

The models performed well on the scoreboard (0.64717), indicating competitive accuracy relative to other submissions. Key features such as volatile acidity, alcohol content, and sulphates were identified as significant predictors of wine quality, providing valuable insights for wine producers. Residual analysis confirmed the models' reliability, showing no systematic biases in predictions.
