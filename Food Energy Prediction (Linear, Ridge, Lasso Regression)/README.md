### 1. Introduction
This project focuses on the prediction of the energy content, measured in kilocalories (Kcal), of a diverse array of foods using several predictive modeling techniques.
Our dataset is split into two segments: a training set and a test set. The training dataset has been further divided into a training subset and a validation subset to facilitate a rigorous evaluation of our predictive models. This setup enables the assessment of model performance on unseen data, ensuring the robustness and generalizability of our predictions.
For this project, we have explored several regression techniques, including Linear Regression, Lasso Regression, Principal Component Analysis (PCA), Principal Component Regression (PCR), and Partial Least Squares (PLS). These models were evaluated based on their Root Mean Square Error (RMSE) on the validation dataset, a metric chosen for its sensitivity to the accuracy of predicted values.
Among the models tested, PCA demonstrated superior performance, showcasing the lowest RMSE. This indicates that PCA, with its capability to reduce dimensionality while retaining most of the variance, is particularly effective at capturing the underlying patterns that are most predictive of the energy content in foods. This informed our decision to select PCA for our final predictions.

### 2. Methodology
This project employs a comprehensive set of statistical and machine learning methods to predict the energy content, in kilocalories (Kcal), of various foods based on their nutritional properties. The methodologies used include Linear Regression, Lasso Regression, Ridge Regression, Principal Component Analysis (PCA), Principal Component Regression (PCR), and Partial Least Squares (PLS) regression.

## 2-1) Data Preprocessing
Data preparation involved reading and cleaning the datasets to handle any encoding issues or presence of Byte Order Marks (BOM). 
```
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
```

Missing values in numeric variables were replaced with the mean of the respective variable to maintain dataset integrity. 
```
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
```

Furthermore, data normalization was applied to standardize the range of independent variables, facilitating better model performance.
```
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
```

## 2-2) Linear Regression
Linear Regression aims to predict a continuous target variable based on one or more predictor variables. The model assumes a linear relationship between the predictors and the target. This simple yet powerful technique provides a baseline for comparison with more complex models.
```
# linear regression
linear_model <- lm(as.formula(paste(target, "~ .")), data = train)
linear_pred <- predict(linear_model, validation)
linear_rmse <- rmse(validation[[target]], linear_pred)
print(linear_rmse)
```

## 2-3) Ridge Regression
Ridge Regression is used to address multicollinearity in linear regression models, which can lead to unstable estimates of the regression coefficients. It introduces a penalty term (L2 regularization) to the loss function, shrinking the coefficients and reducing model complexity.
```
# ridge regression
x_train <- model.matrix(as.formula(paste(target, "~ .")), train)[,-1]
y_train <- train[[target]]
x_val <- model.matrix(as.formula(paste(target, "~ .")), validation)[,-1]
y_val <- validation[[target]]
ridge_model <- cv.glmnet(x_train, y_train, alpha = 0)
ridge_pred <- predict(ridge_model, x_val, s = "lambda.min")
ridge_rmse <- rmse(y_val, ridge_pred)
print(ridge_rmse)
```

## 2-4) Lasso Regression
Lasso Regression, similar to Ridge, modifies Linear Regression by adding an L1 penalty to the loss function, which encourages sparsity in the model coefficients. This method effectively performs variable selection, helping in identifying the most significant variables.
```
# lasso regression
lasso_model <- cv.glmnet(x_train, y_train, alpha = 1)
lasso_pred <- predict(lasso_model, x_val, s = "lambda.min")
lasso_rmse <- rmse(y_val, lasso_pred)
print(lasso_rmse)
```

## 2-5) Principal Component Analysis (PCA)
PCA is employed for dimensionality reduction, transforming the original variables into a smaller set of uncorrelated variables called principal components, which capture the most variance in the data. This is beneficial for improving model efficiency and handling multicollinearity..
```
# PCA
pca_model <- prcomp(x_train, scale. = TRUE)
pca_train <- as.data.frame(predict(pca_model, x_train))
pca_val <- as.data.frame(predict(pca_model, x_val))
pca_model <- lm(y_train ~ ., data = pca_train)
pca_pred <- predict(pca_model, pca_val)
pca_rmse <- rmse(y_val, pca_pred)
print(pca_rmse)
```

## 2-6) Principal Component Regression (PCR)
PCR is a two-stage regression technique that first reduces the dimensionality of the data using PCA and then applies Linear Regression on the reduced dataset. It combines the benefits of PCA and regression, making it suitable for datasets with high multicollinearity or numerous predictors.
```
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
```

## 2-7) Partial Least Squares (PLS)
PLS Regression also aims at predicting a dependent variable by extracting the essential predictors through a latent variable approach. It is particularly effective when the predictors are highly collinear or when the number of predictors is large compared to the number of observations.
```
# PLS
pls_model <- plsr(as.formula(paste(target, "~ .")), data = train, validation = "CV")
pls_rmsep <- RMSEP(pls_model)
pls_ncomp <- which.min(pls_rmsep$val[2, , 1])
pls_pred <- predict(pls_model, validation, ncomp = pls_ncomp)
pls_rmse <- rmse(validation[[target]], pls_pred)
print(pls_rmse)
```

## 2-8) Model Evaluation and Selection
Models were evaluated using the Root Mean Square Error (RMSE) metric on a validation set derived from the original training data. This metric quantifies the average magnitude of the prediction errors, providing a measure of model accuracy.

## 2-9) Final Prediction
The final model was selected based on the lowest RMSE during validation. The chosen model was then re-trained on the combined training and validation dataset and used to make predictions on the test dataset. The predictions were output in a structured format suitable for further analysis or submission in case of a competition.
```
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
```

### 3. Data
The dataset used in this project consists of detailed nutritional information for a variety of foods, represented through 45 variables. These variables include macronutrients (e.g., fats, proteins, carbohydrates), micronutrients (e.g., vitamins, minerals), and other relevant biochemical properties that characterize each food item. The primary target variable of interest is the energy content of these foods, measured in kilocalories (Kcal), which we aim to predict based on the nutritional characteristics provided.

Here are the actual column names included in the dataset.
<img width="890" height="239" alt="Image" src="https://github.com/user-attachments/assets/cc49ba84-02f4-434a-b3e2-c324fd16e8b2" />

