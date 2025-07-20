### 1. Introduction
In this project, we aim to develop a predictive model using linear regression to estimate the sale prices of properties. We start with our datasets, train and test, which include various features of properties such as area, year of construction, quality of material, condition, and several others, alongside the sale prices.
The primary goal is to accurately predict the ‘SalePrice’ of properties based on their characteristics. To achieve this, a linear regression model will be applied, which is a fundamental statistical approach that assumes a linear relationship between the input variables and the single output variable (SalePrice). Then, we develop our models using additional features: best subset regression, step regression, cross validation, polynomial regression, regression splines, and smoothing splines. After developing the model using the training data, we will evaluate its performance on the validation data, which is splitted from original data, using the Root Mean Squared Error (RMSE) metric. RMSE will provide a clear indication of how close the predicted sale prices are to the actual values in the dataset, with a lower RMSE indicating a better fit of the model to the data. Lastly, final predictions will be made using the trained and evaluated model and test dataset. 

### 2. Methodology
The primary objective of this methodology is to accurately predict a numeric variable (SalePrice) using a linear regression model. Linear regression is used to establish a relationship between the dependent variable (SalePrice) and several independent variables (features of the houses).

## 2-1) Data Preprocessing
**Loading Data:** Train and test datasets were loaded using the ‘readr’ package in R.
```
# load data
data <- read_csv("/Users/siheonjung/Desktop/psu/summer 2024/stat380/2/data/Stat_380_train.csv")
test <- read_csv("/Users/siheonjung/Desktop/psu/summer 2024/stat380/2/data/Stat_380_test.csv")
```

**Handling Missing Values:** To handle missing values, null values in numeric columns were replaced with the mean of the respective columns. This ensures that the dataset is complete and can be used for modeling without errors.
```
# replace null to average value if the variable is numeric
na_mean <- function(df) {
  df %>%
    mutate(across(where(is.numeric), ~ ifelse(is.na(.), mean(., na.rm = TRUE), .)))
}

data <- na_mean(data)
test <- na_mean(test)
```

**Converting Categorical Variables:** Categorical variables were converted to factors, which is essential for regression modeling in R.
```
# Convert categorical variables to factors
data <- data %>% mutate(across(where(is.character), as.factor))
test <- test %>% mutate(across(where(is.character), as.factor))
```

## 2-2) Feature Engineering
**Log-Transformation:** The SalePrice variable was log-transformed to stabilize the variance and make the distribution more normal. This helps in improving the performance of the linear regression model.
```
# Log-transform SalePrice
data$SalePrice <- log(data$SalePrice)
```

**Normalization:** Numerical features were normalized to ensure that they are on a comparable scale. This step excludes the Id column.
```
# normalization
normalize <- function(df, excol) {
  numeric_columns <- colnames(df)[sapply(df, is.numeric) & !colnames(df) %in% excol]
  df[, numeric_columns] <- scale(df[, numeric_columns])
  df
}

norm_data <- normalize(data, excol = "Id")
norm_test <- normalize(test, excol = "Id")
```

**Splitting Data:** The normalized training dataset was split into training and validation sets using an 80-20 split. This allows for model evaluation on unseen data.
```
# split data
set.seed(9091)
trainIndex <- createDataPartition(norm_data$SalePrice, p = 0.8, list = FALSE, times = 1)
train <- norm_data[trainIndex, ] # train dataset
validation <- norm_data[-trainIndex, ] # validation dataset
```

## 2-3) Multiple Linear Regression
**Objective:** Predict the SalePrice using a simple and interpretable model based on the linear relationship between the predictors and the response variable.

**Modeling:** A linear regression model was built using all available predictors.

**Evaluation:** Predictions were made on the validation set, and the Root Mean Squared Error (RMSE) was calculated to assess model performance.
```
# multiple linear regression
model_lm <- lm(SalePrice ~ ., data = train)
predictions_lm <- predict(model_lm, newdata = validation)
rmse_lm <- calc_rmse(validation$SalePrice, predictions_lm)
print(paste("rmse_lm: ", rmse_lm)) # 0.593680939406237
```

## 2-4) Best Subsets Regression
**Objective:** Select the most relevant subset of predictors that best explains the variability in the response variable.

**Variable Selection:** The regsubsets function was used to identify subsets of predictors that maximize the adjusted R-squared value.

**Modeling:** A linear regression model was constructed using the best subset of predictors.

**Evaluation:** The model was evaluated on the validation set using RMSE.
```
# best subsets regression
subsets <- regsubsets(SalePrice ~ ., data = train, nvmax = 13)
summary <- summary(subsets)
vars <- names(coef(subsets, which.max(summary$adjr2)))[-1]
real_vars <- vars[vars %in% colnames(train)]
model_bsr <- lm(as.formula(paste("SalePrice ~", paste(real_vars, collapse = " + "))), data = train)
predictions_bsr <- predict(model_bsr, newdata = validation)
rmse_bsr <- calc_rmse(validation$SalePrice, predictions_bsr)
print(paste("rmse_bsr: ", rmse_bsr)) # 0.59504310857517
```

## 2-5) Stepwise Regression
**Objective:** Improve the linear regression model by iteratively adding or removing predictors based on the AIC criterion.

**Variable Selection:** Stepwise regression (both forward and backward) was performed using the stepAIC function to select the optimal set of predictors.

**Modeling:** A linear regression model was built with the selected predictors.

**Evaluation:** The model's performance was evaluated on the validation set using RMSE.
```
# step regression
model_sr <- stepAIC(lm(SalePrice ~ ., data = train), direction = "both", trace = FALSE)
predictions_sr <- predict(model_sr, newdata = validation)
rmse_sr <- calc_rmse(validation$SalePrice, predictions_sr)
print(paste("rmse_sr: ", rmse_sr)) # 0.593581576851409
```

## 2-6) Stepwise Regression
**Objective:** Capture non-linear relationships between the predictors and the response variable by introducing flexibility into the model.

**Modeling:** Spline terms were created for numeric predictors using the bs function from the splines package, with a specified degree of 3.

**Evaluation:** A linear regression model incorporating spline terms was built and evaluated on the validation set using RMSE.
```
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
```

## 2-7) Smoothing Splines
**Objective:** Provide a flexible yet smooth fit to the data by minimizing the penalized residual sum of squares.

**Modeling:** A Generalized Additive Model (GAM) was constructed using the gam function from the mgcv package, fitting smoothing splines to the predictors.

**Evaluation:** The model was evaluated on the validation set using RMSE.
```
# smoothing splines
formula <- as.formula(paste("SalePrice ~", paste(colnames(train)[!colnames(train) %in% c("SalePrice", "Id")], collapse = " + ")))
model_ss <- gam(formula, data = train, family = gaussian(), method = "REML")
predictions_ss <- predict(model_ss, newdata = validation)
rmse_ss <- calc_rmse(validation$SalePrice, predictions_ss)
print(paste("rmse_ss: ", rmse_ss)) # 0.593721957145594
```

## 2-8) Final Predictions on Test Data
The final model, based on regression splines, was used to predict SalePrice values for the test dataset. These predictions were then transformed back from the log scale and denormalized to obtain the final predicted SalePrice values, which were saved to a CSV file. 
```
# prediction using test dataset
final_predictions <- predict(model_rs, newdata = norm_test)

# denormalization
denorm_final_predictions <- exp(final_predictions * saleprice_sd + saleprice_mean)

# save file
write_csv(data.frame(Id = test_id, SalePrice = denorm_final_predictions), "/Users/siheonjung/Desktop/psu/summer 2024/stat380/2/data/final_predictions.csv")
```

### 3. Data
The dataset used in this analysis consists of a training set and a test set, both containing information about various attributes of houses. One difference between training set and test set is that the test set does not include SalePrice column, as it is the target variable to be predicted. 
<img width="883" height="367" alt="Image" src="https://github.com/user-attachments/assets/c822a165-4979-48f5-a67c-a4aa8f641c10" />
<img width="961" height="374" alt="Image" src="https://github.com/user-attachments/assets/b1e4de26-6e74-4110-8244-657c72d57d5c" />

As mentioned above, data cleaning involved handling missing values and converting categorical variables.

### 4. Analyze
Initially, I have developed different linear regression models: multiple linear regression, best subsets regression, step regression, regression splines, and smoothing splines. Similar to the project in week 1, using all the features ultimately showed the best RMSE. Then, applying normalization, the RMSE, which had been over 10,000, decreased to 0.57. Then, I applied log-transform method, and strangely, the result showed a higher RMSE value (0.59) when both normalization and log transform methods were applied than when only normalization was applied. However, applying log-transform showed better performance with test dataset. As a result, among different models, regression splines model showed the lowest RMSE (0.5517). For hyperparameter tuning, I have tested five different degree values: 1 to 10, considering that high degree may result in overfitting. Finally, degree of 5 showed the lowest RMSE score. Therefore, regression splines model was used for final predictions.

### 5. Conclusion
In this project, I developed a predictive model to estimate property sale prices using various regression techniques. Starting with comprehensive data preprocessing, I handled missing values, converted categorical variables, and normalized numeric features. I explored multiple models, including multiple linear regression, best subsets regression, stepwise regression, regression splines, and smoothing splines. Each model was evaluated using the Root Mean Squared Error (RMSE) on a validation set. Regression splines, which capture non-linear relationships by introducing spline terms for numeric predictors, performed best with the lowest RMSE. Hyperparameter tuning further improved the model, and the final regression spline model was used to predict sale prices on the test dataset. The final predictions showed the fair result score 22916.29873.
