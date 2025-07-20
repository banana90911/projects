### 1. Introduction
In this project, we aim to develop a predictive model using linear regression to estimate the sale prices of properties. We start with our datasets, train and test, which include various features of properties such as area, year of construction, quality of material, condition, and several others, alongside the sale prices. The primary goal is to accurately predict the ‘SalePrice’ of properties based on their characteristics. To achieve this, a linear regression model will be applied, which is a fundamental statistical approach that assumes a linear relationship between the input variables and the single output variable (SalePrice). After developing the model using the training data, we will evaluate its performance on the validation data, which is splitted from original data, using the Root Mean Squared Error (RMSE) metric. RMSE will provide a clear indication of how close the predicted sale prices are to the actual values in the dataset, with a lower RMSE indicating a better fit of the model to the data. Lastly, final predictions will be made using the trained and evaluated model and test dataset.

### 2. Methodology
The primary objective of this methodology is to accurately predict a numeric variable (SalePrice) using a linear regression model. Linear regression is used to establish a relationship between the dependent variable (SalePrice) and several independent variables (features of the houses).

## 2-1) Data Preprocessing
**Loading Data:** Train and test datasets were loaded using the ‘readr’ package in R.
```
# load data
data <- read_csv("/Users/siheonjung/Desktop/psu/summer 2024/stat380/1/assignment/data/Stat_380_train.csv")
test <- read_csv("/Users/siheonjung/Desktop/psu/summer 2024/stat380/1/assignment/data/Stat_380_test.csv")
```

**Handling Missing Values:** To handle missing values, null values in numeric columns were replaced with the
mean of the respective columns. This ensures that the dataset is complete and can be used for modeling without
errors.
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
  numeric_columns <- df %>%
    select(where(is.numeric)) %>%
    select(-all_of(excol)) %>%
    colnames()
  df %>%
    mutate(across(all_of(numeric_columns), ~ ( . - mean(.)) / sd(.)))
}

norm_data <- normalize(data, excol = "Id")
norm_test <- normalize(test, excol = "Id")
```

## 2-3) Model Training
**Splitting Data:** The normalized training dataset was split into training and validation sets using an 80-20 split. This allows for model evaluation on unseen data.
```
# split data
set.seed(9091)
trainIndex <- createDataPartition(norm_data$SalePrice, p = 0.8, list = FALSE, times = 1)
train <- norm_data[trainIndex, ] # train dataset
validation <- norm_data[-trainIndex, ] # validation dataset
```

**Training the Model:** A linear regression model was trained using the training dataset. The lm function in R was used to fit the model.
```
# multiple linear regression
model <- lm(SalePrice ~ ., data = train)
```

## 2-4) Model Evaluation
**Predictions:** Predictions were made on the validation dataset. The predicted values were then denormalized and exponentiated to revert the normalization and the log-transformation applied earlier.
```
# predict using validation
predictions <- predict(model, newdata = validation)

# denormalization
saleprice_mean <- mean(data$SalePrice)
saleprice_sd <- sd(data$SalePrice)
denorm_predictions <- exp(predictions * saleprice_sd + saleprice_mean)
```

**Calculating RMSE:** The Root Mean Square Error (RMSE) was calculated to evaluate the performance of the model. The RMSE provides a measure of how well the model's predictions match the actual values. And it resulted in RMSE of 0.5937.
```
# RMSE using validation dataset
rmse <- sqrt(mean((validation$SalePrice - predictions)^2))
print(paste("RMSE: ", rmse))
```
> RMSE: 0.593680939406237

## 2-5) Final Predictions on Test Data
**Predictions on Test Data:** The model was used to predict SalePrice on the test dataset. These predictions were also denormalized and exponentiated to revert the normalization and the log-transformation.
```
# prediction using test dataset
final_predictions <- predict(model, newdata = norm_test)

# denormalization
denorm_final_predictions <- exp(final_predictions * saleprice_sd + saleprice_mean)
```

**Save File:** The predictions were saved to a csv file for submission.
```
# save file
write_csv(data.frame(Id = test_id, SalePrice = denorm_final_predictions), "/Users/siheonjung/Desktop/psu/summer 2024/stat380/1/assignment/data/final_predictions.csv")
```

### 3. Data
The dataset used in this analysis consists of a training set and a test set, both containing information about various attributes of houses. One difference between training set and test set is that the test set does not include SalePrice column, as it is the target variable to be predicted.

<img width="960" height="742" alt="Image" src="https://github.com/user-attachments/assets/ae38b69c-d641-4fba-92c6-de8eeec3d70e" />

As mentioned above, data cleaning involved handling missing values and converting categorical variables.

### 4. Analyze
Initially, I focused on feature selection when attempting to train the model. To do this, I calculated the correlation of each feature with SalePrice, and used different combinations of the features that had the highest correlations. However, using all the features ultimately showed the best RMSE, so I shifted my focus from feature selection to feature engineering. Applying normalization, the RMSE, which had been over 10,000, decreased to 0.57. Then, I applied log-transform method, and strangely, the result showed a higher RMSE value (0.59) when both normalization and log transform methods were applied than when only normalization was applied. However, the score on the scoreboard was higher when both methods were applied.

### 5. Conclusion
The project to develop a predictive model using linear regression for estimating sale prices was performed. Through data preprocessing, feature engineering, and model evaluation, the linear regression model provided reliable estimates that closely aligned with actual sale prices, as evidenced by a final RMSE of 0.5937. Notably, the model's performance on the validation set was robust, suggesting good generalizability on unseen data. The log transformation and normalization of features significantly improved the model's accuracy, although the RMSE was slightly higher when both methods were combined than when normalization was applied alone. Furthermore, the model ranked well on the scoreboard. It was 26712.59 at first and 26175.04 at the second.
