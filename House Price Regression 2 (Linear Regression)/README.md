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
