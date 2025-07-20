library(readr)
library(caret)
library(tidyverse)

# load data
data <- read_csv("/Users/siheonjung/Desktop/psu/summer 2024/stat380/1/assignment/data/Stat_380_train.csv")
test <- read_csv("/Users/siheonjung/Desktop/psu/summer 2024/stat380/1/assignment/data/Stat_380_test.csv")

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
  numeric_columns <- df %>%
    select(where(is.numeric)) %>%
    select(-all_of(excol)) %>%
    colnames()
  df %>%
    mutate(across(all_of(numeric_columns), ~ ( . - mean(.)) / sd(.)))
}

norm_data <- normalize(data, excol = "Id")
norm_test <- normalize(test, excol = "Id")

# split data
set.seed(9091)
trainIndex <- createDataPartition(norm_data$SalePrice, p = 0.8, list = FALSE, times = 1)
train <- norm_data[trainIndex, ] # train dataset
validation <- norm_data[-trainIndex, ] # validation dataset

# multiple linear regression
model <- lm(SalePrice ~ ., data = train)

# predict using validation
predictions <- predict(model, newdata = validation)

# denormalization
saleprice_mean <- mean(data$SalePrice)
saleprice_sd <- sd(data$SalePrice)
denorm_predictions <- exp(predictions * saleprice_sd + saleprice_mean)

# RMSE using validation dataset
rmse <- sqrt(mean((validation$SalePrice - predictions)^2))
print(paste("RMSE: ", rmse))

# prediction using test dataset
final_predictions <- predict(model, newdata = norm_test)

# denormalization
denorm_final_predictions <- exp(final_predictions * saleprice_sd + saleprice_mean)

# save file
write_csv(data.frame(Id = test_id, SalePrice = denorm_final_predictions), "/Users/siheonjung/Desktop/psu/summer 2024/stat380/1/assignment/data/final_predictions.csv")
