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
