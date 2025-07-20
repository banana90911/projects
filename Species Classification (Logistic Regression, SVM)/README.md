## 1. Introduction
In this project, we aim to categorize observations into different species based on 15 covariates using unsupervised learning techniques. The dataset provided consists of several observations with various numerical attributes. Our objective is to employ clustering methods to identify the inherent groupings within the data, assign appropriate labels to these groups, and then build predictive models to classify the species accurately.

Given that the groups have no pre-existing labels, K-means clustering will be utilized to determine the natural clusters within the data. The number of species (clusters) is known to be between 3 and 4. Based on the clustering results, labels will be assigned to these groups and subsequently use supervised learning techniques, specifically multinomial logistic regression and Support Vector Machine (SVM), to build models that can predict the species of new observations.

The project will proceed through the following steps:
1. Load and preprocess the data, including standardization of numerical attributes.
2. Apply K-means clustering to identify natural groupings within the data.
3. Assign labels to the clusters based on specific samples provided.
4. Split the data into training and validation sets.
5. Train and evaluate multinomial logistic regression and SVM models.
6. Make final predictions using the best-performing model and save the results.

## 2. Methodology
### 2-1) Data Loading and Explanation
The first step in methodology involves loading the dataset and performing an initial exploration to understand its structure and summary statistics. This helps in identifying the types of variables present and any potential issues with the data such as missing values.
```
# load data
data <- read.csv("../volume/data/raw/train.csv")

str(data)
summary(data)
```

### 2-2) Data Preprocessing and Standardization
Since the dataset contains numerical attributes, these variables were standardized to have a mean of zero and a standard deviation of one. Standardization is essential to ensure that all variables contribute equally to the distance calculations during clustering.
```
# standardization
numeric_data <- data %>% select(where(is.numeric))
data_scaled <- scale(numeric_data)
```

### 2-3) Clustering using K-means
To uncover the inherent groupings in the data, the K-means clustering algorithm was employed. The objective of K-means clustering is to partition the data into K distinct clusters based on feature similarity. Both 3-cluster and 4-cluster solutions were explored to determine the optimal number of clusters. 

Objective is to identify natural groupings in the data. 
```
# number of clusters using K-means
set.seed(9091)
k3 <- kmeans(data_scaled, centers = 3, nstart = 25)
k4 <- kmeans(data_scaled, centers = 4, nstart = 25)
```

### 2-4) Assigning Labels to Clusters
Once the clusters are identified, labels were assigned to these clusters based on specific samples provided. This labeling process is necessary to map the clusters to actual species names for further classification tasks. 
```
# assign labels to clusters
assign_labels <- function(clusters, num_clusters) {
  labels <- rep(NA, length(clusters))
  if (num_clusters == 3) {
    labels[clusters == clusters[3]] <- 'species1'
    labels[clusters == clusters[9]] <- 'species2'
    labels[is.na(labels)] <- 'species3'
  } else if (num_clusters == 4) {
    labels[clusters == clusters[3]] <- 'species1'
    labels[clusters == clusters[9]] <- 'species2'
    labels[clusters == clusters[6]] <- 'species3'
    labels[is.na(labels)] <- 'species4'
  }
  return(labels)
}

# create labeled datasets
labels3 <- assign_labels(k3$cluster, 3)
labels4 <- assign_labels(k4$cluster, 4)
```

### 2-5) Splitting the Data
The labeled dataset was split into training and validation sets. This step is crucial for evaluating the performance of our classification models. 
```
# split data
set.seed(123)
trainIndex <- createDataPartition(data3$species, p = .7, 
                                  list = FALSE, 
                                  times = 1)
trainData <- data3[trainIndex,]
validData <- data3[-trainIndex,]
```

### 2-6) Multinomial Logistic Regression
A multinomial logistic regression model was trained to classify the species based on the 15 covariates. 
```
# train multinomial logistic regression model
multinom_model <- multinom(species ~ ., data = trainData)
```

### 2-7) Support Vector Machine
In addition to logistic regression, a support vector machine model was trained with a linear kernel for classification.
```
# train SVM model
svm_model <- svm(species ~ ., data = trainData, kernel = "linear")
```

### 2-8) Model Evaluation
The performance of both models were evaluated using accuracy on the validation set.
```
# validate the logistic regression model
pred_valid <- predict(multinom_model, newdata = validData)
valid_acc <- mean(pred_valid == validData$species)
cat("Accuracy (logistic regression): ", valid_acc, "\n")

# validate the SVM model
pred_valid_svm <- predict(svm_model, newdata = validData)
valid_acc_svm <- mean(pred_valid_svm == validData$species)
cat("Accuracy (support vector machine): ", valid_acc_svm, "\n")
```

### 2-9) Final Predictions
Using the best-performing model, final predictions were made on the entire dataset and save the results.
```
# final predictions
final_pred <- predict(svm_model, newdata = data3)

final_predictions <- data.frame(
  Id = paste0("sample_", 1:nrow(data3)),
  Species = final_pred
)

head(final_predictions)

# save file
write.csv(final_predictions, "/Users/siheonjung/Desktop/psu/summer 2024/stat380/5/Week05/project/volume/data/raw/final_predictions_svm.csv", row.names = FALSE)
```
> Logistic Regression Accuracy: 0.9836553
> Support Vector Machine Accuracy: 0.9895988

## 3. Data
The dataset used in this project consists of several observations, each characterized by 15 numerical covariates. These covariates serve as the features for classifying different species. The primary objective is to use these covariates to identify and classify the species through unsupervised and supervised learning techniques. Since the dataset includes only numerical variables, it is well-suited for clustering and classification algorithms that require numerical inputs.

The dataset contains 15 numerical variables, each representing a different attribute of the observations. These attributes include various measurements and features that help distinguish between different species. The data was loaded into R using the read.csv function, and an initial exploration was conducted using the str and summary functions to understand the structure and summary statistics of the dataset. This exploration helped identify the types of variables present and any potential issues, such as missing values.

To ensure that all variables contribute equally to the clustering process, the numerical variables were standardized to have a mean of zero and a standard deviation of one. This standardization is crucial because it prevents variables with larger scales from dominating the distance calculations used in clustering algorithms. The standardized data was then subjected to K-means clustering to identify natural groupings within the data. Both 3-cluster and 4-cluster solutions were explored to determine the optimal number of clusters.

After clustering, labels were assigned to the clusters based on specific sample indices provided. This labeling process was necessary to map the clusters to actual species names, facilitating subsequent classification tasks. A custom function was defined to assign species labels to the clusters based on the clustering results.

To prepare the data for supervised learning, the labeled dataset was split into training and validation sets. This step ensures that the models are trained on a subset of the data and evaluated on an unseen subset, providing a robust measure of their performance. The createDataPartition function from the caret package was used to split the data while maintaining the distribution of species labels.

Throughout the preprocessing phase, no significant data quality issues or missing values were identified, so no additional data cleaning steps were necessary. The preprocessed, labeled, and split dataset was then used for training and evaluating multinomial logistic regression and Support Vector Machine (SVM) models to classify the species based on the given covariates. This thorough data preparation ensured that the models were built on a clean and standardized dataset, maximizing their potential for accurate classification.


## 4. Analyze
The first step in analysis involved applying K-means clustering to the standardized dataset to uncover the inherent groupings within the data. Both 3-cluster and 4-cluster solutions were explored. The K-means algorithm partitions the data into K distinct clusters based on feature similarity, aiming to minimize the variance within each cluster. This unsupervised learning method does not require pre-labeled data and is ideal for discovering natural patterns.
After determining the optimal number of clusters, species labels were assigned to the clusters based on specific sample indices provided. This process involved defining a function to map clusters to species names, ensuring that certain samples were associated with known species. For example, in the 3-cluster solution, sample_3 was labeled as species1, and sample_9 as species2.

After labeling the clusters, it is proceeded to build two supervised learning models: multinomial logistic regression and Support Vector Machine (SVM) with a linear kernel. These models were trained to classify the species based on the 15 covariates.

Multinomial logistic regression model generalizes logistic regression to handle multiple classes. It models the probability of each class (species) as a function of the input features. The training data consisted of the labeled observations, with the species column as the response variable.
The SVM model with a linear kernel was used for classification. SVM aims to find the hyperplane that best separates the classes in the feature space. For multi-class classification, SVM uses strategies like one-vs-one or one-vs-all.

The models were evaluated using accuracy on the validation set. Accuracy measures the proportion of correctly classified instances out of the total instances in the validation set. This metric provides a straightforward assessment of model performance. 

The accuracy of the multinomial logistic regression model was calculated by comparing the predicted species to the actual species in the validation set. Similarly, the accuracy of the SVM model was calculated using the validation set. Logistic regression model resulted in 0.9836553, and SVM resulted in 0.9895988.

Using the trained SVM model, final predictions were made for all observations in the dataset. These predictions were saved in a dataframe with two columns: Id (ranging from sample_1 to sample_2250) and Species (species1, species2, species3, species4). 


## 5. Conclusion
In this project, species are classified based on 15 numerical covariates using both unsupervised and supervised learning techniques. Approach began with K-means clustering to identify natural groupings within the data, followed by labeling these clusters to facilitate supervised learning. Then, two models were trained: multinomial logistic regression and Support Vector Machine (SVM), and evaluated their performance using accuracy on a validation set.

The final SVM model demonstrated superior performance with an accuracy of 0.9895988 on the validation set, indicating its effectiveness in correctly classifying the species. This high level of accuracy underscores the model's robustness and reliability for species classification tasks. 
The final predictions, made using the trained SVM model, resulted in accuracy of 0.95777.
