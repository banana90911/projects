library(tidyverse)
library(caret)
library(cluster)
library(nnet)
library(e1071)

# load data
data <- read.csv("../volume/data/raw/train.csv")

str(data)
summary(data)

# standardization
numeric_data <- data %>% select(where(is.numeric))
data_scaled <- scale(numeric_data)

# number of clusters using K-means
set.seed(9091)
k3 <- kmeans(data_scaled, centers = 3, nstart = 25)
k4 <- kmeans(data_scaled, centers = 4, nstart = 25)

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

data3 <- data.frame(numeric_data, species = labels3)
data4 <- data.frame(numeric_data, species = labels4)

# ensure the species column is a factor
data3$species <- as.factor(data3$species)
data4$species <- as.factor(data4$species)

# split data
set.seed(123)
trainIndex <- createDataPartition(data3$species, p = .7, 
                                  list = FALSE, 
                                  times = 1)
trainData <- data3[trainIndex,]
validData <- data3[-trainIndex,]

# train multinomial logistic regression model
multinom_model <- multinom(species ~ ., data = trainData)

# train SVM model
svm_model <- svm(species ~ ., data = trainData, kernel = "linear")

# validate the logistic regression model
pred_valid <- predict(multinom_model, newdata = validData)
valid_acc <- mean(pred_valid == validData$species)
cat("Accuracy (logistic regression): ", valid_acc, "\n")

# validate the SVM model
pred_valid_svm <- predict(svm_model, newdata = validData)
valid_acc_svm <- mean(pred_valid_svm == validData$species)
cat("Accuracy (support vector machine): ", valid_acc_svm, "\n")


# final predictions
final_pred <- predict(svm_model, newdata = data3)

final_predictions <- data.frame(
  Id = paste0("sample_", 1:nrow(data3)),
  Species = final_pred
)

head(final_predictions)

# save file
write.csv(final_predictions, "/Users/siheonjung/Desktop/psu/summer 2024/stat380/5/Week05/project/volume/data/raw/final_predictions_svm.csv", row.names = FALSE)
