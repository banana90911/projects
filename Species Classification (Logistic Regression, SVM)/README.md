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
