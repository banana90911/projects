**Introduction**

These days, many websites use a verification system that we need to select the correct photo(s) to prove that we are not a robot. I've always wondered if that's really something robots can't do. And while taking this class, I learned about deep learning, which allows robots or computers to perform image classification. So, I decided to create a system for image classification, and for classes I chose cat and dog.

Among Convolutional Neural Network (CNN), Recurrent Neural Network (RNN), and Transformer-based systems, RNN is more commonly used for sequential data where the order matters, such as natural language processing tasks or time series analysis rather than image classification. Therefore, I decided to use Convolutional Neural Network and Transformer-based systems. 

Data files include two sets: train and test datasets. Train dataset has 25000 images of cats and dogs with labels shown on their image file name, such as “dog.12234.jpg.” Test dataset has 12500 images that do not show the label.

**Convolutional Neural Network**

Data Preprocessing: After reading the folders, I have created data frames for convenience in handling data. Then, I split train dataset to train and validation datasets with test_size = 0.3. 
