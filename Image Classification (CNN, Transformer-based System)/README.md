### Introduction

These days, many websites use a verification system that we need to select the correct photo(s) to prove that we are not a robot. I've always wondered if that's really something robots can't do. And while taking this class, I learned about deep learning, which allows robots or computers to perform image classification. So, I decided to create a system for image classification, and for classes I chose cat and dog.

Among Convolutional Neural Network (CNN), Recurrent Neural Network (RNN), and Transformer-based systems, RNN is more commonly used for sequential data where the order matters, such as natural language processing tasks or time series analysis rather than image classification. Therefore, I decided to use Convolutional Neural Network and Transformer-based systems. 

Data files include two sets: train and test datasets. Train dataset has 25000 images of cats and dogs with labels shown on their image file name, such as “dog.12234.jpg.” Test dataset has 12500 images that do not show the label.

### Convolutional Neural Network

**Data Preprocessing:** After reading the folders, I have created data frames for convenience in handling data. Then, I split train dataset to train and validation datasets with test_size = 0.3. 
```
# Read train file
data = os.listdir("/Users/siheonjung/Desktop/psu/fall 2023/cmpsc448/final project/data/train")


labels = []
for image in data:
    if image.split(".")[0] == "dog":
        labels.append(1)
    else:
        labels.append(0)


df = pd.DataFrame({
    "filename": data,
    "label": labels
})


# Read test file
test = os.listdir("/Users/siheonjung/Desktop/psu/fall 2023/cmpsc448/final project/data/test1")
test_df = pd.DataFrame({
    "filename": test
})


# Split data
df["label"] = df["label"].replace({0: "cat", 1: "dog"})


train_df, validate_df = train_test_split(df, test_size = 0.3, random_state = 42)
train_df = train_df.reset_index(drop = True) # to establish a continuous index as well as remove one or more unwanted levels
validate_df = validate_df.reset_index(drop = True)
```

To build a model, we need to generate batches for each dataset. I used “ImageDataGenerator” imported from “keras.” There are only two classes: cat and dog. Therefore, “class_mode” can be “binary.”
```
# Generate batches
train_datagen = ImageDataGenerator(
    rescale = 1./255,
    rotation_range = 15,
    width_shift_range = 0.1,
    height_shift_range = 0.1,
    shear_range = 0.1,
    zoom_range = 0.2,
    horizontal_flip = True,
    fill_mode = "nearest"
)


train_generator = train_datagen.flow_from_dataframe(
    train_df, 
    "/Users/siheonjung/Desktop/psu/fall 2023/cmpsc448/final project/data/train/", 
    x_col = "filename",
    y_col = "label",
    target_size = (128, 128),
    batch_size = 15,
    class_mode = "categorical"
)


validation_datagen = ImageDataGenerator(rescale = 1./255)


validation_generator = validation_datagen.flow_from_dataframe(
    validate_df, 
    "/Users/siheonjung/Desktop/psu/fall 2023/cmpsc448/final project/data/train/", 
    x_col = "filename",
    y_col = "label",
    target_size = (128, 128),
    batch_size = 15,
    class_mode = "categorical"
)


test_datagen = ImageDataGenerator(rescale = 1./255)


test_generator = test_datagen.flow_from_dataframe(
    test_df, 
    "/Users/siheonjung/Desktop/psu/fall 2023/cmpsc448/final project/data/test1/", 
    x_col = "filename",
    y_col = None,
    target_size = (128, 128),
    batch_size = 15,
    class_mode = None,
    shuffle = False
)
```

**Build CNN model:** Building a CNN model, I used “Sequential” imported from “keras” and added layers: convolutional layers with increasing number of filters, pooling layers, and fully connected layers. Then, the model is compiled to be prepared to train.
```
model = Sequential([
    # Layer 1
    Conv2D(32, (3, 3), activation = "relu", input_shape = (128, 128, 3)), # 32 filters, 
    MaxPooling2D((2, 2)),


    # Layer 2
    Conv2D(64, (3, 3), activation = "relu"),
    MaxPooling2D((2, 2)),


    # Layer 3
    Conv2D(128, (3, 3), activation = "relu"),
    MaxPooling2D((2, 2)),


    # Fully connected layer
    Flatten(),
    Dense(512, activation = "relu"), # 512: number of neurons. can be changed
    Dense(2, activation = "softmax") # 2: number of classes, cat and dog
])


# Compile model
model.compile(optimizer = "rmsprop", loss = "categorical_crossentropy", metrics = ["accuracy"])
```

**Train model:** “train_generator” and “validation_generator” were used as parameters of “fit_generator” to train the model. In the context of neural network, an epoch is one complete pass through the entire training dataset. First, I set number of epochs to 30 to see the accuracy trend. 
```
history = model.fit_generator(
    train_generator, 
    steps_per_epoch = train_df.shape[0] // 15, # number of rows of train data // batch size
    epochs = 30,
    validation_data = validation_generator,
    validation_steps = validate_df.shape[0] // 15 # number of rows of validation data // batch size
)
```

```
Then, I observed that the accuracy increases at first but remains almost the same, similar to logarithm graph.
Epoch 1/30
1166/1166 [==============================] - 99s 85ms/step - loss: 0.6488 - accuracy: 0.6422 - val_loss: 0.5491 - val_accuracy: 0.7363
Epoch 2/30
1166/1166 [==============================] - 103s 89ms/step - loss: 0.5725 - accuracy: 0.7118 - val_loss: 0.5428 - val_accuracy: 0.7285
Epoch 3/30
1166/1166 [==============================] - 102s 88ms/step - loss: 0.5310 - accuracy: 0.7474 - val_loss: 0.4874 - val_accuracy: 0.7587
Epoch 4/30
1166/1166 [==============================] - 103s 89ms/step - loss: 0.5006 - accuracy: 0.7655 - val_loss: 0.4733 - val_accuracy: 0.7769
Epoch 5/30
1166/1166 [==============================] - 104s 89ms/step - loss: 0.4900 - accuracy: 0.7756 - val_loss: 0.4196 - val_accuracy: 0.8093
Epoch 6/30
1166/1166 [==============================] - 101s 86ms/step - loss: 0.4730 - accuracy: 0.7850 - val_loss: 0.4760 - val_accuracy: 0.7928
Epoch 7/30
1166/1166 [==============================] - 100s 85ms/step - loss: 0.4630 - accuracy: 0.7915 - val_loss: 0.4901 - val_accuracy: 0.7913
Epoch 8/30
1166/1166 [==============================] - 102s 88ms/step - loss: 0.4529 - accuracy: 0.8013 - val_loss: 0.3823 - val_accuracy: 0.8323
Epoch 9/30
1166/1166 [==============================] - 102s 88ms/step - loss: 0.4449 - accuracy: 0.8029 - val_loss: 0.3853 - val_accuracy: 0.8443
Epoch 10/30
1166/1166 [==============================] - 103s 89ms/step - loss: 0.4403 - accuracy: 0.8095 - val_loss: 0.3804 - val_accuracy: 0.8336
Epoch 11/30
1166/1166 [==============================] - 102s 87ms/step - loss: 0.4230 - accuracy: 0.8161 - val_loss: 0.4320 - val_accuracy: 0.8445
Epoch 12/30
1166/1166 [==============================] - 102s 88ms/step - loss: 0.4251 - accuracy: 0.8161 - val_loss: 0.4808 - val_accuracy: 0.8021
Epoch 13/30
1166/1166 [==============================] - 104s 89ms/step - loss: 0.4200 - accuracy: 0.8184 - val_loss: 0.5782 - val_accuracy: 0.7881
Epoch 14/30
1166/1166 [==============================] - 103s 88ms/step - loss: 0.4222 - accuracy: 0.8200 - val_loss: 0.3356 - val_accuracy: 0.8596
Epoch 15/30
1166/1166 [==============================] - 105s 90ms/step - loss: 0.4132 - accuracy: 0.8251 - val_loss: 0.3429 - val_accuracy: 0.8607
Epoch 16/30
1166/1166 [==============================] - 105s 90ms/step - loss: 0.4028 - accuracy: 0.8275 - val_loss: 0.3402 - val_accuracy: 0.8676
Epoch 17/30
1166/1166 [==============================] - 105s 90ms/step - loss: 0.4114 - accuracy: 0.8259 - val_loss: 0.3866 - val_accuracy: 0.8557
Epoch 18/30
1166/1166 [==============================] - 107s 92ms/step - loss: 0.4068 - accuracy: 0.8269 - val_loss: 0.3179 - val_accuracy: 0.8727
Epoch 19/30
1166/1166 [==============================] - 105s 90ms/step - loss: 0.3964 - accuracy: 0.8310 - val_loss: 0.4525 - val_accuracy: 0.8325
Epoch 20/30
1166/1166 [==============================] - 104s 89ms/step - loss: 0.4197 - accuracy: 0.8333 - val_loss: 0.3261 - val_accuracy: 0.8747
Epoch 21/30
1166/1166 [==============================] - 106s 91ms/step - loss: 0.4046 - accuracy: 0.8297 - val_loss: 0.3562 - val_accuracy: 0.8629
Epoch 22/30
1166/1166 [==============================] - 103s 88ms/step - loss: 0.3914 - accuracy: 0.8360 - val_loss: 0.3255 - val_accuracy: 0.8753
Epoch 23/30
1166/1166 [==============================] - 104s 89ms/step - loss: 0.3934 - accuracy: 0.8367 - val_loss: 0.3372 - val_accuracy: 0.8665
Epoch 24/30
1166/1166 [==============================] - 104s 89ms/step - loss: 0.3857 - accuracy: 0.8344 - val_loss: 0.4036 - val_accuracy: 0.8520
Epoch 25/30
1166/1166 [==============================] - 105s 90ms/step - loss: 0.3967 - accuracy: 0.8364 - val_loss: 0.5821 - val_accuracy: 0.8269
Epoch 26/30
1166/1166 [==============================] - 104s 89ms/step - loss: 0.4029 - accuracy: 0.8333 - val_loss: 0.3441 - val_accuracy: 0.8687
Epoch 27/30
1166/1166 [==============================] - 105s 90ms/step - loss: 0.4024 - accuracy: 0.8375 - val_loss: 0.4458 - val_accuracy: 0.8607
Epoch 28/30
1166/1166 [==============================] - 104s 89ms/step - loss: 0.3885 - accuracy: 0.8395 - val_loss: 0.4656 - val_accuracy: 0.8337
Epoch 29/30
1166/1166 [==============================] - 106s 90ms/step - loss: 0.3826 - accuracy: 0.8396 - val_loss: 0.4349 - val_accuracy: 0.8572
Epoch 30/30
1166/1166 [==============================] - 103s 88ms/step - loss: 0.3890 - accuracy: 0.8406 - val_loss: 0.3838 - val_accuracy: 0.8695
```

**Prediction:** “predict_generator” was used to identify the labels for each image in test dataset and converted them into data frame. Lastly, data frame is saved to csv file.
```
predict = model.predict_generator(test_generator, steps = np.ceil(test_df.shape[0] / 15)) # number of rows of test data / batch size

test_df["label"] = np.argmax(predict, axis = -1)

label_map = dict((v,k) for k,v in train_generator.class_indices.items())
test_df["label"] = test_df["label"].replace(label_map)
print(test_df.head())

test_df.to_csv("/Users/siheonjung/Desktop/psu/fall 2023/cmpsc448/final project/Convolutional Neural
```
<img width="168" height="122" alt="Image" src="https://github.com/user-attachments/assets/83b5d87e-9dc3-4b4e-8fe1-34bdbe358abe" />


### Transformer-based System 
Data Preprocessing: It is pretty much similar to that of CNN. After reading the folders, I have created data frames for convenience in handling data. Then, I split train dataset to train and validation datasets with test_size = 0.3. 
```
# Read train file
data = os.listdir("/Users/siheonjung/Desktop/psu/fall 2023/cmpsc448/final project/data/train")
labels = []
for image in data:
    if image.split(".")[0] == "dog":
        labels.append((image, "dog"))
    else:
        labels.append((image, "cat"))


df = pd.DataFrame(labels, columns = ["filename", "label"])


# Read test file
test = os.listdir("/Users/siheonjung/Desktop/psu/fall 2023/cmpsc448/final project/data/test1")
test_df = pd.DataFrame({
    "filename": test
})


# Split data
train_df, validate_df = train_test_split(df, test_size = 0.3, random_state = 42)
train_df = train_df.reset_index(drop = True) # to establish a continuous index as well as remove one or more unwanted levels
validate_df = validate_df.reset_index(drop = True)
```

And we have the same data generators, but “class_mode” for “train_generator” and “validate_generator” are “binary,” not “categorical,” which won’t make difference since dealing with only two classes.
```
# Generate batches
train_datagen = ImageDataGenerator(
    rescale = 1./255,
    rotation_range = 15,
    width_shift_range = 0.1,
    height_shift_range = 0.1,
    shear_range = 0.1,
    zoom_range = 0.2,
    horizontal_flip = True,
    fill_mode = "nearest"
)


train_generator = train_datagen.flow_from_dataframe(
    train_df,
    "/Users/siheonjung/Desktop/psu/fall 2023/cmpsc448/final project/data/train/", 
    x_col = "filename",
    y_col = "label",
    target_size = (128, 128),
    batch_size = 15,
    class_mode = "binary"
)


val_datagen = ImageDataGenerator(rescale = 1./255)


val_generator = val_datagen.flow_from_dataframe(
    validate_df,
    "/Users/siheonjung/Desktop/psu/fall 2023/cmpsc448/final project/data/train/", 
    x_col = "filename",
    y_col = "label",
    target_size = (128, 128),
    batch_size = 15,
    class_mode = "binary"
)


test_datagen = ImageDataGenerator(rescale = 1./255)


test_generator = test_datagen.flow_from_dataframe(
    test_df, 
    "/Users/siheonjung/Desktop/psu/fall 2023/cmpsc448/final project/data/test1/", 
    x_col = "filename",
    y_col = None,
    target_size = (128, 128),
    batch_size = 15,
    class_mode = None,
    shuffle = False
)
```

**Build VIT model:** I used Vision Transformer (ViT) as a transformer-based system. For layers, “Dense” layers, “BatchNormalization” for normalization, and “Dropout” for regularization, are added. Then, similar to CNN, model is combined and compiled. One thing that is different from CNN is “learning rate,” which is 0.0001.
```
vit = vit.vit_b16(
    image_size = 128,  # Image size expected by the model
    pretrained = True,  # Load pre-trained weights
    include_top = False,  # Exclude classification head
    pretrained_top = False,  # Exclude top classification layers
)


# Add layers
layer = Dense(32, activation = "relu")(Flatten()(vit.output))
layer = BatchNormalization()(layer) # normalization
layer = Dropout(0.5)(layer) # regularization


layer = Dense(64, activation = "relu")(layer)
layer = Dropout(0.3)(layer)


layer = Dense(128, activation = "relu")(layer)
layer = Dropout(0.2)(layer)


output = Dense(1, activation = "sigmoid")(layer)


# Combine model
model = Model(inputs = vit.input, outputs = output)


# Compile model
model.compile(optimizer = tf.keras.optimizers.legacy.Adam(learning_rate = 0.0001), loss = "binary_crossentropy", metrics = ["accuracy"]) # a lower learning rate (e.g., 0.0001) means smaller steps during training.
```

**Train model:** Due to slow learning rate, it took about 1500 seconds per one epoch whereas CNN took about 100 seconds. I set number of epochs to 5.
```
history = model.fit(
    train_generator,
    steps_per_epoch = train_generator.samples // 15,
    epochs = 5,
    validation_data = test_generator,
    validation_steps = test_generator.samples // 15
)
```

Then, Like CNN, accuracy initially increased, but as the number of epochs passed, the rate of increase slowed down. However, I observed much higher accuracy at the end than that of CNN’s.
```
Epoch 1/5
1166/1166 [==============================] - 1565s 1s/step - loss: 0.3983 - accuracy: 0.8166 - val_loss: 0.0000e+00 - val_accuracy: 0.0000e+00
Epoch 2/5
1166/1166 [==============================] - 1507s 1s/step - loss: 0.1935 - accuracy: 0.9278 - val_loss: 0.0000e+00 - val_accuracy: 0.0000e+00
Epoch 3/5
1166/1166 [==============================] - 1484s 1s/step - loss: 0.1621 - accuracy: 0.9414 - val_loss: 0.0000e+00 - val_accuracy: 0.0000e+00
Epoch 4/5
1166/1166 [==============================] - 1499s 1s/step - loss: 0.1520 - accuracy: 0.9446 - val_loss: 0.0000e+00 - val_accuracy: 0.0000e+00
Epoch 5/5
1166/1166 [==============================] - 1485s 1s/step - loss: 0.1388 - accuracy: 0.9481 - val_loss: 0.0000e+00 - val_accuracy: 0.0000e+00
```

**Prediction:** “predict_generator” was used to identify the labels for each image in test dataset and converted them into data frame after changing them to “cat” and “dog.” Lastly, data frame is saved to csv file.
```
predict = model.predict_generator(test_generator) # number of rows of test data / batch size
predicted_labels = ["dog" if pred > 0.5 else "cat" for pred in predict]

test_df["label"] = predicted_labels

label_map = dict((v,k) for k,v in train_generator.class_indices.items())
test_df["label"] = test_df["label"].replace(label_map)
print(test_df.head(50))

test_df.to_csv("/Users/siheonjung/Desktop/psu/fall 2023/cmpsc448/final project/Transformer-based Systems/vit.csv", index = False)
```
<img width="158" height="124" alt="Image" src="https://github.com/user-attachments/assets/b65be3e9-a084-4723-b08c-b017f89d81c2" />

### Conclusion
Two deep learning systems suitable for image classification, Convolutional Neural Network and Transformer-based System, were used. Data preprocessing, which is very important in both machine learning and deep learning, was the biggest obstacle for me. It was difficult to find and meet the dataset required by the Sequential model and ViT model. So, I used a data frame that allows us to handle data more easily. Thus, I was able to train models with modified datasets and obtain output. According to observations, accuracy has increased for both systems, but as the number of cycles increased (as the number of epochs increased), the rate at which accuracy increased slowed down. The difference between the two models was speed. As mentioned before, CNN took about 100 seconds per epoch, but ViT took about 1500 seconds. And while CNN's accuracy was 0.8406 after 30 epochs, ViT recorded 0.9481 after 5 epochs. In other words, it can be concluded that ViT is superior to Image Classification than CNN.
