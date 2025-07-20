import os
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import numpy as np

## Data preprocessing
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
    batch_size = 2,
    class_mode = None,
    shuffle = False
)

## Build CNN model
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

## Train model
history = model.fit_generator(
    train_generator, 
    steps_per_epoch = train_df.shape[0] // 15, # number of rows of train data // batch size
    epochs = 3,
    validation_data = validation_generator,
    validation_steps = validate_df.shape[0] // 15 # number of rows of validation data // batch size
)

## Predict
predict = model.predict_generator(test_generator, steps = np.ceil(test_df.shape[0] / 15)) # number of rows of test data / batch size
print(predict)

test_df["label"] = np.argmax(predict, axis = -1)

label_map = dict((v,k) for k,v in train_generator.class_indices.items())
test_df["label"] = test_df["label"].replace(label_map)
print(test_df.head())

## Save file
test_df.to_csv("/Users/siheonjung/Desktop/psu/fall 2023/cmpsc448/final project/Convolutional Neural Network/cnn.csv", index = False)