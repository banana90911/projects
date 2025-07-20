import pandas as pd
import os
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Flatten, Dropout, BatchNormalization
from keras.models import Model
import tensorflow as tf
from vit_keras import vit
import numpy as np

## Data preprocessing
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

#print(train_df.head())
#print(validate_df.head())
#print(test_df.head())

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

## Build model
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

## Train model
history = model.fit(
    train_generator,
    steps_per_epoch = train_generator.samples // 15,
    epochs = 5,
    validation_data = test_generator,
    validation_steps = test_generator.samples // 15
)

## Predict
predict = model.predict_generator(test_generator) # number of rows of test data / batch size
predicted_labels = ["dog" if pred > 0.5 else "cat" for pred in predict]

test_df["label"] = predicted_labels
print(test_df.head(50))

label_map = dict((v,k) for k,v in train_generator.class_indices.items())
test_df["label"] = test_df["label"].replace(label_map)
print(test_df.head(50))

## Save file
test_df.to_csv("/Users/siheonjung/Desktop/psu/fall 2023/cmpsc448/final project/Transformer-based Systems/vit.csv", index = False)
