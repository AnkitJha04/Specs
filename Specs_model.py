import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Constants
IMG_SIZE = 64
BATCH_SIZE = 32
EPOCHS = 10
DATASET_DIR = "C:/Users/ankit/OneDrive/Documents/Personal/Repls/Raw Code/Specs/Dataset/asl_alphabet_train/asl_alphabet_train"

# Build CNN model for grayscale images
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),  # 1 channel for grayscale
    MaxPooling2D(2, 2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(29, activation='softmax')  # 29 classes
])

# Print model summary
model.summary()

# Preprocess the data using ImageDataGenerator
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode='grayscale',  # Grayscale conversion
    batch_size=BATCH_SIZE,
    subset='training',
    class_mode='categorical'
)

val_generator = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode='grayscale',  # Grayscale conversion
    batch_size=BATCH_SIZE,
    subset='validation',
    class_mode='categorical'
)

# Compile the model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS
)

# Evaluate the model
val_loss, val_acc = model.evaluate(val_generator)
print(f"\nValidation Accuracy: {val_acc:.4f}, Validation Loss: {val_loss:.4f}")

# Save the model
model.save("C:/Users/ankit/OneDrive/Documents/Personal/Repls/Resource files/asl_latest.h5")
print("Model saved as 'asl_latest.h5'")
