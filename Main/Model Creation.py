# Import Libraries
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import os

# Constants
IMG_SIZE = (256, 256)  # Image size for resizing
BATCH_SIZE = 32  # Number of samples per batch
EPOCHS = 20  # Training epochs
MAX_SEED = 2**64 - 1  # Max Minecraft seed (for normalization)

# Data Preprocessing and Augmentation
datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,  # Normalize pixel values to [0, 1]
    validation_split=0.2,  # Split data for validation
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

# Load Data
data_dir = "data"  # Replace with your dataset directory
train_data = datagen.flow_from_directory("data")
    data_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="raw",  # For regression tasks, use "raw" mode
    subset="training"
)
val_data = datagen.flow_from_directory(
    data_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="raw",
    subset="validation"
)

# Model Architecture
model = Sequential([
    Input(shape=(256, 256, 3)),  # Define input shape
    Conv2D(16, (3, 3), activation='relu'),
    MaxPooling2D(),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),  # Add dropout to prevent overfitting
    Dense(1, activation='linear')  # Linear activation for regression
])

# Compile Model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Train the Model
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS
)

# Save the Model
os.makedirs('models', exist_ok=True)
model.save('models/Minecraft-Seed-Finder.h5')

# Visualize Training Results
# Plot Loss
plt.figure()
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plot Mean Absolute Error
plt.figure()
plt.plot(history.history['mae'], label='Train MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.title('Mean Absolute Error')
plt.xlabel('Epochs')
plt.ylabel('MAE')
plt.legend()
plt.show()
