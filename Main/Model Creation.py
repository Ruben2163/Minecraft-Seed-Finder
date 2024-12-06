import tensorflow as tf
import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, Input
from tensorflow.keras.utils import Sequence
from sklearn.model_selection import train_test_split
import cv2
import matplotlib.pyplot as plt
import time
start = time.time()

# Constants
IMG_SIZE = (256, 256)
BATCH_SIZE = 200
EPOCHS = 10

# Custom Data Generator for Regression
class RegressionDataGenerator(Sequence):
    def __init__(self, image_paths, labels, batch_size, img_size):
        self.image_paths = image_paths
        self.labels = labels
        self.batch_size = batch_size
        self.img_size = img_size

    def __len__(self):
        return int(np.ceil(len(self.image_paths) / self.batch_size))

    def __getitem__(self, idx):
        batch_image_paths = self.image_paths[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_labels = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]

        images = np.array([cv2.resize(cv2.imread(img), self.img_size) / 255.0 for img in batch_image_paths])
        labels = np.array(batch_labels, dtype=np.float32)

        return images, labels

# Load Dataset
data_dir = "data"  # Update with your data path
image_paths = []
labels = []

# Loop through directories (seeds) and process images inside each seed folder
for seed_folder in os.listdir(data_dir):
    folder_path = os.path.join(data_dir, seed_folder)

    # Check if it's a directory (skip files like .DS_Store)
    if not os.path.isdir(folder_path):
        continue

    try:
        # Extract the seed value from the folder name (e.g., "seed_1" -> 1)
        seed = int(seed_folder.split("_")[1])  # Assumes the format is "seed_X"
    except ValueError:
        continue  # Skip if the folder name is not in the expected format

    # Loop through images inside the seed folder
    for image_name in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_name)

        # Add image path and corresponding label (seed)
        if image_path.lower().endswith(('.png', '.jpg', '.jpeg')):  # Check if it's an image file
            image_paths.append(image_path)
            labels.append(seed)

# Debug: Check how many images and labels were loaded
print(f"Number of images found: {len(image_paths)}")
print(f"Number of labels found: {len(labels)}")

# If no images were found, raise an error or provide feedback
if len(image_paths) == 0:
    raise ValueError("No images were found in the dataset directory. Check the folder structure.")

# Train-Test Split
train_paths, val_paths, train_labels, val_labels = train_test_split(image_paths, labels, test_size=0.2, random_state=42)

# Create Data Generators
train_gen = RegressionDataGenerator(train_paths, train_labels, BATCH_SIZE, IMG_SIZE)
val_gen = RegressionDataGenerator(val_paths, val_labels, BATCH_SIZE, IMG_SIZE)

# Model Architecture
model = Sequential([
    Input(shape=(256, 256, 3)),
    Conv2D(16, (3, 3), activation='relu'),
    MaxPooling2D(),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='linear')  # Linear activation for regression
])

# Compile Model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Train the Model
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS
)

# Save the Model
os.makedirs('models', exist_ok=True)
model.save('models/Minecraft-Seed-Finder_2.h5')

end = time.time()
print(end-start)
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
