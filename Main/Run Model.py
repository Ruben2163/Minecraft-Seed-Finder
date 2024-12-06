from datetime import datetime
import os
import tensorflow as tf
import cv2
import numpy as np
import csv

# Record the start time
StartTime = datetime.now()

# Load the trained model
model_path = 'models/Minecraft-Seed-Finder_2.h5'  # Path to your trained model
model = tf.keras.models.load_model(model_path)

# Function to preprocess the input image
def load_and_preprocess_image(image_path):
    """Loads and preprocesses an image for the model."""
    # Read the image
    img = cv2.imread(image_path)
    # Check if the image exists
    if img is None:
        raise FileNotFoundError(f"Could not read the image at {image_path}")
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Resize the image to the input size of the model
    img = cv2.resize(img, (256, 256))
    # Normalize pixel values to the range [0, 1]
    img = img.astype('float32') / 255.0
    # Add a batch dimension
    img = np.expand_dims(img, axis=0)
    return img

# Folder containing the images
folder_path = 'test_img'  # Update with your folder's path
output_csv_path = 'seed_predictions.csv'  # Output CSV file for predictions

# Prepare the CSV file for writing predictions
with open(output_csv_path, mode='w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    # Write the header row
    csv_writer.writerow(['Filename', 'Predicted Seed'])

    # Iterate through all files in the folder
    for filename in os.listdir(folder_path):
        # Process only image files
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(folder_path, filename)
            try:
                # Preprocess the image
                input_image = load_and_preprocess_image(image_path)

                # Make a prediction using the model
                prediction = model.predict(input_image, verbose=0)  # Suppress progress output
                predicted_seed = int(round(prediction[0][0]))  # Assuming the model outputs a seed value

                # Write results to the CSV file
                csv_writer.writerow([filename, predicted_seed])
                print(f"Processed {filename}: Predicted Seed {predicted_seed}")

            except Exception as e:
                print(f"Error processing {filename}: {e}")

# Calculate and print the benchmark time
print(f'Benchmark Time: {datetime.now() - StartTime}')