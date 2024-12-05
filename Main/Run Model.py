from datetime import datetime
StartTime = datetime.now()
import os
import tensorflow as tf
import cv2
import numpy as np
import csv

# Load the model
model_path = 'models/Minecraft-Seed-Finder.h5'
model = tf.keras.models.load_model(model_path)

# Function to preprocess the input image
def load_and_preprocess_image(image_path):
    # Read the image
    img = cv2.imread(image_path)
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Resize the image to the input shape of the model
    img = cv2.resize(img, (256, 256))
    # Normalize the image
    img = img.astype('float32') / 255.0
    # Add a batch dimension
    img = np.expand_dims(img, axis=0)
    return img

# Folder containing the images
folder_path = 'imageclassifcation'  # Change this to your folder path
output_csv_path = 'predictions.csv'  # Output CSV file

# Prepare CSV file
with open(output_csv_path, mode='w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['Filename', 'Predicted Class'])  # Header

    # Process each image in the folder
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(folder_path, filename)
            input_image = load_and_preprocess_image(image_path)

            # Make a prediction
            prediction = model.predict(input_image)
            predicted_class = (prediction > 0.5).astype(int)  # Thresholding for binary classification

            # Determine the predicted class
            if predicted_class[0][0] == 1:
                predicted_label = "Persons"
            else:
                predicted_label = "Desert"

            # Write the prediction to the CSV file
            csv_writer.writerow([filename, predicted_label, prediction])

# Print benchmark time
print(f'Benchmark Time was: {(datetime.now() - StartTime)}')