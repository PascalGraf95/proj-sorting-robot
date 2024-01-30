import json
import os
import cv2
path_Dataset = 'Scanned Data'
path_Json = 'Dataset.json'

# List of all Data in Dataset
file_names = os.listdir(path_Dataset)

# Load Data from Path
with open(path_Json, 'r') as file:
    loaded_data = json.load(file)

# Extract all Names and Length of Images
image_names_array = list(loaded_data.keys())
length_array = list(loaded_data.values())

# Load Images with CV2
images = []
for i in range(len(image_names_array)):
    image = cv2.imread(path_Dataset + '\\' + image_names_array[i])

    if image is not None:
        images.append(image)

# Output of Data
print("Names of Images: ", image_names_array)
print("Length: ", length_array)
print("Image Count: ", len(images))
