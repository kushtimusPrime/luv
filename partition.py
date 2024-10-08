import os
import random
import shutil

print(os.listdir())

# Define paths to image and label folders
IMAGE_FOLDER = "../3_25/images"
LABEL_FOLDER = "../3_25/targets"

# Get list of image files
image_files = os.listdir(IMAGE_FOLDER)

# Define paths to training and validation folders
TRAIN_FOLDER = "/home/kushtimusprime/U-Net/3_25/train"
VAL_FOLDER = "/home/kushtimusprime/U-Net/3_25/val"

if os.path.exists(TRAIN_FOLDER):
    os.rmdir(TRAIN_FOLDER)

if os.path.exists(VAL_FOLDER):
    os.rmdir(VAL_FOLDER)

if not os.path.exists(TRAIN_FOLDER):
    os.mkdir(TRAIN_FOLDER)

if not os.path.exists(VAL_FOLDER):
    os.mkdir(VAL_FOLDER)

# Define the split ratio (90-10 split)
TRAIN_RATIO = 0.85
VALID_RATIO = 0.15


# Shuffle the list of image files
random.shuffle(image_files)

# Calculate the number of images for training and validation
total_images = len(image_files) - 500
print(f"total images: {total_images}")
num_train_images = int(total_images * TRAIN_RATIO)
num_valid_images = total_images - num_train_images

# Counter for copied images
train_counter = 0
valid_counter = 0

# Function to copy files and create necessary directories
def copy_files(source_folder, dest_folder, files):
    # Create destination directories if they don't exist
    os.makedirs(os.path.join(dest_folder, "images"), exist_ok=True)
    os.makedirs(os.path.join(dest_folder, "targets"), exist_ok=True)
    
    # Copy files to destination folder
    for file in files:
        if file.endswith(".png"):
            label_file = "lb_" + file.split("_")[1]
            shutil.copy(os.path.join(source_folder, file), os.path.join(dest_folder, "images", file))
            shutil.copy(os.path.join(LABEL_FOLDER, label_file), os.path.join(dest_folder, "targets", label_file))

# Copy images and labels to training folder
copy_files(IMAGE_FOLDER, TRAIN_FOLDER, image_files[:num_train_images])

# Copy images and labels to validation folder
copy_files(IMAGE_FOLDER, VAL_FOLDER, image_files[num_train_images:])

print("Training images:", num_train_images)
print("Validation images:", num_valid_images)

