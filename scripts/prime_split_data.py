import os
import shutil
import random

def split_data(data_folder, output_folder, train_ratio=0.8):
    images_folder = os.path.join(data_folder, 'images')
    targets_folder = os.path.join(data_folder, 'targets')

    # Create output directories for train and val
    train_images_folder = os.path.join(output_folder, 'train', 'images')
    train_targets_folder = os.path.join(output_folder, 'train', 'targets')
    val_images_folder = os.path.join(output_folder, 'val', 'images')
    val_targets_folder = os.path.join(output_folder, 'val', 'targets')

    os.makedirs(train_images_folder, exist_ok=True)
    os.makedirs(train_targets_folder, exist_ok=True)
    os.makedirs(val_images_folder, exist_ok=True)
    os.makedirs(val_targets_folder, exist_ok=True)

    # Get all images and targets, assume corresponding image/target pairs have matching names
    image_files = sorted(os.listdir(images_folder))
    target_files = sorted(os.listdir(targets_folder))

    # Ensure the number of images matches the number of targets
    assert len(image_files) == len(target_files), "Mismatch in number of images and targets."

    # Create a list of indices and shuffle them
    indices = list(range(len(image_files)))
    random.shuffle(indices)

    # Split indices into train and validation
    train_size = int(train_ratio * len(indices))
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    # Function to copy files based on indices
    def copy_files(indices, src_img_folder, src_tgt_folder, dest_img_folder, dest_tgt_folder):
        for i in indices:
            img_file = image_files[i]
            tgt_file = target_files[i]
            
            # Copy the image and target to the destination
            shutil.copy(os.path.join(src_img_folder, img_file), os.path.join(dest_img_folder, img_file))
            shutil.copy(os.path.join(src_tgt_folder, tgt_file), os.path.join(dest_tgt_folder, tgt_file))

    # Copy training data
    copy_files(train_indices, images_folder, targets_folder, train_images_folder, train_targets_folder)

    # Copy validation data
    copy_files(val_indices, images_folder, targets_folder, val_images_folder, val_targets_folder)

    print(f"Data split complete: {train_size} images in train and {len(indices) - train_size} images in val.")

# Example usage
data_folder = "/home/lawrence/U-Net/prime_data"  # Replace with your data folder path
output_folder = "/home/lawrence/U-Net/prime_data_split"  # Replace with the path where you want the train/val folders
split_data(data_folder, output_folder)
