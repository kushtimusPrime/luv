import os
import shutil

def create_image_and_target_folders(data_folder, output_folder):
    # Create paths for the new 'images' and 'targets' folders
    images_folder = os.path.join(output_folder, "images")
    targets_folder = os.path.join(output_folder, "targets")
    
    # Create the output directories if they don't exist
    os.makedirs(images_folder, exist_ok=True)
    os.makedirs(targets_folder, exist_ok=True)
    
    # Initialize counters for naming the files
    image_counter = 1
    target_counter = 1
    
    # Loop over each subfolder in the data folder
    for subfolder in os.listdir(data_folder):
        subfolder_path = os.path.join(data_folder, subfolder)
        
        if os.path.isdir(subfolder_path):
            # Inside each subfolder, there will be scenario subfolders
            for scenario_folder in os.listdir(subfolder_path):
                scenario_path = os.path.join(subfolder_path, scenario_folder)

                if os.path.isdir(scenario_path):
                    # Copy 'images_left' and 'images_right' to 'images' folder
                    for folder in ['images_left', 'images_right']:
                        source_folder = os.path.join(scenario_path, folder)
                        if os.path.exists(source_folder):
                            # Sort filenames to ensure ordered processing
                            for img_file in sorted(os.listdir(source_folder)):
                                source_img_path = os.path.join(source_folder, img_file)
                                if os.path.isfile(source_img_path):
                                    # Create a new filename like 'im_00001.png'
                                    new_image_name = f"im_{image_counter:05d}.png"
                                    shutil.copy(source_img_path, os.path.join(images_folder, new_image_name))
                                    image_counter += 1
                    
                    # Copy 'left_masked' and 'right_masked' to 'targets' folder
                    for folder in ['left_masked', 'right_masked']:
                        source_folder = os.path.join(scenario_path, folder)
                        if os.path.exists(source_folder):
                            # Sort filenames to ensure ordered processing
                            for target_file in sorted(os.listdir(source_folder)):
                                source_target_path = os.path.join(source_folder, target_file)
                                if os.path.isfile(source_target_path):
                                    # Create a new filename like 'target_00001.png'
                                    new_target_name = f"target_{target_counter:05d}.png"
                                    shutil.copy(source_target_path, os.path.join(targets_folder, new_target_name))
                                    target_counter += 1

    print("Images and targets have been successfully organized, renamed, and sorted!")

# Example usage
data_folder = "/home/lawrence/U-Net/data"  # Replace with the path to your data folder
output_folder = "/home/lawrence/U-Net/prime_data"  # Replace with the path to where you want to save the new folders

create_image_and_target_folders(data_folder, output_folder)
