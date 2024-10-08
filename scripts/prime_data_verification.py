import os
import cv2
import numpy as np

def browse_images_and_targets(output_folder):
    images_folder = os.path.join(output_folder, 'images')
    targets_folder = os.path.join(output_folder, 'targets')
    
    # Get all image and target files
    image_files = sorted(os.listdir(images_folder))
    target_files = sorted(os.listdir(targets_folder))
    import pdb
    pdb.set_trace()
    # Make sure the number of images matches the number of targets
    assert len(image_files) == len(target_files), "Mismatch in number of images and targets."

    current_index = 0  # Start viewing from the first image/target pair

    while True:
        # Load the current image and corresponding target
        img_file = image_files[current_index]
        target_file = target_files[current_index]

        img_path = os.path.join(images_folder, img_file)
        target_path = os.path.join(targets_folder, target_file)

        # Read the image and target
        im = cv2.imread(img_path)
        pred = cv2.imread(target_path, cv2.IMREAD_GRAYSCALE)

        # Create overlay
        overlayed = np.copy(im)
        overlayed[pred > 0.5 * 255] = [255, 0, 0]  # Overlay in blue
        overlayed = (overlayed).astype(np.uint8)  # Scale to 255

        # Display the overlayed image
        cv2.imshow("Overlayed Image", overlayed)

        # Wait for user input
        key = cv2.waitKey(0)

        # Left arrow key: go to the previous image
        if key == 81:  # Left arrow key
            if current_index > 0:
                current_index -= 1
            else:
                print("Reached the beginning of the folder.")
                break

        # Right arrow key: go to the next image
        elif key == 83:  # Right arrow key
            if current_index < len(image_files) - 1:
                current_index += 1
            else:
                print("Reached the end of the folder.")
                break

        # 'd' key: delete the current image and target
        elif key == ord('d'):
            print(f"Deleting {img_file} and {target_file}...")

            # Remove the image and target files
            os.remove(img_path)
            os.remove(target_path)

            # Remove from lists
            del image_files[current_index]
            del target_files[current_index]

            # Move to the previous image if the last one is deleted
            if current_index >= len(image_files):
                current_index = len(image_files) - 1

            # If there are no images left, exit the loop
            if len(image_files) == 0:
                print("No more images left.")
                break

        # Escape key: exit the viewer
        elif key == 27:  # Escape key
            break

    cv2.destroyAllWindows()

# Example usage
output_folder = "/home/lawrence/U-Net/prime_data"  # Replace with the path to your output folder
browse_images_and_targets(output_folder)
