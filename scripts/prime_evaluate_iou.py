import os
import os.path as osp
import cv2
import numpy as np
import matplotlib.pyplot as plt
from statistics import mean, stdev
import torch
import fcvision.utils.pytorch_utils as ptu
import fcvision.utils.run_utils as ru
from fcvision.utils.arg_utils import parse_yaml

# Dummy function for model prediction (replace this with your actual model prediction code)
def run_model_on_image(image):
    pred = np.random.randint(0, 2, size=image.shape[:2])  # Random binary mask
    return pred

# Function to calculate Intersection over Union (IoU)
def calculate_iou(pred, target):
    intersection = np.logical_and(pred, target).sum()
    union = np.logical_or(pred, target).sum()
    if union == 0:
        return 0.0  # Prevent division by zero
    iou = intersection / union
    return iou

def evaluate_iou(data_folder):
    cfg, params = parse_yaml(osp.join("cfg", "apps", "prime_test_config.yaml"))
    model = params["model"]
    images_folder = os.path.join(data_folder, 'images')
    targets_folder = os.path.join(data_folder, 'targets')

    # Get all image and target files
    image_files = sorted(os.listdir(images_folder))
    target_files = sorted(os.listdir(targets_folder))

    # Ensure that the number of images matches the number of targets
    assert len(image_files) == len(target_files), "Mismatch in number of images and targets."

    # Thresholds to loop through [0, 0.1, 0.2, ..., 1.0]
    thresholds = np.arange(0, 1.1, 0.1)

    avg_ious = []  # Store average IoUs for each threshold
    std_ious = []  # Store standard deviation of IoUs for each threshold

    # Variables to track the max IoU and corresponding threshold
    max_iou = -1  # Start with a very low value
    best_threshold = 0

    for threshold in thresholds:
        iou_list = []
        
        # Loop through each image and target pair
        for img_file, tgt_file in zip(image_files, target_files):
            img_path = os.path.join(images_folder, img_file)
            tgt_path = os.path.join(targets_folder, tgt_file)
            
            im = cv2.imread(img_path)
            print(f"IM FILE: {img_path}")
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            target = cv2.imread(tgt_path)[...,0]
            original_height, original_width, _ = im.shape
            im = cv2.resize(im, (960, 600))
            target = cv2.resize(target, (960, 600))
            
            im = np.transpose(im, (2, 0, 1))
            if im.max() > 1.0:
                im = im / 255.0
            if target.max() > 1.0:
                target = target / 255.0
            if len(target.shape) == 2:
                target = target[np.newaxis, :, :]
            else:
                target = np.transpose(target, (2, 0, 1))
            im = torch.unsqueeze(ptu.torchify(im), 0).cuda()
            
            with torch.no_grad():
                pred = torch.sigmoid(model(im))[0, 0].cpu().numpy()
            pred = (pred * 255).astype(np.uint8)
            target = (target * 255)[0].astype(np.uint8)
            im = np.transpose(im.cpu().numpy()[0], (1, 2, 0))
            pred = cv2.resize(pred, (original_width, original_height))
            target = cv2.resize(target, (original_width, original_height))
            im = cv2.resize(im, (original_width, original_height))

            # Modify pred with the current threshold
            pred = pred > (threshold * 255)
            ret, target = cv2.threshold(target, 127, 255, cv2.THRESH_BINARY)
            target = target.astype(bool)

            # Calculate IoU
            iou = calculate_iou(pred, target)
            iou_list.append(iou)
        
        # Calculate average IoU and standard deviation for this threshold
        avg_iou = mean(iou_list)
        std_iou = stdev(iou_list) if len(iou_list) > 1 else 0.0

        avg_ious.append(avg_iou)
        std_ious.append(std_iou)

        # Check if the current IoU is the maximum
        if avg_iou > max_iou:
            max_iou = avg_iou
            best_threshold = threshold

        print(f"Threshold {threshold}: Avg IoU = {avg_iou:.4f}, Std IoU = {std_iou:.4f}")

    # Plot average IoU and standard deviation vs. thresholds
    plot_results(thresholds, avg_ious, std_ious)

    # Print max IoU and corresponding threshold
    print(f"\nMax IoU: {max_iou:.4f} at Threshold: {best_threshold}")

def plot_results(thresholds, avg_ious, std_ious):
    # Plot Average IoU
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(thresholds, avg_ious, marker='o', color='b')
    plt.title('Average IoU vs. Threshold')
    plt.xlabel('Threshold')
    plt.ylabel('Average IoU')

    # Plot Standard Deviation of IoU
    plt.subplot(1, 2, 2)
    plt.plot(thresholds, std_ious, marker='o', color='r')
    plt.title('Standard Deviation of IoU vs. Threshold')
    plt.xlabel('Threshold')
    plt.ylabel('Standard Deviation of IoU')

    # Show both plots
    plt.tight_layout()
    plt.show()

# Example usage
data_folder = "/home/lawrence/U-Net/prime_data"  # Replace with the path to your data folder
evaluate_iou(data_folder)
