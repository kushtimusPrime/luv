import numpy as np
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max
from skimage import data, img_as_float
from scipy import ndimage
import cv2
from matplotlib.patches import Circle


def find_peaks(im):
    return peak_local_max(im, min_distance=20, threshold_abs=0.05)  # 0.1 #0.25 0.7


def find_center_of_mass(im):
    """
    Finds the center of mass of a binary image.
    """
    return ndimage.measurements.center_of_mass(im)


def closest_nonzero_pt(im, target_pt):
    """
    Returns the location of the closest nonzero point in IM to TARGET_PT.
    """
    nonzero_idx = np.vstack(im.nonzero()).T
    displacements = nonzero_idx - target_pt
    displacements = np.linalg.norm(displacements, axis=1)
    closest_idx = np.argmin(displacements)
    closest_pt = nonzero_idx[closest_idx]
    return closest_pt


def get_valid_depth_mask(color_mask, depth_mask):
    """
    Returns COLOR_MASK, where points with 0 (undefined) depth in
    DEPTH_MASK are set to 0.
    """
    depth_mask = depth_mask > 0
    return np.multiply(color_mask, depth_mask)


def get_cable_mask(im):
    gray = im[:, :, 0]
    # plt.imshow(gray); plt.show()
    mask = np.where(gray > 100, 1.0, 0.0)  # masking the rope
    mask[700:, 700:] = 0
    mask[650:, :250] = 0
    return mask


def get_highest_depth_pt_within_radius(
    depth_img, nominal_pt, radius=20, depth_mask=None
):
    """
    Finds the highest depth point within an l1 ball of radius RADIUS centered
    at NOMINAL_PT

    """
    depth_copy = np.copy(depth_img)
    # plt.imshow(depth_copy); plt.show()
    depth_copy[: max(0, nominal_pt[0] - radius)] = 0
    depth_copy[min(depth_img.shape[0] - 1, nominal_pt[0] + radius) :] = 0
    depth_copy[:, : max(0, nominal_pt[1] - radius)] = 0
    depth_copy[:, min(depth_img.shape[1] - 1, nominal_pt[1] + radius) :] = 0

    if depth_mask is not None:
        depth_copy = np.multiply(depth_mask, depth_copy)

    depth_copy[depth_copy == 0] = 2

    pt = np.unravel_index(np.argmin(depth_copy), depth_copy.shape)
    # print(pt)
    # plt.imshow(depth_copy); plt.show()

    return pt


def get_shake_point(phoxi_im, vis=False, random=False):
    """
    Takes in a [h, w, 4] dimension image from the phoxi and outputs
    a point on the cable mask with depth information closest to the
    cable mask's center of mask.
    """
    phoxi_im = phoxi_im.copy()
    phoxi_im[550:] = 0
    phoxi_im[:, :300] = 0

    color_mask = get_cable_mask(phoxi_im[:, :, :3])

    depth_mask = phoxi_im[:, :, 3] > 0
    com = (
        find_center_of_mass(color_mask)
        if not random
        else np.random.choice(np.concatenate(np.nonzero(color_mask), axis=-1))
    )
    valid_depth_mask = get_valid_depth_mask(color_mask, depth_mask)
    # plt.imshow(valid_depth_mask); plt.show()
    closest = closest_nonzero_pt(valid_depth_mask, com)
    closest = get_highest_depth_pt_within_radius(
        phoxi_im[:, :, 3],
        closest,
        radius=100 if not random else 5,
        depth_mask=valid_depth_mask,
    )
    if vis:
        print(closest)
        fig, ax = plt.subplots()
        plt.imshow(color_mask)

        # circ = Circle((closest[1], closest[0]), 1, color='r')
        # ax.add_patch(circ)
        plt.scatter(*closest[::-1])
        plt.show()
    return (int(closest[1]), int(closest[0]))


if __name__ == "__main__":
    from fcvision.phoxi import Phoxi

    cam = Phoxi()
    im = cam.capture()._data
    pt = get_shake_point(im, vis=True)

    # a = np.zeros([5, 5])
    # a[0] = 1
    # a[3, 4] = 1
    # closest = closest_nonzero_pt(a, [2, 3])
    # print(closest)
    # print(a)
