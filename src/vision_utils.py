from scipy import ndimage as ndi
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max
from skimage import data, img_as_float


def find_peaks(im):
	return peak_local_max(im, min_distance=20, threshold_abs=0.7)
