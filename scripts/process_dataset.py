import os
import os.path as osp
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image


START_ID = 0

INPUT_DIR = "data/cable_images"
OUTPUT_DIR = "data/cable_images_labeled2"


def onclick(event):
	global ix, iy
	ix, iy = event.xdata, event.ydata
	if ix is None or iy is None:
		return None
	global coords
	coords.append((int(ix), int(iy)))
	if len(coords) == 2:
		fig.canvas.mpl_disconnect(cid)

	return coords

def draw_gaussian(mu_x, mu_y, size_x, size_y, sigma_x=12., sigma_y=12.):
	x = np.linspace(0, size_x-1, size_x)
	y = np.linspace(0, size_y-1, size_y)

	x, y = np.meshgrid(x, y)
	z = (1/(2*np.pi*sigma_x*sigma_y) * np.exp(-((x - mu_x)**2/(2*sigma_x**2)
	     + (y - mu_y)**2/(2*sigma_y**2))))
	if z.max() > 0:
		z /= z.max()
	return z




if __name__ == '__main__':

	files = sorted(os.listdir(INPUT_DIR))
	if not osp.exists(OUTPUT_DIR):
		os.mkdir(OUTPUT_DIR)

	for i, f in enumerate(files[START_ID:]):
		print(i)
		coords = []
		im = np.load(osp.join(INPUT_DIR, f))
		fig = plt.figure()
		cid = fig.canvas.mpl_connect('button_press_event', onclick)
		plt.imshow(im[:,:,0])
		mng = plt.get_current_fig_manager()
		mng.full_screen_toggle()
		plt.show()

		target = [np.zeros_like(im[:,:,0])]
		for px, py in coords:
			target_tile = draw_gaussian(px, py, im.shape[1], im.shape[0])
			target.append(target_tile)
		target = np.array(target).max(0)


		# normalize both channels
		gray_im = im[:,:,0] / 255
		depth_im = im[:,:,3]
		normalized_image = np.stack([gray_im, depth_im])

		# save processed data
		np.save(osp.join(OUTPUT_DIR, "target_%d"%i), target)
		np.save(osp.join(OUTPUT_DIR, "image_%d"%i), normalized_image)
		vis_im = Image.fromarray((target + gray_im) * 255).convert("RGB")
		vis_im.save(osp.join(OUTPUT_DIR, "vis_%d.jpg"%i))
		np.save(osp.join(OUTPUT_DIR, "coords_%d"%i), coords)

