import os
import os.path as osp
import matplotlib.pyplot as plt

from fcvision.dataset import KPDataset
from fcvision.arg_utils import parse_args
import fcvision.run_utils as ru
from fcvision.vision_utils import find_peaks
from matplotlib.patches import Circle
from fcvision.phoxi import Phoxi
from fcvision.kp_wrapper import KeypointNetwork


def main():
	params = parse_args()
	logdir = ru.get_file_prefix(params)
	os.makedirs(os.path.join(logdir, 'lightning_logs'))
	cam = Phoxi()
	model = KeypointNetwork(params['checkpoint'], params=params, logdir=logdir)

	for idx in range(100):
		input("Press enter when ready to take a new image.")
		im = cam.capture()
		np_im = im._data[:,:,0] / 255.
		pred = model(im, mode='vis')[0]

		plt.imshow(pred + np_im); plt.show()

		coords = find_peaks(pred)
		fig = plt.figure(frameon=False)
		ax = plt.Axes(fig, [0., 0., 1., 1.])
		ax.set_axis_off()
		fig.add_axes(ax)
		ax.imshow(np_im, aspect='auto')
		for xx, yy in coords:
			circ = Circle((yy, xx), 3, color='r')
			ax.add_patch(circ)
		plt.savefig(osp.join(logdir, "vis", "pred_%d.jpg"%idx))
		plt.show()
		plt.clf()


if __name__ == '__main__':
	main()
