from fcvision.losses import *
from fcvision.dataset import KPDataset, StereoSegDataset, KPVectorDataset

def get_task_parameters(params):
	if params['task'] == "cloth":
		params['loss'] = KPVectorLoss()
		params['num_classes'] = 1
		raise NotImplementedError # temporarily deprecated

	elif params['task'] == "cable_endpoints":
		params['num_classes'] = 1
		params['loss'] = SegmentationLoss()
		try:
			params['dataset'] = KPDataset()
			params['dataset_val'] = KPDataset(val=True)
		except:
			pass
	elif params['task'] == "cable_vecs":
		params['num_classes'] = 2 # vector output as well
		params['loss'] = KPVectorLoss()
		try:
			params['dataset'] = KPVectorDataset()
			params['dataset_val'] = KPVectorDataset(val=True)
		except:
			pass
	elif params['task'] == "cable_kp_vecs":
		params['num_classes'] = 2
		params['loss'] = SegmentationLoss()
		try:
			params['dataset'] = KPVectorDataset(dataset_dir="data/cable_kp_vecs")
			params['dataset_val'] = KPVectorDataset(dataset_dir="data/cable_kp_vecs", val=True)
		except:
			pass # in case datasets aren't on the machine (for testing)
	elif params['task'] == "cable_slide":
		params['num_classes'] = 1
		params['loss'] = SegmentationLoss()
		try:
			params['dataset'] = KPDataset(dataset_dir="data/cable_slide")
			params['dataset_val'] = KPDataset(dataset_dir="data/cable_slide", val=True)
		except:
			pass # in case datasets aren't on the machine (for testing)

	else:
		raise Exception("Task not supported.")
	return params
