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
		params['dataset'] = KPDataset()
		params['dataset_val'] = KPDataset(val=True)
	elif params['task'] == "cable_vecs":
		params['num_classes'] = 2 # vector output as well
		params['loss'] = KPVectorLoss()
		params['dataset'] = KPVectorDataset()
		params['dataset_val'] = KPVectorDataset(val=True)
	elif params['task'] == "cable_kp_vecs":
		params['num_classes'] = 2
		params['loss'] = SegmentationLoss()
		params['dataset'] = KPVectorDataset(dataset_dir="data/cable_vecs2")
		params['dataset_val'] = KPVectorDataset(dataset_dir="data/cable_vecs2", val=True)

	else:
		raise Exception("Task not supported.")
	return params
