from fcvision.losses import *
from fcvision.dataset import KPDataset, StereoSegDataset, KPVectorDataset

def get_task_parameters(params):
	if params['task'] == "cloth":
		params['loss'] = KPVectorLoss()
		params['num_classes'] = 1
		raise NotImplementedError # temporarily deprecated

	elif params['task'] == "cable_endpoints":
		params['num_classes'] = 1 # vector output as well
		params['loss'] = KPVectorLoss()
		params['dataset'] = KPDataset()
		params['dataset_val'] = KPDataset(val=True)
	else:
		raise Exception("Task not supported.")
	return params
