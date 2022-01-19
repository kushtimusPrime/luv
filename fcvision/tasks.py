from fcvision.losses import *

def get_task_parameters(params):
	if params['task'] == "cloth":
		params['loss'] = KPVectorLoss()
		params['num_classes'] = 1
		raise NotImplementedError # temporarily deprecated

	elif params['task'] == "cable_endpoints":
		params['num_classes'] = 2 # vector output as well
		params['loss'] = KPVectorLoss()
	else:
		raise Exception("Task not supported.")
