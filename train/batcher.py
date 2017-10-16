import torch
import numpy as np
from  data_loading.sample import Sample

class Batcher:
	
	# turns an array of samples into a batched input
	# assumes cudnn (BCHW) ordering
	@staticmethod
	def batch(sample_array):
		return torch.stack(sample_array,0)

	# turns a batched output into an array of outputs
	@staticmethod
	def debatch(outputs):
		return torch.chunk(outputs,outputs.shape(0),0)

