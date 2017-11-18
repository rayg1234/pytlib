import torch
import numpy as np
from  data_loading.sample import Sample
from torch.autograd import Variable

class Batcher:
	# turns an array of inputs into a batched Variables
	# assumes cudnn (BCHW) ordering
	# input is a list of list -> [[x1,x2,x3],[y1,y2,y3]]
	@staticmethod
	def batch(inputs):
		assert isinstance(inputs,list) and isinstance(inputs[0],list)
		return [Variable(torch.stack(x,0)) for x in inputs]

	# turns a batched output into an array of outputs
	@staticmethod
	def debatch(outputs):
		assert isinstance(outputs,list)
		return [map(torch.squeeze,torch.chunk(x.data,x.size(0),0)) for x in outputs]

	# turns an array of samples into a batch of inputs and targets
    # for each s0 in sample_array -> [s0,s1,...,sn] 
	# s0 -> [data -> [d0,d1,...,dn], target -> [t0,t1,...,tn]]
	@staticmethod
	def batch_samples(sample_array):
		data_list = list(map(lambda x: x.get_data(), sample_array))
		target_list = list(map(lambda x: x.get_target(), sample_array))
		return Batcher.batch(map(list,zip(*data_list))), Batcher.batch(map(list,zip(*target_list)))

	# store the batched outputs in the corresponding sample array
	@staticmethod
	def debatch_outputs(sample_array,batched_outputs):
		output_array = Batcher.debatch(batched_outputs)
		output_array = map(list,zip(*output_array))
		assert len(output_array)==len(sample_array), 'sample array size {} is the not the same as output_array {}!'.format(len(sample_array),len(output_array))
		for i in range(0,len(output_array)):
			sample_array[i].set_output(output_array[i])