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
    # the expected input dimensions to this function is:
    # array(batched output tensor) output_type size -> array(array(output tensor)) output_type size x batch size 
    # OR
    # array(array(batched output tensor)) output_type size x sequence size -> array(array(array(output tensor))) output_type x batch size x sequence size
    @staticmethod
    def debatch(outputs):
        assert isinstance(outputs,list)
        # if outputs is a list of list, we recurse
        result = []
        def debatch_helper(batched_data):
            return map(lambda y: torch.squeeze(y,0),torch.chunk(batched_data.data,batched_data.size(0),0))

        # we want: N results x n_sequence x n_batch
        for x in outputs: # loop over individual outputs
            if isinstance(x,list): # this is sequence data
                # this is an array(array(output tensor)) -> sequence size x batch size
                sequence_batches = map(lambda y: debatch_helper(y),x)
                result.append(map(list,zip(*sequence_batches)))
            else:
                result.append(debatch_helper(x))
        return result
        # return [map(lambda y: torch.squeeze(y,0),torch.chunk(x.data,x.size(0),0)) for x in outputs]

    # turns an array of samples into a batch of inputs and targets
    # for each s0 in sample_array -> [s0,s1,...,sn] 
    # s0 -> [data -> [d0,d1,...,dn], target -> [t0,t1,...,tn]]
    @staticmethod
    def batch_samples(sample_array):
        data_list = list(map(lambda x: x.get_data(), sample_array))
        target_list = list(map(lambda x: x.get_target(), sample_array))
        return Batcher.batch(map(list,zip(*data_list))), Batcher.batch(map(list,zip(*target_list)))

    # store the batched outputs in the corresponding sample array
    # batched outputs have the form [out0*batch_size,out1*batch_size,out2*batch_size...]
    @staticmethod
    def debatch_outputs(sample_array,batched_outputs):
        output_array = Batcher.debatch(batched_outputs)
        # output_type size x batchsize -> batch size x outputtype size
        output_array = map(list,zip(*output_array))
        assert len(output_array)==len(sample_array), 'sample array size {} is the not the same as output_array {}!'.format(len(sample_array),len(output_array))
        for i in range(0,len(output_array)):
            sample_array[i].set_output(output_array[i])