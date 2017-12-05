# this is needed to make matplotlib work without explicitly connect to X
import matplotlib 
matplotlib.use('svg')

import torch
from image.ptimage import PTImage
import argparse
import imp
import os
import time
import random
from datetime import datetime
from utils.logger import Logger
from utils.debug import pp
from utils.directory_tools import mkdir, list_files
from utils.random_utils import random_str
from utils.batcher import Batcher
from visualization.image_visualizer import ImageVisualizer

class Tester:
    def __init__(self,model_config,args):
        self.model_config = model_config

        # should use strings, namespace or functions here?
        self.loader = self.model_config.loader
        self.model = self.model_config.model
        self.args = args
        self.iteration = 0
        
        # initialize logging and model saving
        if self.args.model_dir is not None:
            self.logger = Logger(os.path.join(self.args.model_dir,'infer_log.json'))
        else:
            self.logger = Logger()

    def load(self):
        # list model files and find the latest_model
        all_models = list_files(self.args.model_dir,ext_filter='.mdl')
        if not all_models:
            print 'No previous checkpoints found!'
            return
            
        all_models_indexed = [(m,int(m.split('.mdl')[0].split('_')[-1])) for m in all_models]
        all_models_indexed.sort(key=lambda x: x[1],reverse=True)
        print 'Loading model from disk: {0}'.format(all_models_indexed[0][0])
        checkpoint = torch.load(all_models_indexed[0][0])
        self.model.load_state_dict(checkpoint['state_dict'])
        self.iteration = checkpoint['iteration']

    # a wrapper for model.forward to feed inputs as list and get outputs as a list
    def evaluate_model(self,inputs):
        output = self.model.infer(*inputs)
        return list(output) if isinstance(output,tuple) else [output] 

    def load_samples(self):
        sample_array = []
        while len(sample_array)<args.batch_size:
            s = self.loader.next()
            if s is not None:
                sample_array.append(s)
        batched_data, batched_targets = Batcher.batch_samples(sample_array)
        if self.args.cuda:
            batched_data = map(lambda x: x.cuda(), batched_data)
            batched_targets = map(lambda x: x.cuda(), batched_targets)
        return batched_data,batched_targets,sample_array

    def test(self):
        # load after a forward call for dynamic models
        batched_data,_,_ = self.load_samples()
        self.evaluate_model(batched_data)
        self.load()

        for i in range(self.iteration,self.iteration+self.args.iterations):
            #################### LOAD INPUTS ############################
            t0 = time.time()
            batched_data,batched_targets,sample_array = self.load_samples()
            self.logger.set('timing.input_loading_time',time.time() - t0)
            #############################################################

            #################### FORWARD ################################
            t1 = time.time()
            outputs = self.evaluate_model(batched_data)
            self.logger.set('timing.foward_pass_time',time.time() - t1)
            #############################################################

            #################### LOGGING, VIZ ###################
            print 'iteration: {0}'.format(self.iteration)

            self.logger.set('time',time.time())
            self.logger.set('date',str(datetime.now()))
            self.logger.set('iteration',self.iteration)
            self.logger.dump_line()
            self.iteration+=1

            if self.args.visualize_iter>0 and self.iteration%self.args.visualize_iter==0:
                Batcher.debatch_outputs(sample_array,outputs)
                map(lambda x:x.visualize({'title':random_str(5),'mode':'test'}),sample_array)
                ImageVisualizer().dump_image(os.path.join(self.args.model_dir,'testviz_{0:08d}.svg'.format(self.iteration)))

            #############################################################


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('-t','--test_config',required=True,type=str,help='the train configuration')
    parser.add_argument('-b','--batch_size',default=1, required=False,type=int,help='the batch_size')
    parser.add_argument('-i','--iterations',required=False, type=int, help='the number of iterations', default=1)
    parser.add_argument('-v','--visualize_iter',required=False, default=1,type=int, help='save visualizations every this many iterations')
    parser.add_argument('-m','--model_dir',required=False,type=str,default='tmp',help='the directory where the model weights are')
    parser.add_argument('-e','--seed',type=int,help='the random seed for torch',default=123)
    args=parser.parse_args()

    print "Loading Model ..."
    config_file = imp.load_source('test_config', args.test_config)
    args.cuda = config_file.test_config.cuda
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    tester = Tester(config_file.test_config,args)

    print "Starting Inference ..."
    tester.test()
