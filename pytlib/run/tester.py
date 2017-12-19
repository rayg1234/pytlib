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
from utils.random_utils import random_str
from utils.batcher import Batcher
from visualization.image_visualizer import ImageVisualizer
from run_utils import load,save,load_samples

class Tester:
    def __init__(self,model,args):
        self.model = model
        self.args = args
        self.iteration = 0
        
        # initialize logging and model saving
        if self.args.output_dir is not None:
            self.logger = Logger(os.path.join(self.args.output_dir,'infer_log.json'))
        else:
            self.logger = Logger()

    def evaluate_model(self,inputs):
        output = self.model.get_model().infer(*inputs)
        return list(output) if isinstance(output,tuple) else [output] 

    def test(self):
        # load after a forward call for dynamic models
        batched_data,_,_ = load_samples(self.model.get_loader(),self.model.cuda,self.args.batch_size)
        self.evaluate_model(batched_data)
        self.iteration = load(self.args.output_dir,self.model.get_model(),self.iteration)

        for i in range(self.iteration,self.iteration+self.args.iterations):
            #################### LOAD INPUTS ############################
            t0 = time.time()
            batched_data,batched_targets,sample_array = load_samples(self.model.get_loader(),self.model.cuda,self.args.batch_size)
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
                ImageVisualizer().dump_image(os.path.join(self.args.output_dir,'testviz_{0:08d}.svg'.format(self.iteration)))

            #############################################################


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('-t','--test_config',required=True,type=str,help='the train configuration')
    parser.add_argument('-b','--batch_size',default=1, required=False,type=int,help='the batch_size')
    parser.add_argument('-i','--iterations',required=False, type=int, help='the number of iterations', default=1)
    parser.add_argument('-v','--visualize_iter',required=False, default=1,type=int, help='save visualizations every this many iterations')
    parser.add_argument('-o','--output_dir',required=False,type=str,default='tmp',help='the directory where the model weights are')
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
