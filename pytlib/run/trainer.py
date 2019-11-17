from __future__ import print_function
from __future__ import absolute_import
from builtins import str
from builtins import map
from builtins import range
from builtins import object
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
from visualization.graph_visualizer import compute_graph
from visualization.image_visualizer import ImageVisualizer
from utils.batcher import Batcher
from run_utils import load,save,load_samples
from utils.directory_tools import mkdir
from utils.memory import Memory

class Trainer(object):
    def __init__(self,model,args):
        self.model = model
        self.args = args
        self.iteration = 0
        self.memory = Memory()

        if self.args.override or not os.path.isdir(self.args.output_dir) or self.args.output_dir=='tmp':
            mkdir(self.args.output_dir,wipe=True)     

        # initialize logging and model saving
        if self.args.output_dir is not None:
            self.logger = Logger(os.path.join(self.args.output_dir,'train_log.json'))
        else:
            self.logger = Logger()

    # a wrapper for model.forward to feed inputs as list and get outputs as a list
    def evaluate_model(self,inputs):
        output = self.model.get_model().forward(*inputs)
        return list(output) if isinstance(output,tuple) else [output] 

    def train(self):
        # load after a forward call for dynamic models
        batched_data,_,_ = load_samples(self.model.get_loader(),self.model.cuda,self.args.batch_size)
        self.evaluate_model(batched_data)
        self.iteration = load(self.args.output_dir,self.model.get_model(),self.iteration,self.model.get_optimizer())

        for i in range(self.iteration,self.iteration+self.args.iterations):
            #################### LOAD INPUTS ############################
            # TODO, make separate timer class if more complex timings arise
            t0 = time.time()
            batched_data,batched_targets,sample_array = load_samples(self.model.get_loader(),self.model.cuda,self.args.batch_size)
            self.logger.set('timing.input_loading_time',time.time() - t0)
            #############################################################

            #################### FORWARD ################################
            t1 = time.time()
            outputs = self.evaluate_model(batched_data)
            self.logger.set('timing.foward_pass_time',time.time() - t1)
            #############################################################

            #################### BACKWARD AND SGD  #####################
            t2 = time.time()
            loss = self.model.get_lossfn()(*(outputs + batched_targets))
            self.model.get_optimizer().zero_grad()
            loss.backward()
            self.model.get_optimizer().step()
            self.logger.set('timing.loss_backward_update_time',time.time() - t2)
            #############################################################

            #################### LOGGING, VIZ and SAVE ###################
            print('iteration: {0} loss: {1}'.format(self.iteration,loss.data.item()))

            if self.args.compute_graph and i==self.iteration:
                compute_graph(loss,output_file=os.path.join(self.args.output_dir,self.args.compute_graph))

            if self.iteration%self.args.save_iter==0:
                save(self.model.get_model(),self.model.get_optimizer(),self.iteration,self.args.output_dir)

            self.logger.set('time',time.time())
            self.logger.set('date',str(datetime.now()))
            self.logger.set('loss',loss.data.item())
            self.logger.set('iteration',self.iteration)
            self.logger.set('resident_memory',str(self.memory.resident(scale='mB'))+'mB')
            self.logger.dump_line()
            self.iteration+=1

            if self.args.visualize_iter>0 and self.iteration%self.args.visualize_iter==0:
                Batcher.debatch_outputs(sample_array,outputs)
                list([x.visualize({'title':random_str(5)}) for x in sample_array])
                ImageVisualizer().dump_image(os.path.join(self.args.output_dir,'visualizations_{0:08d}.svg'.format(self.iteration)))
            #############################################################


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('-t','--train_config',required=True,type=str,help='the train configuration')
    parser.add_argument('-b','--batch_size',default=1, required=False,type=int,help='the batch_size')
    parser.add_argument('-i','--iterations',required=False, type=int, help='the number of iterations', default=1)
    parser.add_argument('-v','--visualize_iter',required=False, default=1000,type=int, help='save visualizations every this many iterations')
    parser.add_argument('-o','--output_dir',required=False,type=str,default='tmp',help='the directory to output the model params and logs')
    parser.add_argument('-s','--save_iter',type=int,help='save params every this many iterations',default=5000)
    parser.add_argument('-r','--override',action='store_true',help='if override, the directory will be wiped, otherwise resume from the current dir')
    parser.add_argument('-e','--seed',type=int,help='the random seed for torch',default=123)
    parser.add_argument('-g','--compute_graph',default='cgraph',type=str,help='generate the computational graph on the first iteration and write to this file')
    args=parser.parse_args()

    print("Loading Configuration ...")
    config_file = imp.load_source('train_config', args.train_config)
    args.cuda = config_file.train_config.cuda
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    trainer = Trainer(config_file.train_config,args)

    print("Starting Training ...")
    trainer.train()
