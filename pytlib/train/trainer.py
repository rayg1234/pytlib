# this is needed to make matplotlib work without explicitly connect to X
import matplotlib 
matplotlib.use('svg')

import torch
from image.visualization import visualize_ptimage_array
from image.ptimage import PTImage
import argparse
import imp
import os
import time
from datetime import datetime
from utils.logger import Logger
from utils.debug import pp
from utils.directory_tools import mkdir, list_files
from utils.random_utils import random_str
from train.batcher import Batcher
from visualization.graph_visualizer import compute_graph
from visualization.image_visualizer import ImageVisualizer

class Trainer:

    def __init__(self,model_config,args):
        self.model_config = model_config

        # should use strings, namespace or functions here?
        self.loader = self.model_config.loader
        self.model = self.model_config.model
        self.lossfn = self.model_config.loss
        self.optimizer = self.model_config.optimizer
        self.args = args
        self.iteration = 0
        self.first_iteration = True

        if self.args.override or not os.path.isdir(self.args.output_dir) or self.args.output_dir=='tmp':
            mkdir(self.args.output_dir,wipe=True)
        else:
            self.load()        

        # initialize logging and model saving
        if self.args.output_dir is not None:
            self.logger = Logger(os.path.join(self.args.output_dir,'train_log.json'))
        else:
            self.logger = Logger()

    def save(self):
        state = {}
        state['iteration']=self.iteration+1
        state['state_dict']=self.model.state_dict()
        state['optimizer']=self.optimizer.state_dict()
        with open(os.path.join(self.args.output_dir,'model_{0:08d}.mdl'.format(self.iteration)),'wb') as f:
            torch.save(state,f)

    def load(self):
        # list model files and find the latest_model
        all_models = list_files(self.args.output_dir,ext_filter='.mdl')
        if not all_models:
            print 'No previous checkpoints found!'
            return

        all_models_indexed = [(m,int(m.split('.mdl')[0].split('_')[-1])) for m in all_models]
        all_models_indexed.sort(key=lambda x: x[1],reverse=True)
        print 'Loading model from disk: {0}'.format(all_models_indexed[0][0])
        checkpoint = torch.load(all_models_indexed[0][0])
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.iteration = checkpoint['iteration']

    # a wrapper for model.forward to feed inputs as list and get outputs as a list
    def evaluate_model(self,inputs):
        output = self.model(*inputs)
        return list(output) if isinstance(output,tuple) else [output] 

    def train(self):
        for i in range(self.iteration,self.iteration+self.args.iterations):
            #################### LOAD INPUTS ############################
            # TODO, make separate timer class if more complex timings arise
            t0 = time.time()
            sample_array = [self.loader.next() for i in range(0,args.batch_size)]
            batched_data, batched_targets = Batcher.batch_samples(sample_array)
            if self.args.cuda:
                batched_data = map(lambda x: x.cuda(), batched_data)
                batched_targets = map(lambda x: x.cuda(), batched_targets)
            self.logger.set('timing.input_loading_time',time.time() - t0)
            #############################################################

            #################### FORWARD ################################
            t1 = time.time()
            outputs = self.evaluate_model(batched_data)
            self.logger.set('timing.foward_pass_time',time.time() - t1)
            #############################################################

            #################### BACKWARD AND SGD  #####################
            t2 = time.time()
            loss = self.lossfn(*(outputs + batched_targets))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.logger.set('timing.loss_backward_update_time',time.time() - t2)
            #############################################################

            #################### LOGGING, VIZ and SAVE ###################
            print 'iteration: {0} loss: {1}'.format(self.iteration,loss.data[0])

            # TODO Move this to a function in the model, ie the model should know how to draw its own graph
            # if self.args.compute_graph and self.first_iteration:
            #     compute_graph(output_data,output_file=os.path.join(self.args.output_dir,self.args.compute_graph))

            if self.iteration%self.args.save_iter==0:
                self.save()

            self.logger.set('time',time.time())
            self.logger.set('date',str(datetime.now()))
            self.logger.set('loss',loss.data[0])
            self.logger.set('iteration',self.iteration)
            self.logger.dump_line()
            self.iteration+=1

            if self.args.visualize_iter>0 and self.iteration%self.args.visualize_iter==0:
                Batcher.debatch_outputs(sample_array,outputs)
                map(lambda x:x.visualize({'title':random_str(5)}),sample_array)
                ImageVisualizer().dump_image(os.path.join(self.args.output_dir,'visualizations_{0:08d}.svg'.format(self.iteration)))

            self.first_iteration = False
            #############################################################


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('-t','--train_config',required=True,type=str,help='the train configuration')
    parser.add_argument('-b','--batch_size',default=1, required=False,type=int,help='the batch_size')
    parser.add_argument('-i','--iterations',required=False, type=int, help='the number of iterations', default=1)
    parser.add_argument('-v','--visualize_iter',required=False, default=100,type=int, help='save visualizations every this many iterations')
    parser.add_argument('-o','--output_dir',required=False,type=str,default='tmp',help='the directory to output the model params and logs')
    parser.add_argument('-s','--save_iter',type=int,help='save params every this many iterations',default=1000)
    parser.add_argument('-r','--override',action='store_true',help='if override, the directory will be wiped, otherwise resume from the current dir')
    parser.add_argument('-e','--seed',type=int,help='the random seed for torch',default=123)
    parser.add_argument('-g','--compute_graph',default='cgraph',type=str,help='generate the computational graph on the first iteration and write to this file')
    args=parser.parse_args()

    print "Loading Configuration ..."
    config_file = imp.load_source('train_config', args.train_config)
    args.cuda = config_file.train_config.cuda
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    trainer = Trainer(config_file.train_config,args)

    print "Starting Training ..."
    trainer.train()
