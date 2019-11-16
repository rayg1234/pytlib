from __future__ import print_function
from builtins import range
import torch
import os
from utils.directory_tools import mkdir, list_files
from utils.batcher import Batcher

def save(model,optimizer,iteration,output_dir):
    state = {}
    state['iteration']=iteration+1
    state['state_dict']=model.state_dict()
    state['optimizer']=optimizer.state_dict()
    with open(os.path.join(output_dir,'model_{0:08d}.mdl'.format(iteration)),'wb') as f:
        torch.save(state,f)

def load(output_dir,model,iteration,optimizer=None):
    # list model files and find the latest_model
    all_models = list_files(output_dir,ext_filter='.mdl')
    if not all_models:
        print('No previous checkpoints found!')
        return iteration

    all_models_indexed = [(m,int(m.split('.mdl')[0].split('_')[-1])) for m in all_models]
    all_models_indexed.sort(key=lambda x: x[1],reverse=True)
    print('Loading model from disk: {0}'.format(all_models_indexed[0][0]))
    checkpoint = torch.load(all_models_indexed[0][0])
    model.load_state_dict(checkpoint['state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
    return checkpoint['iteration']

def load_samples(loader,cuda,batch_size):
    sample_array = [next(loader) for i in range(0,batch_size)]
    batched_data, batched_targets = Batcher.batch_samples(sample_array)
    if cuda:
        batched_data = [x.cuda() for x in batched_data]
        batched_targets = [x.cuda() for x in batched_targets]
    return batched_data,batched_targets,sample_array