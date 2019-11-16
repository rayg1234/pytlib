from __future__ import print_function
from future import standard_library
standard_library.install_aliases()
from builtins import range
from threading import Thread
import queue
from data_loading.loaders.loader import Loader
from interface import Interface, implements
import time

def worker_function(worker,queue):
    print('Starting Load Worker {}'.format(id(worker)))
    while True:
        next_sample = next(worker)
        queue.put(next_sample)
    return

class MultiLoader(implements(Loader)):
    def __init__(self,loader,loader_args,num_procs=10,max_queue_size=50):
        self.loader = loader
        self.loader_args = loader_args
        self.num_procs = num_procs
        self.max_queue_size = max_queue_size
        self.result_queue = queue.Queue(self.max_queue_size)
        self.workers = []
        self.threads = []

        for i in range(0,self.num_procs):
            # todo, add random seed here and make the loader creation process parallel
            self.workers.append(self.loader(**self.loader_args))
            p = Thread(target=worker_function,args=((self.workers[-1],self.result_queue)))
            self.threads.append(p)
            p.daemon = True
            p.start()

    def __next__(self):
        # while self.result_queue.empty():
        #     print 'Waiting for queue to fill'
            # time.sleep(0.001)
        return self.result_queue.get()

    def create_sample(self,output,target):
        return loader.create_sample(output,target)
        
    def queue_size(self):
        return self.result_queue.qsize()