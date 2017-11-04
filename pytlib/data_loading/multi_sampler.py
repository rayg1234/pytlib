from threading import Thread
import Queue
from data_loading.samplers.sampler import Sampler
import time

def worker_function(worker,queue):
    print 'Starting Load Worker {}'.format(id(worker))
    while True:
        next_sample = worker.next()
        queue.put(next_sample)
        # print 'queue size: {}'.format(queue.qsize())
        # time.sleep(0.001)
    return

class MultiSampler(Sampler):
    def __init__(self,loader,loader_args,max_queue_size=50,num_procs=8):
        Sampler.__init__(self,None)
        self.max_queue_size = max_queue_size
        self.result_queue = Queue.Queue(max_queue_size)
        self.workers = []
        self.threads = []

        for i in range(0,num_procs):
            # todo, add random seed here
            self.workers.append(loader(**loader_args))
            p = Thread(target=worker_function,args=((self.workers[-1],self.result_queue)))
            self.threads.append(p)
            p.start()

    def next(self):
        # print self.result_queue.qsize()
        # print 'empty {}'.format(self.result_queue.empty())
        # while self.result_queue.empty():
        #     print 'Waiting for queue to fill'
            # time.sleep(0.001)
        return self.result_queue.get()
        
    # def queue_size(self):
    #     print 'results_queue length {}'.format(len(self.result_queue))

