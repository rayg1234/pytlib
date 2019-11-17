from __future__ import division
from builtins import object
from past.utils import old_div
import ujson as json
from utils.dict_utils import set_deep,get_deep
import sys

# Singleton class for logging
class Logger(object):
    instance = None

    class __Logger(object):
        def __init__(self,output_file):
            self.output_file = output_file
            # raise exception here
            self.cur_line = {}
            self.counter = {}

        def dump_line(self):
            line = json.dumps(self.cur_line)
            if self.output_file is None:
                 sys.stdout.write(line)
                 sys.stdout.write('\n')
            else:
                with open(self.output_file,'a') as f:
                    f.write(line)
                    f.write('\n')
            self.cur_line = {}
            self.counter = {}

        def set(self,key,value):
            set_deep(self.cur_line,key,value)

        def average(self,key,value):
            cur_value = get_deep(self.cur_line,key,0)
            cur_count = get_deep(self.counter,key,0)
            set_deep(self.cur_line,key,old_div((float(cur_value)*cur_count+value),(cur_count+1)))
            set_deep(self.counter,key,cur_count+1)           

    def __init__(self,output_file=None):
        if not Logger.instance:
            Logger.instance = Logger.__Logger(output_file)

    def __getattr__(self,name):
        return getattr(self.instance,name)
