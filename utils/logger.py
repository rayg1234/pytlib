import ujson as json
from utils.dict_utils import set_deep,get_deep

# Singleton class for logging
class Logger:
    instance = None

    class __Logger:
        def __init__(self,output_file):
            self.output_file = output_file
            # raise exception here
            self.cur_line = {}

        def dump_line(self):
            with open(self.output_file,'a') as f:
                f.write(json.dumps(self.cur_line))
                f.write('\n')

        def set(self,key,value):
            set_deep(self.cur_line,key,value)

    def __init__(self,output_file):
        if not Logger.instance:
            Logger.instance = Logger.__Logger(output_file)

    def __getattr__(self,name):
        return getattr(self.instance,name)
