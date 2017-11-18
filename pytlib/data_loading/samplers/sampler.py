from interface import Interface, implements

class Sampler(Interface):
    def __init__(self,source,params):
    	pass

    # returns a sample object
    def next(self):
        pass
