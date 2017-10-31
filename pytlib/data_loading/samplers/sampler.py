class Sampler:

    def __init__(self,source):
        self.source = source

    # returns a sample object
    def next(self):
        raise NotImplementedError