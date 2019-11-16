from builtins import object
# This is equivalent of the TrainConfiguration for test time
# see TrainConfiguration's comments
class TestConfiguration(object):
    def __init__(self,loader_params,model_params,lossfn,cuda=True):
        self.loader_params = loader_params
        self.model_params = model_params
        self.model = None
        self.loader = None
        self.cuda = cuda

    def get_model(self):
        if self.model is None:
            self.model = self.model_params[0](**self.model_params[1])
            if self.cuda:
                self.model.cuda()
        return self.model

    def get_loader(self):
        if self.loader is None:
            self.loader = self.loader_params[0](**self.loader_params[1])
        return self.loader
