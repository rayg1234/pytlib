# Holds all the configuration data to train a model
# the loader, optimizer and model are all passed in as tuples of the form
# (class,params) so that they can be instantiated lazily later
# This is done so that we can for example instantiate the optimizer after
# the module's forward function has been called with the correct input 
# to accomedate for dynamically allocated parameters
class TrainConfiguration:
    def __init__(self,loader_params,optimizer_params,model_params,lossfn,cuda=True):
        self.loader_params = loader_params
        self.model_params = model_params
        self.optimizer_params = optimizer_params
        self.model = None
        self.optimizer = None
        self.loader = None
        self.lossfn = lossfn
        self.cuda = cuda

    def get_model(self):
        if self.model is None:
            self.model = self.model_params[0](**self.model_params[1])
            if self.cuda:
                self.model.cuda()
        return self.model

    def get_optimizer(self):
        if self.optimizer is None:
            self.optimizer_params[1]['params']=self.get_model().parameters()
            self.optimizer = self.optimizer_params[0](**self.optimizer_params[1])
        return self.optimizer

    def get_loader(self):
        if self.loader is None:
            self.loader = self.loader_params[0](**self.loader_params[1])
        return self.loader

    def get_lossfn(self):
        return self.lossfn