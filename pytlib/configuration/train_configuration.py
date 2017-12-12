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