class TrainConfiguration:
    def __init__(self,loader,optimizer,model,lossfn,cuda=True):
        self.model = model
        self.optimizer = optimizer
        self.loader = loader
        self.loss = lossfn
        self.cuda = cuda
