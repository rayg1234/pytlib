def TrainConfiguration:
    def __init__(self,loader,optimizer,model,lossfn):
        self.loader = loader
        self.optimizer = optimizer 
        self.model = model
        self.lossfn = lossfn
