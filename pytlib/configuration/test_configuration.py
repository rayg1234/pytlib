class TestConfiguration:
    def __init__(self,loader,model,lossfn,cuda=True):
        self.model = model
        self.loader = loader
        self.cuda = cuda