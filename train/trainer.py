from torch.autograd import Variable
import argparse

class Trainer:

    def __init__(self,config):
        self.config = config

        # should use strings, namespace or functions here?
        self.loader = self.config.loader
        self.model = self.config.model
        self.lossfn = self.config.lossfn
        self.optimizer = self.config.optimizer

    def train(iterations):
        for i in range(0,iterations):
            sample = self.loader.next()
            inputs, target = Variable(sample.data), Variable(sample.target)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = lossfn(outputs, labels)
            loss.backward()
            optimizer.step()

if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--train_config", type=str, help="the train configuration")
    args=parser.parse_args()

