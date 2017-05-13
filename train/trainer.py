from torch.autograd import Variable
import argparse
import imp

class Trainer:

    def __init__(self,config):
        self.config = config

        # should use strings, namespace or functions here?
        self.loader = self.config.loader
        self.model = self.config.model
        self.lossfn = self.config.loss
        self.optimizer = self.config.optimizer

    def train(self,iterations):
        for i in range(0,iterations):
            sample = self.loader.next()
            inputs, target = Variable(sample.data), Variable(sample.target)

            self.optimizer.zero_grad()

            outputs = self.model(inputs)
            # print outputs.size()
            # print inputs.size()
            # print inputs.mean()
            # print outputs.mean()
            # import ipdb;ipdb.set_trace()
            loss = self.lossfn(outputs, inputs)
            loss.backward()
            self.optimizer.step()

if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument('-t',"--train_config", required=True, type=str, help="the train configuration")
    args=parser.parse_args()

    train_config = imp.load_source('train_config', args.train_config)
    trainer = Trainer(train_config)
    trainer.train(1)
