from torch.autograd import Variable
from image.visualization import tensor_to_pil_image_array, visualize_pil_array
import argparse
import imp
import time
from utils.debug import pp

class Trainer:

    def __init__(self,config):
        self.config = config

        # should use strings, namespace or functions here?
        self.loader = self.config.loader
        self.model = self.config.model
        self.lossfn = self.config.loss
        self.optimizer = self.config.optimizer

    def train(self,iterations,visualize=False):
        for i in range(0,iterations):
            sample = self.loader.next()
            inputs, target = Variable(sample.data), Variable(sample.target)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)

            # enc = self.model.get_encoding
            # import ipdb;ipdb.set_trace()
            # print enc.size()

            if visualize:
                visualize_pil_array(tensor_to_pil_image_array(outputs.data))
            # pp(inputs.size(),'input size')
            # pp(inputs.mean(),'input mean')
            # pp(outputs.size(),'output size')
            # pp(outputs.mean(),'output mean')
            # pp(outputs.min(),'output min')
            loss = self.lossfn(outputs, inputs)
            pp(loss,'loss')
            loss.backward()
            self.optimizer.step()

if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument('-t',"--train_config", required=True, type=str, help="the train configuration")
    parser.add_argument('-i',"--iterations",required=False, type=int, help="the number of iterations", default=1)
    parser.add_argument('-v',"--visualize",required=False,action='store_true', help="visualize output")
    args=parser.parse_args()

    train_config = imp.load_source('train_config', args.train_config)
    trainer = Trainer(train_config)
    trainer.train(args.iterations,visualize=args.visualize)
