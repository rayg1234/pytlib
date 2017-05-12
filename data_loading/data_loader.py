# base class for input generation
# Source provides frames
# Source could be a sequence of multiple items
# Transformer takes frames/sequence of frames and turns into samples
# To add: a sampler and a perturber step
import sys

class DataLoader:
    def __init__(self,source,transformer):
        self.source = source
        self.transformer = transformer
        self.max_tries = 10
    
    def __iter__(self):
        return self

    # iterate over source inorder
    # call transformer with each set of frames
    # return the corresponding sample    
    def next(self):
        next_frames = self.source.next()
        for i in range(0,self.max_tries):
            next_sample = self.transformer.transform(next_frames)
            if next_sample:
                break

        if i>=self.max_tries:
            print "Too many tries without a successful sample, breaking!"
            sys.exit(0)

        return next_sample


#### Data Loader factory for kind of dependency injection
from data_loading.sources.kitti_source import KITTISource
from data_loading.transformers.crop_transformer import CropTransformer

class DataLoaderFactory:

    @classmethod
    def GetKITTILoader(cls):
        source = KITTISource('../Data/KITTI/training',max_frames=200)
        transformer = CropTransformer([100,100],['Car'])
        return DataLoader(source,transformer)