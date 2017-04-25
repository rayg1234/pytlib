# base class for input generation
# Source provides frames
# Source could be a sequence of multiple items
# Transformer takes frames/sequence of frames and turns into samples
# To add: a sampler and a perturber step

class DataLoader:
    def __init__(self,source,transformer):
        self.source = source
        self.transformer = transformer
    
    def __iter__(self):
        return self

    # iterate over source inorder
    # call transformer with each set of frames
    # return the corresponding sample    
    def next(self):
        next_frames = self.source.next()
        next_sample = self.transformer.transform(next_frames)
        return next_sample


#### Data Loader factory for kind of dependency injection
from data_loading.sources.kitti_source import KITTISource
from data_loading.transformers.crop_transformer import CropTransformer

class DataLoaderFactory:

    @classmethod
    def GetKITTILoader(cls):
        source = KITTISource('path/to/source')
        transformer = CropTransformer(['people'])
        return DataLoader(source,transformer)
