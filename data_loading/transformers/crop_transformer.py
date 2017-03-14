from transformer import Transformer
from image.frame import Frame

class CropTransformer(Transformer):
    def __init__(self,crop,obj_types=[]):
        # box object
        self.crop = crop
        self.obj_types = obj_types

    def transform(self,frame):
        frame.load_image()
        return frame.image.crop(self.crop.to_single_array())
