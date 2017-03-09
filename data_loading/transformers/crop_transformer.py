from transformer import Transformer
from image.frame import Frame

class CropTransformer(Transformer):
    def __init__(self,crop):
        # box object
        self.crop = crop

    def transform(self,frame):
        frame.load_image()
        return frame.image.crop(self.crop.to_single_array())
