# base class for input generation
# a loader has
# a) some source or grab the data from
# b) an iterator to figure out the next sample to return
# currently the DataLoader is responsible for translating the raw data 
# to "trainable" data (for example, get some crop). So a subclass
# here is very specific to the application. 
# Ie: a CarDataLoader can grab data from a KITTI set, crop around cars
# and return the tensor associated with the car.

class DataLoader:
    def __init__(self,source):
        self.source = source
    
    def __iter__(self):
        return self
    
    def __next__(self):
        pass