import numpy as np

class BoxCoordinatesInvalidException(Exception):
    pass

class Box:
    def __init__(self,xmin,ymin,xmax,ymax):
        if not (xmax>xmin and ymax>ymin):
            raise BoxCoordinatesInvalidException("BoxCoords invalid! [{0},{1}], [{2},{3}]".format(xmin,ymin,xmax,ymax))
        self.xmin=xmin
        self.xmax=xmax
        self.ymin=ymin
        self.ymax=ymax
    
    @classmethod
    def from_double_array(cls,box):
        return cls(box[0][0],box[0][1],box[1][0],box[1][1])

    @classmethod
    def from_augmented_matrix(cls,box):
        if box.shape != (3,2):
            raise BoxCoordinatesInvalidException('BoxCoords shape is not (3,2) {0}'.format(box.shape))
        return cls(box[0][0],box[1][0],box[0][1],box[1][1])

    @classmethod
    def from_single_array(cls,box):
        return cls(box[0],box[1],box[2],box[3])
    
    def xy_min(self):
        return (self.xmin,self.ymin)
    
    def edges(self):
        return (self.xmax-self.xmin,self.ymax-self.ymin)
    
    def area(self):
        return (self.xmax-self.xmin)*(self.ymax-self.ymin)

    def to_single_array(self):
        return [self.xmin,self.ymin,self.xmax,self.ymax]

    def to_single_np_array(self):
        return np.array(self.to_single_array())

    def to_2x2_np_array(self):
        return np.array([[self.xmin,self.xmax],[self.ymin,self.ymax]])

    def augmented_matrix(self):
        return np.array([[self.xmin,self.xmax],[self.ymin,self.ymax],[1,1]])

    def __str__(self):
        return 'Box [[x0,y0],[x1,y1]]:' + str([[self.xmin,self.ymin],[self.xmax,self.ymax]])

    def __repr__(self):
        return self.__str__()