import scipy
import numpy as np
import math

# class to chain affine transforms
class Affine:

    @classmethod
    def translation(cls,a,b):
        return np.array([[1,0,a],[0,1,b],[0,0,1]])
  
    @classmethod
    def scaling(cls,a,b):
        return np.array([[a,0,0],[0,b,0],[0,0,1]])

    @classmethod
    def rotation(cls,x):
        return np.array([[math.cos(x),-math.sin(x),0],[math.sin(x),math.cos(x),0],[0,0,1]])

    @classmethod
    def identity(cls):
        return np.array([[1,0,0],[0,1,0],[0,0,1]])

    def __init__(self):
        self.transform = identity()
        self.inverse = identity()

    # matrix inoput has to be non-singular
    def append(matrix):
        self.transform = numpy.dot(self.transform * matrix)
        self.inverse = numpy.dot(numpy.linalg.inv(matrix),self.inverse)

    def apply(input):
        return numpy.dot(self.transform,input)

    def unapply(input):
        return numpy.dot(self.inverse,input)


