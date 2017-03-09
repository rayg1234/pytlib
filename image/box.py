class Box:
    def __init__(self,xmin,ymin,xmax,ymax):
        assert xmax>xmin and ymax>ymin, "xmax>xmin and ymax>ymin!"
        self.xmin=xmin
        self.xmax=xmax
        self.ymin=ymin
        self.ymax=ymax
    
    @classmethod
    def from_double_array(cls,box):
        return cls(box[0][0],box[0][1],box[1][0],box[1][1])

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

    def __str__(self):
        return 'Box:' + str([[self.xmin,self.xmax],[self.ymin,self.ymax]])

    def __repr__(self):
        return self.__str__()