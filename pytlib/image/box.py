import torch
import numpy as np

class BoxCoordinatesInvalidException(Exception):
    pass

class Box:
    def __init__(self,xmin,ymin,xmax,ymax):
        # if not (xmax>xmin and ymax>ymin):
        #     raise BoxCoordinatesInvalidException("BoxCoords invalid! [{0},{1}], [{2},{3}]".format(xmin,ymin,xmax,ymax))
        self.xmin=xmin
        self.xmax=xmax
        self.ymin=ymin
        self.ymax=ymax

    @classmethod
    def from_xywh(cls,data):
        return cls(data[0],data[1],data[0]+data[2],data[1]+data[3])

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

    @classmethod
    def from_edges(cls,edges):
        assert len(edges)==2,'Edges must be a 2-vector!'
        return cls(0,0,edges[0],edges[1])

    def xy_min(self):
        return np.array([self.xmin,self.ymin])

    def xy_max(self):
        return np.array([self.xmax,self.ymax])

    def center(self):
        return np.array([(self.xmin+self.xmax)/2,(self.ymin+self.ymax)/2])

    def edges(self):
        return np.array([self.xmax-self.xmin,self.ymax-self.ymin])

    def area(self):
        return (float)(self.xmax-self.xmin)*(self.ymax-self.ymin)

    def to_single_array(self):
        return np.array([self.xmin,self.ymin,self.xmax,self.ymax])

    def to_2x2_np_array(self):
        return np.array([[self.xmin,self.xmax],[self.ymin,self.ymax]])

    def augmented_matrix(self):
        return np.array([[self.xmin,self.xmax],[self.ymin,self.ymax],[1,1]])

    # rescale x and y separately
    def scale(self,scale):
        assert len(scale)==2, 'must provide 2d scale for x and y'
        return Box(self.xmin * scale[0],
                   self.ymin * scale[1],
                   self.xmax * scale[0],
                   self.ymax * scale[1])

    # generate bounded box bound by a second box2,
    # ie: the current box must lie entirely within the second box
    def bound(self,box2):
        coords = [None]*4
        coords[0] = self.xmin if self.xmin<box2.xmin else box2.xmin
        coords[1] = self.ymin if self.ymin<box2.ymin else box2.ymin
        coords[2] = self.xmax if self.xmax>box2.xmax else box2.xmax
        coords[3] = self.ymax if self.ymax>box2.ymax else box2.ymax
        return Box.from_single_array(coords)

    @staticmethod
    def intersection(b1,b2):
        xmin,ymin = max(b1.xmin,b2.xmin),max(b1.ymin,b2.ymin)
        xmax,ymax = min(b1.xmax,b2.xmax),min(b1.ymax,b2.ymax)
        new_box = Box(xmin,ymin,xmax,ymax)
        if new_box.xmin < new_box.xmax and new_box.ymin < new_box.ymax:
            return new_box
        else:
            return None

    #TODO unify these methods, some of them are normalized, some are not :(
    @staticmethod
    def box_to_tensor(box,frame_size):
        # normalize box coord to between 0 and 1
        box_array = box.scale(1/np.array(frame_size,dtype=float)).to_single_array().astype(float)
        return torch.Tensor(box_array)

    @staticmethod
    def boxes_to_tensor(boxes,frame_size):
        # normalize box coord to between 0 and 1
        box_array = []
        for box in boxes:
            box_array.append(box.scale(1/np.array(frame_size,dtype=float)).to_single_array().astype(float))
        return torch.Tensor(np.stack(box_array))

    @staticmethod
    def tensor_to_boxes(tensor):
        assert len(tensor.shape)==2 and tensor.shape[1]==4, \
            'tensor boxes must be Nx4 shape'
        chunks = torch.chunk(tensor,tensor.shape[0],dim=0)
        box_list = []
        for c in chunks:
            box_list.append(Box.tensor_to_box(c.squeeze()))
        return box_list

    @staticmethod
    def tensor_to_box(tensor,frame_size=None):
        assert tensor.size()==torch.Size([4]), 'tensor must of size 4 got {}'.format(tensor.size())
        ret_box = Box.from_single_array(tensor.numpy())
        if frame_size is not None:
            ret_box.scale(frame_size)
        return ret_box

    def __str__(self):
        return 'Box [[x0,y0],[x1,y1]]:' + str([[self.xmin,self.ymin],[self.xmax,self.ymax]])

    def __repr__(self):
        return self.__str__()