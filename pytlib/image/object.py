from image.box import Box

# and Object contains an id,box,and type
class Object:
    def __init__(self,box,unique_id=-1,obj_type=''):
        self.box = box
        self.unique_id = unique_id
        self.obj_type = obj_type
        
    @classmethod
    def from_dict(cls,objdict):
        #print objdict
        box = Box(objdict['box']['min'][0],objdict['box']['max'][0],objdict['box']['min'][1],objdict['box']['max'][1])
        uid = objdict['unique_identifier']
        obj_type = objdict['attributes']['type']
        return cls(box,uid,obj_type)