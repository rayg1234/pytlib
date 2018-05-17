from image.box import Box

# and Object contains an id,box,and type
class Object:
    def __init__(self,box,unique_id=-1,obj_type='',polygon=None):
        self.box = box
        self.unique_id = unique_id
        self.obj_type = obj_type
        self.polygon = polygon