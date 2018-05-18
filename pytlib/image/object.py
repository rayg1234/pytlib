from image.box import Box

# and Object contains an id,box,and type
class Object:
    def __init__(self,box,unique_id=-1,obj_type='',polygons=[]):
        self.box = box # box object
        self.unique_id = unique_id # int
        self.obj_type = obj_type # string 
        self.polygons = polygons # list of polygon objects