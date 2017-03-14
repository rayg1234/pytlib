import os
from image.box import Box
from image.frame import Frame
from image.object import Object
from os import listdir
from os.path import isfile, join, basename, splitext
from data_loading.source import Source
from collections import defaultdict

class KITTILabel:
    @classmethod
    def labels_from_file(cls,filename):
        labels = []
        with open(filename,'r') as f:
            for line in f:
                labels.append(cls(line))
        return labels
    
    def __init__(self,line):
        linearr = line.split(' ')
        self.frame_idx = linearr[0]
        self.track_idx = linearr[1]
        self.type = linearr[2]
        self.truncated = linearr[3]
        self.occluded = linearr[4]
        self.bbox = [float(x) for x in linearr[6:10]]
        
    def to_object(self):
        box_format = [self.bbox[0],self.bbox[1],self.bbox[2],self.bbox[3]]
        return Object(Box.from_single_array(box_format),self.track_idx,self.type)
    
class KITTILoader:
    @classmethod
    def load_labelled_frames(cls,frame_dir,labels_file):
        files = [f for f in listdir(frame_dir) if isfile(join(frame_dir, f))]
        labels = KITTILabel.labels_from_file(labels_file)
        frames = []
        for f in files:
            file_index = int(splitext(basename(f))[0])
            objects = []
            for l in labels:
                if int(l.frame_idx) == file_index:
                    objects.append(l.to_object())
            frames.append(Frame(join(frame_dir,f),objects))
        return frames
