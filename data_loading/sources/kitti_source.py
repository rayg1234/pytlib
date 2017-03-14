import os
from image.box import Box
from image.frame import Frame
from image.object import Object
from os import listdir
from os.path import isfile, join, basename, splitext
from data_loading.sources.source import Source
from collections import defaultdict
from datasets.kitti_loader import KITTILoader

class KITTISource(Source):

    def __init__(self,dir_path,max_frames=float("inf")):
        self.dir_path = dir_path
        self.frames = []
        self.max_frames = max_frames

        self.load_frames(dir_path)
        self.size = len(self.frames)
        self.cur = 0

    # label folders are of the form label_02
    def __parse_label(self,full_path,label_prefix):
        chunks =  os.path.basename(full_path).split('_')
        if len(chunks)==2 and chunks[0]==label_prefix and chunks[1].isdigit():
            return int(chunks[1])
        else:
            return None

    # image folders are of the form image_02
    def __parse_imagedir(self,full_path,image_prefix):
        chunks = os.path.basename(full_path).split('_')
        if len(chunks)==2 and chunks[0]==image_prefix and chunks[1].isdigit() and os.path.isdir(full_path):
            return int(chunks[1])
        else:
            return None

    # KITTI images files are numerical only, ie: 00001, 00002 etc...
    def __validate_file_name(self,file_name):
        return file_name.isdigit()

    # assumes image dirs are of type image_xx and labels are label_xx.txt
    def load_frames(self,dir_path,frame_dir_prefix='image',label_prefix='label'):
        imagedirs = dict()
        labelfiles = dict()
        for item in listdir(dir_path):
            full_item_path = os.path.join(dir_path,item)
            ret = self.__parse_imagedir(full_item_path,frame_dir_prefix)
            if ret:
                imagedirs[ret] = full_item_path
            ret = self.__parse_label(full_item_path,label_prefix)
            if ret:
                labelfiles[ret] = full_item_path

        for k,image_path in imagedirs.items():
            if k in labelfiles:
                label_path = labelfiles[k]
                for item in listdir(image_path):
                    if self.__validate_file_name(item):
                        new_frames = KITTILoader.load_labelled_frames(os.path.join(image_path,item),os.path.join(label_path,item+'.txt'))
                        if len(self.frames) + len(new_frames) > self.max_frames:
                            return
                        self.frames.extend(new_frames)


    def __next__(self):
        cur+=1
        return frames[cur]

    def __iter__(self):
        return self

    def reset(self):
        cur = 0