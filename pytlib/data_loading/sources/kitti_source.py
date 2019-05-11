import os
from image.box import Box
from image.frame import Frame
from image.object import Object
from os import listdir
from os.path import isfile, join, basename, splitext
from data_loading.sources.source import Source
from collections import defaultdict
from interface import Interface, implements
import numpy as np

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

class KITTISource(implements(Source)):

    image_dir = 'image_02'
    label_dir = 'label_02'
    calib_dir = 'calib'

    def __init__(self,dir_path,max_frames=float("inf")):
        self.dir_path = dir_path
        self.frames = []
        self.max_frames = max_frames

        self.__load_frames(dir_path)
        self.size = len(self.frames)
        self.cur = 0

    def __load_camera_matrix_from_calib(self,calib_file,line_prefix='P2'):
        # only load the p_rect calibration
        with open(calib_file, 'r') as f:
            lines = f.readlines()
            lines = [line.rstrip() for line in lines]
            mat = None
            for line in lines:
                nline = line.split(': ')
                if nline[0]==line_prefix:
                    mat = nline[1].split(' ')
                    mat = np.array([float(r) for r in mat], dtype=float)
                    mat = mat.reshape((3,4))[0:3, 0:3]
                    break
        return mat


    def __load_labelled_frames(self,frame_dir,labels_file,calib_file=None):
        files = [f for f in listdir(frame_dir) if isfile(join(frame_dir, f))]
        labels = KITTILabel.labels_from_file(labels_file) if labels_file is not None else []
        calibration_mat = None
        if os.path.isfile(calib_file):
            calibration_mat = self.__load_camera_matrix_from_calib(calib_file)
        frames = []
        for f in files:
            file_index = int(splitext(basename(f))[0])
            objects = []
            for l in labels:
                if int(l.frame_idx) == file_index:
                    objects.append(l.to_object())
            frames.append(Frame(join(frame_dir,f),objs=objects,calib_mat=calibration_mat))
        return frames

    # KITTI images files are numerical only, ie: 00001, 00002 etc...
    def __validate_file_name(self,file_name):
        return file_name.isdigit()

    # assumes image dirs are of type image_xx and labels are label_xx.txt
    def __load_frames(self,dir_path):
        imagedir_full = os.path.join(dir_path,KITTISource.image_dir)
        labeldir_full = os.path.join(dir_path,KITTISource.label_dir)
        calibdir_full = os.path.join(dir_path,KITTISource.calib_dir)
        assert os.path.exists(imagedir_full), "Cannot find image dir at {}".format(imagedir_full)
        assert os.path.exists(labeldir_full), "Cannot find image dir at {}".format(labeldir_full)
        assert os.path.exists(calibdir_full), "Cannot find image dir at {}".format(calibdir_full)
        for item in listdir(imagedir_full):
            if self.__validate_file_name(item):
                label_path = os.path.join(labeldir_full,item+'.txt')
                calib_path = os.path.join(calibdir_full,item+'.txt')
                new_frames = self.__load_labelled_frames(os.path.join(imagedir_full,item),label_path,calib_path)
                if len(self.frames) >= self.max_frames:
                    return
                self.frames.extend(new_frames[0:min(len(new_frames),self.max_frames - len(self.frames))])


    def next(self):
        if self.cur >= len(self.frames):
            raise StopIteration
        else:
            ret = self.frames[self.cur]
            self.cur+=1
            return ret

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.frames)

    def __getitem__(self,index):
        return self.frames[index]

    def reset(self):
        self.cur = 0