import os
import sys
import shutil
import json
from pymongo import MongoClient
from bson.objectid import ObjectId
import re
import pymongo
from numpy import uint32
import warnings

warnings.filterwarnings('ignore')

class PilotMongo:
    def __init__(self,addr="mongodb://localhost:27017"):
        self.client = MongoClient(addr)
        print "init mongoclient, found these dbs: " + str(self.client.database_names())
        self.db = self.client.detection

    def get_framepath_objects(self,frame_id):
        query = {"_id":ObjectId(frame_id)}
        cursor = self.db.frames.find(query)
        return FrameLoader.frame_path(frame_id),[x['objects'] for x in cursor][0]
        
    def get_all_sets_ids(self):
        cursor = self.db.sets.find()
        return [x['_id'] for x in cursor]
    
    def get_set(self,setid):
        query = {"_id":ObjectId(setid)}
        cursor = self.db.sets.find(query)
        output = [x for x in cursor]
        return output

    def get_frames_from_set(self,setid):
        query = {"set_id":ObjectId(setid)}
        cursor = self.db.frames.find(query).sort('frame_index',pymongo.ASCENDING)
        print cursor.count()
        output = [x for x in cursor]
        return output
    
    def get_sets_from_tag(self,tag):
        query = {"tags":re.compile(tag)}
        cursor = self.db.sets.find(query)
        output = [x for x in cursor]
        return output
    
    def update_set_assignment(self,setid,assignment):
        query = {"_id":ObjectId(setid)}
        update = {'$set': {'assigned_to': assignment}}
        return mon.db.sets.update_one(query,update)
    
    def get_tags_from_set(self,setid):
        query = {"_id":ObjectId(setid)}
        cursor = self.db.sets.find(query)
        if cursor.count>=0:
            return cursor[0]['tags']
        else:
            return None
        
    def get_frame_indexes_from_set(self,setid):
        query = {"set_id":ObjectId(setid)}
        cursor = self.db.frames.find(query).sort('frame_index',pymongo.ASCENDING)
        output = [x['frame_index'] for x in cursor]
        return output
    
    def add_tags_to_set(self,setid,tag):
        query = {"_id":ObjectId(setid)}
        cursor = self.db.sets.find(query)
        cur_tags = [str(x) for x in cursor[0]['tags']]
        if tag not in cur_tags:
            update = {'$set': {'tags': cur_tags + [tag]}}
            return mon.db.sets.update_one(query,update)
        
    def remove_tags_from_set(self,setid,tag):
        query = {"_id":ObjectId(setid)}
        the_set = self.get_set(setid)
        cur_tags = the_set[0]['tags']
        new_tags = filter(lambda x: str(x)!=tag, cur_tags)
        update = {'$set': {'tags': new_tags}}
        return mon.db.sets.update_one(query,update)

class FrameLoader:

    @staticmethod
    def hash_(key):
        key += ~(key << uint32(15));
        key ^=  (key >> uint32(10));
        key +=  (key << uint32(3));
        key ^=  (key >> uint32(6));
        key += ~(key << uint32(11));
        key ^=  (key >> uint32(16));
        return key;

    @staticmethod
    def hash3(a,b,c):
        a-=b;a-=c;a^=(c>>uint32(13));
        b-=c;b-=a;b^=(a<<uint32(8));
        c-=a;c-=b;c^=(b>>uint32(13));
        a-=b;a-=c;a^=(c>>uint32(12));
        b-=c;b-=a;b^=(a<<uint32(16));
        c-=a;c-=b;c^=(b>>uint32(5));
        a-=b;a-=c;a^=(c>>uint32(3));
        b-=c;b-=a;b^=(a<<uint32(10));
        c-=a;c-=b;c^=(b>>uint32(15));
        return c;

    @staticmethod
    def hash_str(s):
      current_hash = FrameLoader.hash_(uint32(ord(s[0])));
      for i in range(1,len(s)):
        current_hash = FrameLoader.hash3(current_hash, uint32(ord(s[i])), uint32(15760506));

      return current_hash;

    @staticmethod
    def objectid_to_dir(_id):
        mask = uint32(255)
        hashcode = FrameLoader.hash_str(_id)
        first_dir = hashcode & mask
        second_dir = (hashcode >> 8) & mask
        return os.path.join("%03d"%int(first_dir), "%03d"%int(second_dir))

    @staticmethod
    def video_path(set_id,data_basedir='/local_data/DataServer'):
        return os.path.join(data_basedir,"Videos", FrameLoader.objectid_to_dir(set_id), set_id)

    @staticmethod
    def frame_path(frame_id,data_basedir='/local_data/DataServer'):
        return os.path.join(data_basedir,"Frames", FrameLoader.objectid_to_dir(frame_id), frame_id)