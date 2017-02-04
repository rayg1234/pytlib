import os
import sys
import shutil
import json
from pymongo import MongoClient
from bson.objectid import ObjectId
import re
import pymongo

class PilotMongo:
    def __init__(self,addr="mongodb://localhost:27017"):
        self.client = MongoClient(addr)
        print "init mongoclient, found these dbs: " + str(self.client.database_names())
        self.db = self.client.detection
        
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

