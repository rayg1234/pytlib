import os
import shutil

def get_file_list(dir_path,matcher=lambda x:True):
    f = []
    # load all files, recursively descend into dirs
    for item in os.listdir(dir_path):
        full_path = os.path.join(dir_path,item)
        if os.path.isfile(full_path) and matcher(item):
            f.append(full_path)
        elif os.path.isdir(full_path):
            f.extend(load_directory(os.path.join(dir_path,item),matcher))
    return f

def mkdir(dir,wipe=False):
    if wipe and os.path.exists(dir):
        shutil.rmtree(dir)

    if not os.path.exists(dir):
        os.makedirs(dir)

def list_files(folder,ext_filter=None):
    onlyfiles = []
    for f in os.listdir(folder):
        fullpath = os.path.join(folder, f)
        if os.path.isfile(fullpath) and os.path.splitext(fullpath)[1]==ext_filter:
            onlyfiles.append(fullpath)
    return onlyfiles
