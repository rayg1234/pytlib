import os
    
def load_directory(dir_path,matcher=lambda x:True):
    f = []
    # load all files, recursively descend into dirs
    for item in os.listdir(dir_path):
        full_path = os.path.join(dir_path,item)
        if os.path.isfile(full_path) and matcher(item):
            f.append(full_path)
        elif os.path.isdir(full_path):
            f.extend(load_directory(os.path.join(dir_path,item),matcher))
    return f