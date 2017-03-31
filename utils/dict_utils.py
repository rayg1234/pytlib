def format_path(path):
    if isinstance(path, basestring):
        return path.split('.')
    else:
        return path

# note this doesn't do array indexing yet, TODO
# def get_deep(d, path):
#     path = format_path(path)
#     return reduce(dict.__getitem__, path, d)

def get_deep(d,path):
    val = d
    path = format_path(path)
    for key in path:
        if(key.isdigit() and isinstance(val,list)):
            val = val[int(key)]
        else:
            val = val.get(key)
    return val

def set_deep(d, path, value):
    path = format_path(path)
    for key in path[:-1]:
        if(key.isdigit() and isinstance(d, list)):
            d = d[int(key)]
        else:
            d = d.setdefault(key, {})
    if(path[-1].isdigit() and isinstance(d, list)):
        d[int(path[-1])] = value
    else:
        d[path[-1]] = value
