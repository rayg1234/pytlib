# a Source knows where to grab the raw data from and returns Frame objects
# eg: DirectoryFileSource grab all files from a dir and also a map from file_path to label_path to grab labels from
# Loads all files and labels into memory (not images)
class Source:

    def __getitem__(self,index):
        raise NotImplementedError

    def next(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __iter__(self):
        raise NotImplementedError