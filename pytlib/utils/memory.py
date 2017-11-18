# derived from https://stackoverflow.com/questions/938733/total-memory-used-by-python-process
import os
class Memory:

    def __init__(self):
        self._proc_status = '/proc/%d/status' % os.getpid()
        self._scale = {'kB': 1024.0, 'mB': 1024.0*1024.0, 'KB': 1024.0, 'MB': 1024.0*1024.0}

    def _VmB(self,VmKey):
        '''Private.
        '''
         # get pseudo file  /proc/<pid>/status
        try:
            t = open(self._proc_status)
            v = t.read()
            t.close()
        except:
            return 0.0  # non-Linux?
         # get VmKey line e.g. 'VmRSS:  9999  kB\n ...'
        i = v.index(VmKey)
        v = v[i:].split(None, 3)  # whitespace
        if len(v) < 3:
            return 0.0  # invalid format?
         # convert Vm value to bytes
        return float(v[1]) * self._scale[v[2]]


    def memory(self,scale='mB'):
        return self._VmB('VmSize:') / self._scale[scale]


    def resident(self,scale='mB'):
        return self._VmB('VmRSS:') / self._scale[scale]


    def stacksize(self,scale='mB'):
        return self._VmB('VmStk:') / self._scale[scale]