import numpy as np

class SimpleCSR:
    def __init__(self, data, cols, row, from_disk=False):
        if not from_disk:
            self.data = data
            self.cols = cols
            self.row = row
            
        else:
            self.row = np.load(row, mmap_mode='r')
            self.cols = np.load(cols, mmap_mode='r')
            self.data = np.load(data, mmap_mode='r')
            
        self.shape = [self.row.shape[0]-1, 1]
        
        
    def __getitem__(self, key):
        start = self.row[key]
        end = self.row[key+1]
        
        return DataObject(self.data[start:end], self.cols[start:end])
        
class DataObject:
    def __init__(self, data, idx):
        self.indices = idx
        self.data = data    