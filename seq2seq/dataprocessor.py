import io
import os
import codecs
from collections import Counter, defaultdict
from itertools import chain, count

from config import data_path
from utils import tokenized,write_ob

import torch.utils.data
import jieba


class Dataset(torch.utils.data.Dataset):
    """
    # TODO: add docstring
    """
    # TODO: Construct function
    def __init__(self,spath,tpath,opt=None):
        self.spath=spath
        self.tpath=tpath
        self.source=tokenized(self.spath)
        self.target=tokenized(self.tpath)
        # TODO maybe save the file
        #
        #




    # TODO: get item function
    def __getitem__(self,i):
        pass
    
    # TODO: get length of the dataset
    def __len__(self):
        pass

    # TODO: iterator
    def __iter__(self):
        pass
    # TODO: get attr
    def __getattr__(self,attr):
        pass
    # TODO: get_batch
    def get_batch(self,batch_size):
        pass





if __name__ == '__main__':
    dataset=Dataset(spath=os.path.abspath(os.path.join(data_path,"train.en")),
                    tpath=os.path.abspath(os.path.join(data_path,"train.zh")))
    print dataset.source

