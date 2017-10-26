
import os


from config import data_path,USE_CUDA,demo_path
from utils import tokenizedAndSave,readLanguages

import torch.utils.data


import random
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim







class Dataset(torch.utils.data.Dataset):
    #TODO: add docs
    """

    """
    # TODO: Construct function
    def __init__(self,spath,tpath,tmpdir=None,opt=None):
        self.spath=spath
        self.tpath=tpath
        datapath =os.path.split(self.spath)[0]
        # TODO: save the tokenized file
        if not tmpdir:
            tmpdir=os.path.abspath(os.path.join(datapath,'tokenized'))
        if not os.path.exists(tmpdir):
            os.makedirs(tmpdir)
        newspath=os.path.abspath(os.path.join(tmpdir,os.path.split(
                self.spath)[1]))
        newtpath=os.path.abspath(os.path.join(tmpdir,os.path.split(
                self.tpath)[1]))
        if not os.path.exists(newspath):
            print("Tokenize the files {}, saved in {}".format(self.spath,newspath))
            tokenizedAndSave(self.spath,newspath)
        else:
            print("Tokenized file exist")

        if not os.path.exists(newtpath):
            print("Tokenize the files {}, saved in {}".format(self.tpath, newtpath))
            tokenizedAndSave(self.tpath,newtpath)
        else:
            print("Tokenized file exist")
        self.spath=newspath
        self.tpath=newtpath

        self.source,self.target,self.pairs=readLanguages(self.spath,self.tpath)


    def __getitem__(self,i):
        return self.pairs
    

    def __len__(self):
        return len(self.pairs)


    def __iter__(self):
        return [pair for pair in self.pairs]


    # TODO: get_batch
    def get_batch(self,batch_size):
        pass





if __name__ == '__main__':
    dataset=Dataset(spath=os.path.abspath(os.path.join(demo_path,"train.en")),
                    tpath=os.path.abspath(os.path.join(demo_path,"train.zh")))


