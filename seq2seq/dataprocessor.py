
import os


from config import DemoConfig,USE_CUDA,EOS_token
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
    def __init__(self,config=DemoConfig,tmpdir=None,opt=None):
        self.spath=config.sourcepath
        self.tpath=config.targetpath
        self.config=config
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

    def getindex(self,sample):
        [s1,s2]=sample
        return [[self.source.word2index[item] for item in s1.split(' ')]+[EOS_token],
                [self.target.word2index[item] for item in s2.split(' ')]+[EOS_token]]

    def get_sample(self):
        sample=random.choice(self.pairs)
        return sample
    def get_index_sample(self):
        sample=self.get_sample()
        return self.getindex(sample)
    def get_sample_var(self):
        index_sample=self.get_index_sample()
        [input_index,output_index]=index_sample
        input_var=Variable(torch.LongTensor(input_index).view(-1,1))
        output_var=Variable(torch.LongTensor(output_index).view(-1,1))
        return [input_var,output_var]
    def get_batch(self,batchsize):
        #TODO:
        batch_input_var=[]
        batch_output_var=[]
        for i in range(batchsize):
            batch_input_var.append(self.get_index_sample)
            batch_output_var.append(self.get_index_sample)
        return batch_input_var,batch_output_var











if __name__ == '__main__':

    dataset=Dataset()


