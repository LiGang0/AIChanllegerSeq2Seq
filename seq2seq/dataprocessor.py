
import os


from config import data_path
from utils import tokenizedAndSave

import torch.utils.data


import random
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim

SOS_token = 0
EOS_token = 1

def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]

def variableFromSentence(lang, sentence, config):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    result = Variable(torch.LongTensor(indexes).view(-1, 1))
    if config.use_cuda:
        return result.cuda()
    else:
        return result





class Dataset(torch.utils.data.Dataset):
    #TODO: add docs
    """

    """
    # TODO: Construct function
    def __init__(self,spath,tpath,tmpdir=None,opt=None):
        self.spath=spath
        self.tpath=tpath

        # TODO: save the tokenized file
        if not tmpdir:
            tmpdir=os.path.abspath(os.path.join(data_path,'tokenized'))
        if not os.path.exists(tmpdir):
            os.makedirs(tmpdir)
        newspath=os.path.exists(os.path.abspath(os.path.join(tmpdir,os.path.split(
                self.spath)[1])))
        newtpath=os.path.exists(os.path.abspath(os.path.join(tmpdir,os.path.split(
                self.tpath)[1])))
        if not newspath:
            tokenizedAndSave(self.spath,newspath)
        if not newtpath:
            tokenizedAndSave(self.tpath,newtpath)





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


