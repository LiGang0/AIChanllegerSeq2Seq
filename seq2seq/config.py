import sys
import os
import os.path

import torch
from torch import optim

SOS_token=0
EOS_token=1

root_path=os.path.abspath(os.path.join(os.path.split(os.path.realpath(__file__))[0],"../"))
data_path=os.path.abspath(os.path.join(root_path,"data"))
demo_path=os.path.abspath(os.path.join(data_path,"demo"))

USE_CUDA=False

MODE=['release','demo']
optimizier_dict={
    'adam'   : torch.optim.Adam,
    'sgd'    : torch.optim.SGD,
    'adagrad': torch.optim.Adagrad,
    'rmsprop': torch.optim.RMSprop,
    'rprop'  : torch.optim.Rprop,
}

class Config(object):
    def __init__(self,
                 batch_size,
                 n_epochs,
                 hidden_dim,
                 input_dim=10000,
                 output_dim=10000,
                 n_input_layers=1,
                 n_output_layers=1,
                 attn_model='dot',
                 dropout_p=0.1,
                 max_length=10,
                 optimizer='adam',
                 learning_rate=0.0001,
                 mode='demo'):
        """
        :param batchsize:
        :param n_epochs:
        :param mode:
        """
        self.batch_size=batch_size
        self.n_epochs=n_epochs
        self.mode=mode
        self.hidden_dim=hidden_dim
        self.attn_model=attn_model
        self.input_dim =input_dim
        self.n_input_layers=n_input_layers
        self.output_dim=output_dim
        self.n_output_layers=n_output_layers
        self.dropout_p=dropout_p
        self.max_length=max_length
        self.optimizier=optimizier_dict[optimizer]
        self.learning_rate=learning_rate
        if self.mode not in MODE:
            raise Exception("{} is not correct".format(self.mode))
        if self.mode==MODE[0]:
            self.sourcepath=os.path.abspath(os.path.join(data_path,"train.en"))
            self.targetpath=os.path.abspath(os.path.join(data_path,"train.zh"))
        else:
            self.sourcepath=os.path.abspath(os.path.join(demo_path,"train.en.rate"))
            self.targetpath=os.path.abspath(os.path.join(demo_path,"train.zh.rate"))
    def __str__(self):
        result=""
        for item,value in self.__dict__.iteritems():
            result+="{} : {}\n".format(item,value)
        return result


DemoConfig    = Config(batch_size=10,n_epochs=1000,hidden_dim=1000)

ReleaseConfig = Config(batch_size=10,n_epochs=1000,hidden_dim=1000,mode='release')

if __name__ == '__main__':
    pass

