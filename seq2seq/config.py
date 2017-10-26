import sys
import os
import os.path


root_path=os.path.abspath(os.path.join(os.path.split(os.path.realpath(__file__))[0],"../"))
data_path=os.path.abspath(os.path.join(root_path,"data"))
demo_path=os.path.abspath(os.path.join(data_path,"demo"))

USE_CUDA=True

MODE=['release','demo']

class Config(object):
    def __init__(self,batch_size,n_epochs,mode='demo'):
        """
        :param batchsize:
        :param n_epochs:
        :param mode:
        """
        self.batch_size=batch_size
        self.n_epochs=n_epochs
        self.mode=mode
        if self.mode not in MODE:
            raise Exception("{} is not correct".format(self.mode))
        if self.mode==MODE[0]:
            self.sourcepath=os.path.abspath(os.path.join(data_path,"train.en"))
            self.targetpath=os.path.abspath(os.path.join(data_path,"train.zh"))
        else:
            self.sourcepath=os.path.abspath(os.path.join(demo_path,"train.en.rate"))
            self.targetpath=os.path.abspath(os.path.join(demo_path,"train.zh.rate"))




DemoConfig    = Config(batch_size=10,n_epochs=1000,mode='demo')
ReleaseConfig = Config(batch_size=10,n_epochs=1000,mode='release')

if __name__ == '__main__':
    print root_path
    print data_path
