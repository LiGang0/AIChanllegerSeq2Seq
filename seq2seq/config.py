import sys
import os
import os.path


root_path=os.path.abspath(os.path.join(os.path.split(os.path.realpath(__file__))[0],"../"))
data_path=os.path.abspath(os.path.join(root_path,"data"))

USE_CUDA=True



if __name__ == '__main__':
    print root_path
    print data_path
