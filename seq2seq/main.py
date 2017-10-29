import argparse

from config import DemoConfig,ReleaseConfig
from model import Train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m","--mode",help= "Chose a mode",default="demo")
    args=parser.parse_args()

    if args.mode=='release':
        Myconfig=ReleaseConfig
    elif args.mode=='demo':
        Myconfig=DemoConfig
    experiment=Train(config=Myconfig)
    experiment.train()



if __name__ == '__main__':
    main()