
from config import DemoConfig,ReleaseConfig
from model.model1 import EncoderRNN,AttnDecoderRNN,Train
from dataprocessor import Dataset

import argparse





def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m","--mode",help= "Chose a mode")
    args=parser.parse_args()

    if args.mode=='release':
        config=ReleaseConfig
    elif args.mode=='demo':
        config=DemoConfig
    experiment=Train(config=config)
    experiment.train()



if __name__ == '__main__':
    main()