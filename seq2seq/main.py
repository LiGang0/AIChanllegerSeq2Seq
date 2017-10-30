import argparse

from config import DemoConfig,ReleaseConfig,SOS_token,EOS_token
from dataprocessor import Dataset
from model import EncoderRNN,AttnDecoderRNN
import torch
import torch.nn as nn
from torch.autograd import Variable
from pylab import *

class Train(object):

    """

    """
    #TODO : need to change, with no dataset in the train class, a apply method should be
    #TODO : in this class, which accept a dataset parameters and train the dataset .

    def __init__(self,config):
        self.n_epochs=config.n_epochs
        self.dataset=Dataset(config=config)
        self.encoder=EncoderRNN(n_dict=self.dataset.source.n_words,config=config)
        self.decoder=AttnDecoderRNN(n_dict=self.dataset.target.n_words,config=config)
        self.encoder_optimizer=config.optimizier(self.encoder.parameters(),lr=config.learning_rate)
        self.decoder_optimizer=config.optimizier(self.decoder.parameters(),lr=config.learning_rate)
        self.criterion = nn.NLLLoss()
        self.is_plot=config.is_plot
        self.clip_value=config.clip_value
        self.losses = []

    def train(self):
        if self.is_plot:
            fig, ax = plt.subplots()
            grid(True)
            plt.ion()
        for epoch in range(self.n_epochs):
            loss=self.step()
            print("At Epoch : {:5},Get loss : {:10}\n".format(epoch,loss))
            self.losses.append(loss)
            if self.is_plot:
                ax.plot(range(epoch+1),self.losses,"b")
                plt.pause(0.0001)
                plt.show()

    def step(self):
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        training_pair=self.dataset.get_sample_var()
        input_variable = training_pair[0]
        target_variable = training_pair[1]

        loss=0
        input_length=input_variable.size()[0]
        target_length=target_variable.size()[0]

        encoder_hidden=self.encoder.init_hidden()
        encoder_outputs,encoder_hidden=self.encoder(input_variable,encoder_hidden)
        decoder_input=Variable(torch.LongTensor([[SOS_token]]))
        decoder_context = Variable(torch.zeros(1, self.decoder.hidden_dim))
        decoder_hidden = encoder_hidden

        for di in range(target_length):
            decoder_output, \
            decoder_context, \
            decoder_hidden, \
            decoder_attention = self.decoder(decoder_input,
                                             decoder_context,
                                             decoder_hidden,
                                             encoder_outputs)
            loss += self.criterion(decoder_output[0], target_variable[di])
            topv, topi = decoder_output.data.topk(1)
            ni = topi[0][0]
            decoder_input = Variable(torch.LongTensor([[ni]]))
            if ni == EOS_token: break
        loss.backward()
        # TODO : clip value
        torch.nn.utils.clip_grad_norm(self.encoder.parameters(), self.clip_value)
        torch.nn.utils.clip_grad_norm(self.decoder.parameters(), self.clip_value)


        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

        return loss.data[0] / target_length


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m","--mode",help= "Chose a mode",default="demo")
    parser.add_argument("-p","--plot",help="Is plot?",default=False)
    args=parser.parse_args()

    if args.mode=='release':
        Myconfig=ReleaseConfig
    elif args.mode=='demo':
        Myconfig=DemoConfig
    if args.plot:
        Myconfig.is_plot=True
    experiment=Train(config=Myconfig)
    experiment.train()



if __name__ == '__main__':
    main()