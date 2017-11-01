import argparse

import torch
import torch.nn as nn
from pylab import *
from torch.autograd import Variable

from config import DemoConfig,ReleaseConfig
from model import EncoderRNN,AttnDecoderRNN
from dataprocessor import Dataset
class Train(object):

    """

    """
    #TODO : need to change, with no dataset in the train class, a apply method should be
    #TODO : in this class, which accept a dataset parameters and train the dataset .

    def __init__(self,config,dataset):
        self.config=config
        self.n_epochs=config.n_epochs
        self.encoder=EncoderRNN(n_dict=dataset.source.n_words,config=config)
        self.decoder=AttnDecoderRNN(n_dict=dataset.target.n_words,config=config)
        self.encoder_optimizer=config.optimizier(self.encoder.parameters(),lr=config.learning_rate)
        self.decoder_optimizer=config.optimizier(self.decoder.parameters(),lr=config.learning_rate)
        self.criterion = nn.NLLLoss()
        self.is_plot=config.is_plot
        self.clip_value=config.clip_value
        self.losses = []
        if self.config.USE_CUDA:
            self.encoder.cuda(self.config.gpu_id)
        if self.config.USE_CUDA:
            self.decoder.cuda(device_id=self.config.gpu_id)
    def train(self,dataset):
        if self.is_plot:
            fig, ax = plt.subplots()
            grid(True)
            plt.ion()
        for epoch in range(self.n_epochs):
            training_pair=dataset.get_sample_var()
            loss,result_output=self.step(training_pair)
            print("At Epoch : {:5},Get loss : {:10}\n".format(epoch,loss))
            self.losses.append(loss)
            if self.is_plot:
                ax.plot(range(epoch+1),self.losses,"b")
                plt.pause(0.0001)
                plt.show()
            if epoch%100==0:
                print ''.join([dataset.target.index2word[i] for i in training_pair[1].squeeze(1).data.tolist()])
                print ''.join([dataset.target.index2word[i] for i in result_output])

    def step(self,training_pair):
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()

        input_variable = training_pair[0]
        target_variable = training_pair[1]

        loss=0
        input_length=input_variable.size()[0]
        target_length=target_variable.size()[0]

        encoder_hidden=self.encoder.init_hidden()
        encoder_outputs,encoder_hidden=self.encoder(input_variable,encoder_hidden)
        decoder_input=Variable(torch.LongTensor([[self.config.SOS_token]]))
        decoder_context = Variable(torch.zeros(1, self.decoder.hidden_dim))
        decoder_hidden = encoder_hidden
        if self.config.USE_CUDA:
            decoder_input=decoder_input.cuda(device_id=self.config.gpu_id)
            decoder_context=decoder_context.cuda(device_id=self.config.gpu_id)
            assert type(decoder_input.data)==torch.cuda.LongTensor
            assert type(decoder_context.data)==torch.cuda.FloatTensor
        result_output=[]
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
            if self.config.USE_CUDA:
                decoder_input=decoder_input.cuda(device_id=self.config.gpu_id)
            result_output.append(ni)
            if ni == self.config.EOS_token: break
        loss.backward()
        # TODO : clip value
        torch.nn.utils.clip_grad_norm(self.encoder.parameters(), self.clip_value)
        torch.nn.utils.clip_grad_norm(self.decoder.parameters(), self.clip_value)


        self.encoder_optimizer.step()
        self.decoder_optimizer.step()
        if self.config.USE_CUDA:
            return loss.cpu().data[0]/target_length,result_output
        return loss.data[0]/target_length,result_output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m","--mode",help= "Chose a mode",default="demo")
    parser.add_argument("-p","--plot",help="Is plot?",default=False)
    parser.add_argument("-g","--gpu_id",help="choose 1 or 0",default=0)
    args=parser.parse_args()

    if args.mode=='release':
        Myconfig=ReleaseConfig
    elif args.mode=='demo':
        Myconfig=DemoConfig
    if args.plot:
        Myconfig.is_plot=True
    Myconfig.gpu_id=int(args.gpu_id)
    print Myconfig
    dataset=Dataset(config=Myconfig)

    experiment=Train(config=Myconfig,dataset=dataset)
    experiment.train(dataset)



if __name__ == '__main__':
    main()
