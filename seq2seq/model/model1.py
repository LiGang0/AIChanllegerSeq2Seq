import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from seq2seq.config import EOS_token,SOS_token
from seq2seq.config import USE_CUDA
from seq2seq.config import ReleaseConfig
from seq2seq.config import DemoConfig
from seq2seq.dataprocessor import Dataset


class EncoderRNN(nn.Module):
    def __init__(self, n_dict ,config):
        super(EncoderRNN, self).__init__()

        self.n_dict=n_dict
        self.input_dim = config.input_dim
        self.hidden_dim = config.hidden_dim
        self.n_layers = config.n_input_layers

        self.embedding = nn.Embedding(self.n_dict, self.input_dim)
        self.gru = nn.GRU(self.input_dim, self.hidden_dim, self.n_layers)

    def forward(self, word_inputs, hidden):
        # Note: we run this all at once (over the whole input sequence)
        seq_len = len(word_inputs)
        embedded = self.embedding(word_inputs).view(seq_len, 1, -1)
        output, hidden = self.gru(embedded, hidden)
        return output, hidden

    def init_hidden(self):
        hidden = Variable(torch.zeros(self.n_layers, 1, self.hidden_dim))
        if USE_CUDA: hidden = hidden.cuda()
        return hidden

class Attn(nn.Module):
    def __init__(self,config):
        super(Attn, self).__init__()

        self.method = config.attn_model
        self.hidden_dim = config.hidden_dim
        self.max_length=config.max_length
        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_dim, self.hidden_dim)

        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
            self.other = nn.Parameter(torch.FloatTensor(1, self.hidden_dim))

    def forward(self, hidden, encoder_outputs):
        seq_len = len(encoder_outputs)

        # Create variable to store attention energies
        attn_energies = Variable(torch.zeros(seq_len))  # B x 1 x S
        if USE_CUDA: attn_energies = attn_energies.cuda()

        # Calculate energies for each encoder output
        for i in range(seq_len):
            attn_energies[i] = self.score(hidden, encoder_outputs[i])

        # Normalize energies to weights in range 0 to 1, resize to 1 x 1 x seq_len
        return F.softmax(attn_energies).unsqueeze(0).unsqueeze(0)

    def score(self, hidden, encoder_output):

        if self.method == 'dot':
            energy = hidden.dot(encoder_output)
            return energy

        elif self.method == 'general':
            energy = self.attn(encoder_output)
            energy = hidden.dot(energy)
            return energy

        elif self.method == 'concat':
            energy = self.attn(torch.cat((hidden, encoder_output), 1))
            energy = self.other.dot(energy)
            return energy


class AttnDecoderRNN(nn.Module):
    def __init__(self, n_dict,config):
        super(AttnDecoderRNN, self).__init__()

        # Keep parameters for reference
        self.config=config
        self.n_dict=n_dict
        self.attn_model = config.attn_model
        self.hidden_dim = config.hidden_dim
        self.output_dim = config.output_dim
        self.n_layers = config.n_output_layers
        self.dropout_p = config.dropout_p

        # Define layers
        self.embedding = nn.Embedding(self.n_dict, self.output_dim)
        #TODO : this is wrong
        self.gru = nn.GRU(self.output_dim + self.hidden_dim, self.hidden_dim, self.n_layers, dropout=self.dropout_p)
        #TODO : this is wrong
        self.out = nn.Linear(self.hidden_dim * 2, n_dict)

        # Choose attention model
        if self.attn_model != 'none':
            self.attn = Attn(self.config)

    def forward(self, word_input, last_context, last_hidden, encoder_outputs):
        # Note: we run this one step at a time

        # Get the embedding of the current input word (last output word)
        word_embedded = self.embedding(word_input).view(1, 1, -1)  # S=1 x B x N

        # Combine embedded input word and last context, run through RNN
        rnn_input = torch.cat((word_embedded, last_context.unsqueeze(0)), 2)
        rnn_output, hidden = self.gru(rnn_input, last_hidden)

        # Calculate attention from current RNN state and all encoder outputs; apply to encoder outputs
        attn_weights = self.attn(rnn_output.squeeze(0), encoder_outputs)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))  # B x 1 x N

        # Final output layer (next word prediction) using the RNN hidden state and context vector
        rnn_output = rnn_output.squeeze(0)  # S=1 x B x N -> B x N
        context = context.squeeze(1)  # B x S=1 x N -> B x N
        output = F.log_softmax(self.out(torch.cat((rnn_output, context), 1)))

        # Return final output, hidden state, and attention weights (for visualization)
        return output, context, hidden, attn_weights



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

    def train(self):
        for epoch in range(self.n_epochs):
            loss=self.step()
            print("At Epoch : {:5},Get loss : {:10}\n".format(epoch,loss))

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

        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

        return loss.data[0] / target_length







        pass


def train(input_variable, target_variable, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion,
          teacher_forcing_ratio=0.1,clip=5.0,max_length=10):
    # Zero gradients of both optimizers
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    loss = 0  # Added onto for each word

    # Get size of input and target sentences
    input_length = input_variable.size()[0]
    target_length = target_variable.size()[0]

    # Run words through encoder
    encoder_hidden = encoder.init_hidden()
    encoder_outputs, encoder_hidden = encoder(input_variable, encoder_hidden)

    # Prepare input and output variables
    decoder_input = Variable(torch.LongTensor([[SOS_token]]))
    decoder_context = Variable(torch.zeros(1, decoder.hidden_size))
    decoder_hidden = encoder_hidden  # Use last hidden state from encoder to start decoder
    if USE_CUDA:
        decoder_input = decoder_input.cuda()
        decoder_context = decoder_context.cuda()

    # Choose whether to use teacher forcing
    use_teacher_forcing = random.random() < teacher_forcing_ratio
    if use_teacher_forcing:

        # Teacher forcing: Use the ground-truth target as the next input
        for di in range(target_length):
            decoder_output, decoder_context, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_context,
                                                                                         decoder_hidden,
                                                                                         encoder_outputs)
            loss += criterion(decoder_output[0], target_variable[di])
            decoder_input = target_variable[di]  # Next target is next input

    else:
        # Without teacher forcing: use network's own prediction as the next input
        for di in range(target_length):
            decoder_output, decoder_context, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_context,
                                                                                         decoder_hidden,
                                                                                         encoder_outputs)
            loss += criterion(decoder_output[0], target_variable[di])

            # Get most likely word index (highest value) from output
            topv, topi = decoder_output.data.topk(1)
            ni = topi[0][0]

            decoder_input = Variable(torch.LongTensor([[ni]]))  # Chosen word is next input
            if USE_CUDA: decoder_input = decoder_input.cuda()

            # Stop at end of sentence (not necessary when using known targets)
            if ni == EOS_token: break

    # Backpropagation
    loss.backward()
    torch.nn.utils.clip_grad_norm(encoder.parameters(), clip)
    torch.nn.utils.clip_grad_norm(decoder.parameters(), clip)
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.data[0] / target_length


if __name__ == '__main__':
    pass