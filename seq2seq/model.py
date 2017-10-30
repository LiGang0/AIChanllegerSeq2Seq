import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from config import USE_CUDA




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



if __name__ == '__main__':
    pass