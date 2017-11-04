import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F

from dataprocessor import Dataset
from config import DemoConfig
class NGramLanguageModeler(nn.Module):
    def __init__(self,vocab_size,embedding_dim,context_text):
        super(NGramLanguageModeler, self).__init__()
        self.embedding=nn.Embedding(vocab_size,embedding_dim)
        self.linear1=nn.Linear(context_text*embedding_dim,128)
        self.linear2 = nn.Linear(128, vocab_size)

    def forward(self, inputs):
        embeds = self.embedding(inputs).view((1, -1))
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out)
        return log_probs

class TrainWord2vec(object):
    def __init__(self,config=DemoConfig):
        self.config=config

    def train(self,dataset,mode=0):
        # init the self.model and self.optimizier
        self.model=mode
        if self.mode==0:
            self.model=NGramLanguageModeler(vocab_size=dataset.source.n_words,
                                            embedding_dim=self.config.input_dim,
                                            context_text=2)
        else:
            self.model=NGramLanguageModeler(vocab_size=dataset.target.n_words,
                                            embedding_dim=self.config.output_dim,
                                            context_text=2)

        if self.config.USE_CUDA:
            self.model.cuda()
        self.optimizier=self.config.optimizier(self.model.parameters(),
                                               lr=self.config.learning_rate)
        self.critirion=nn.NLLLoss()


        # training
        for epoch in range(self.config.n_epochs):
            training_index=dataset.get_index_sample()
            print self.step(training_index)


    def step(self,training_index):
        souce_sentence=training_index[self.mode]
        if not self.config.USE_CUDA:
            trigrams_var =[(torch.autograd.Variable(torch.LongTensor(
                [source_sentence[i],source_sentence[i + 1]])),
                            torch.autograd.Variable(torch.LongTensor(
                                [source_sentence[i + 2]])))
                           for i in range(len(source_sentence) - 2)]
        if self.config.USE_CUDA:
            trigrams_var=[(torch.autograd.Variable(torch.LongTensor(
                [source_sentence[i],source_sentence[i + 1]])).cuda(),
                            torch.autograd.Variable(torch.LongTensor(
                                [source_sentence[i + 2]])).cuda())
                           for i in range(len(source_sentence) - 2)]
        total_loss = torch.Tensor([0])
        for context_var, target_var in trigrams_var:
            self.sourcemodel.zero_grad()
            log_probs = self.sourcemodel(context_var)
            loss = self.critirion(log_probs,target_var)
            loss.backward()
            self.optimizier.step()
            if not self.config.USE_CUDA:
                total_loss += loss.data
            else:
                total_loss += loss.cpu.data
        return total_loss/len(trigrams_var)

if __name__ == '__main__':
    dataset=Dataset()
    train=TrainWord2vec()
    train.train(dataset)


