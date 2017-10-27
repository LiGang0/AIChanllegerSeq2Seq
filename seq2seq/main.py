
from config import DemoConfig
from model.model1 import EncoderRNN,AttnDecoderRNN
from dataprocessor import Dataset

if __name__ == '__main__':
    dataset=Dataset(config=DemoConfig)
    encoder=EncoderRNN(input_size=dataset.source.n_words,hidden_size=DemoConfig.hidden_size)