import io
import os
import codecs
from collections import Counter, defaultdict
from itertools import chain, count
from config import data_path
import torch.utils.data
import jieba

def extract_features(tokens):
    "Given a list of token separate out words and features (if any)."
    if not tokens:
        return [], [], -1
    split_tokens = [token.split(u"￨") for token in tokens]
    split_tokens = [token for token in split_tokens if token[0]]
    token_size = len(split_tokens[0])
    assert all(len(token) == token_size for token in split_tokens), \
        "all words must have the same number of features"
    words_and_features = list(zip(*split_tokens))
    words = words_and_features[0]
    features = words_and_features[1:]
    return words, features, token_size - 1

class Dataset(torch.utils.data.Dataset):
    """
    # TODO: add docstring
    """
    # TODO: Construct function
    def __init__(self,spath,tpath,opt,**kwargs):

        pass
    def _read_corpus_file(self,path,truncate,segmentation=False):
        """
        path : location of the source or target file
        truncate : maximum sequence_length
        returns: (word, features, nfeat) triples for each line
        """
        if segmentation:
            with codecs.open(path,"r","utf-8") as corpus_file:
                lines=(jieba.cut(line,cut_all=True) for line in corpus_file)
        else:
            with codecs.open(path,"r","utf-8") as corpus_file:
                lines=(line.split() for line in corpus_file)
                if truncate:
                    lines=(line[:truncate] for line in lines)
                for line in lines:
                    yield extract_features(line)

    # TODO: get item function
    def __getitem__(self,i):
        pass
    
    # TODO: get length of the dataset
    def __len__(self):
        pass

    # TODO: iterator
    def __iter__(self):
        pass
    # TODO: get attr
    def __getattr__(self,attr):
        pass




if __name__ == '__main__':
    dataset=Dataset()
    print dataset._read_corpus_file(os.path.abspath(os.path.join(data_path,"train.en")))
