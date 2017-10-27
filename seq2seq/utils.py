#-*- coding: UTF-8 -*-

from __future__ import print_function
import jieba
import nltk
import codecs
import re
import unicodedata
import os

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


SOS_token = 0
EOS_token = 1

class Lang:
    def __init__(self, name):
        if not name:
            name="Unknown"
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2 # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1
    def getIndex(self,sentence):
        return [self.word2count(word) for word in sentence.split(' ')]



def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize(u'NFD',unicode(s))
        if unicodedata.category(c) != u'Mn'
    )




def _preprocess_sgm(line, is_sgm):
    """Preprocessing to strip tags in SGM files."""
    if not is_sgm:
        return line
    # In SGM files, remove <srcset ...>, <p>, <doc ...> lines.
    if line.startswith("<srcset") or line.startswith("</srcset"):
        return ""
    if line.startswith("<refset") or line.startswith("</refset"):
        return ""
    if line.startswith("<doc") or line.startswith("</doc"):
        return ""
    if line.startswith("<p>") or line.startswith("</p>"):
        return ""
    # Strip <seg> tags.
    line = line.strip()
    if line.startswith("<seg") and line.endswith("</seg>"):
        i = line.index(">")
        return line[i+1:-6]  # Strip first <seg ...> and last </seg>.


def tokenize(line, is_sgm=False, is_zh=False, lower_case=True, delim=' '):
    # strip sgm tags if any
    _line = _preprocess_sgm(line, is_sgm)
    # replace non-breaking whitespace
    _line = _line.replace(u"\xa0", " ").strip()
    # tokenize
    _tok = jieba.cut(_line.rstrip('\r\n')) if is_zh else nltk.word_tokenize(
        _line)
    _tokenized = delim.join(_tok)
    # lowercase. ignore if chinese.
    _tokenized = _tokenized.lower() if lower_case and not is_zh else _tokenized
    return _tokenized


def tokenized(filepath):
    tokenized = ''
    flag_sgm = filepath.endswith('.sgm')
    flag_zh  = filepath.endswith('.zh') or filepath.endswith('.zh.sgm') or filepath.endswith('.zh.rate')
    flag_lowwer = not flag_zh
    with codecs.open(filepath,'rb',encoding='utf-8') as f:
        for index,line in enumerate(f):
            _tokenized = tokenize(line, flag_sgm, flag_zh, flag_lowwer, ' ')
            tokenized+="%s\n" % _tokenized
            if index % 2000 == 0:
                _tokenizer_name = "jieba" if flag_zh else "nltk.word_tokenize"
                logger.info("     [%d] %s: %s" % (index, _tokenizer_name, line))
    return tokenized

def write_ob(filename,s):
    with codecs.open(filename,'w',encoding='utf-8') as f:
        f.write(s)


def tokenizedAndSave(filename,savepath):
    data=tokenized(filename)
    write_ob(savepath,data)


def readLanguages(lang1,lang2):
    source_lang=Lang(name="source")
    target_lang=Lang(name="target")
    pairs=[]
    with codecs.open(lang1,
                     'rb',
                     encoding='utf-8') as f1,codecs.open(lang2,
                                                         'rb',
                                                         encoding='utf-8') as f2:
        lines1=f1.readlines()
        lines2=f2.readlines()
        lines1=[line.strip() for line in lines1]
        lines2=[line.strip() for line in lines2]
        assert (len(lines1)==len(lines2))
        for line1,line2 in zip(lines1,lines2):
            #TODO: hint not normalizing
            pair =[line1,line2]

            source_lang.addSentence(pair[0])
            target_lang.addSentence(pair[1])
            pairs.append(pair)

    print("Read {} sentence pairs".format(len(pairs)))
    return source_lang,target_lang,pairs







