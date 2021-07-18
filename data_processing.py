
import unicodedata
import re
import random
import torch
from device_set_torch import device_set

SOS_token = 0
EOS_token = 1

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0:"SOS",1:"EOS"}
        self.n_words = 2
    
    def addSentence(self, sentence):
        for word in sentence[0].split(' '):
            self.addWord(word)
    
    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1
    

def unicodeToAscii(s):
    return ''.join(
    c for c in unicodedata.normalize('NFD', s)
    if unicodedata.category(c) != 'Mn'
    )


def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

#dataset에 맞게 변환 필요 - 변환 완료
def readLangs(lang1, lang2, datasetType, reverse=False):
    print("Reading lines...")

    lines_lang1 = open('wmt16/%s.%s' % (datasetType, lang1), encoding='utf-8').\
        read().strip().split('\n')
    lines_lang2 = open('wmt16/%s.%s' % (datasetType, lang2), encoding='utf-8').\
        read().strip().split('\n')

    pairs_input_lang = [[normalizeString(s) for s in l.split('\t')] for l in lines_lang1]
    pairs_output_lang = [[normalizeString(s) for s in l.split('\t')] for l in lines_lang2]

    if reverse:
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs_input_lang, pairs_output_lang


def prepareData(lang1, lang2, datasetType, reverse=False):
    input_lang, output_lang, pairs_input_lang, pairs_output_lang = readLangs(lang1, lang2, datasetType, reverse)
    print("Read %s sentence pairs_en" % len(pairs_input_lang))
    print("Read %s sentence pairs_de" % len(pairs_output_lang))
    print("Trimmed to %s sentence pairs_en" % len(pairs_input_lang))
    print("Trimmed to %s sentence pairs_de" % len(pairs_output_lang))
    print("Counting words...")
    for pair in pairs_input_lang, pairs_output_lang:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs_input_lang, pairs_output_lang


def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]


def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device_set()).view(-1, 1)


def tensorsFromPair(input_lang, output_lang, pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)
