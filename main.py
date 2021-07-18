from io import open
import unicodedata
import string
import re
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import optimizer_torch as optim
import criterion_torch as crit
from device_set_torch import device_set
from data_processing import prepareData
from model import *
from train import trainIters
from evaluation import evaluateRandomly
from attn_visualizing import evaluateAndShowAttention


print(device_set())

input_lang, output_lang, pairs_input_lang, pairs_output_lang = prepareData('en', 'de', 'train', True )
print(random.choice(pairs_input_lang))
print(random.choice(pairs_output_lang))


hidden_size = 256
encoder1 = EncoderRNN(input_lang.n_words, hidden_size).to(device_set())
attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1).to(device_set())


trainIters(input_lang, output_lang, pairs_input_lang, pairs_output_lang, encoder1, attn_decoder1, 75000, print_every=5000)


input_lang_test, output_lang_test, pairs_input_lang_test, pairs_output_lang_test = prepareData('en','de','test', True)
evaluateRandomly(input_lang, output_lang, pairs_input_lang_test, pairs_output_lang_test, encoder1, attn_decoder1)


evaluateAndShowAttention(random.choice(pairs_input_lang))
