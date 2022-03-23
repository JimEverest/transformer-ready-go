

#region -------------------------2 -----------------------------------


import tensorflow as tf
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.model_selection import train_test_split
import unicodedata
import re
import os
import io
import time
from bpemb import BPEmb


bpemb_de = BPEmb(lang='de', vs=10000, dim=100)
bpemb_en = BPEmb(lang='en', vs=10000, dim=100)

# From the corpus, you extract only the texts necessary. 
path_to_file = "./data/deu.txt"
lines = io.open(path_to_file, encoding='UTF-8').read().strip().split('\n')
temp_list = []
corpus = []
for i in range(len(lines)):
    temp_list =  lines[i].split('\t')[:-1]
    corpus.append(temp_list)
en, de = np.array(corpus).T

en_encoded = []
de_encoded = []

cnt_en = 0
cnt_de = 0

# You encode the sentences with fewer than 40 words in each row into 
# a list of integers, and you append [10000] (<start>) and [10001] (<end>) 
# at the beginning and the end of each sentence. 
for i in range(len(en)):
    en_encoded_temp = bpemb_en.encode_ids(en[i])
    de_encoded_temp = bpemb_de.encode_ids(de[i])
    
    if (len(en_encoded_temp)<=40) and (len(de_encoded_temp)<=40):
        en_encoded.append([10000] + en_encoded_temp + [10001])
        de_encoded.append([10000] + de_encoded_temp + [10001])

# Zero padding the encoded corpus. 
en_padded = tf.keras.preprocessing.sequence.pad_sequences(en_encoded, padding='post')
de_padded = tf.keras.preprocessing.sequence.pad_sequences(de_encoded, padding='post')


# Splitting the corpus into traiing and validaiton datasets. 
input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(en_padded, de_padded, test_size=0.2)


# (data set size, length of the longest sentence)
print(input_tensor_train.shape, input_tensor_val.shape, target_tensor_train.shape, target_tensor_val .shape)
#(199338, 41) (49835, 41) (199338, 42) (49835, 42)


BUFFER_SIZE = len(input_tensor_train)
BATCH_SIZE = 64
steps_per_epoch = len(input_tensor_train)//BATCH_SIZE
embedding_dim = 256
units = 1024
vocab_inp_size = 10000 + 2
vocab_tar_size = 10000 + 2

# You get an iterator for training the network. 
dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train)).shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

# A sample batch. 
sample_data_pair = next(iter(dataset))
sample_data_pair

'''
(<tf.Tensor: shape=(64, 41), dtype=int32, numpy=
 array([[10000,  5451,  1616, ...,     0,     0,     0],
        [10000,   386,  2689, ...,     0,     0,     0],
        [10000,  2088,  6838, ...,     0,     0,     0],
        ...,
        [10000,   386,  9937, ...,     0,     0,     0],
        [10000,   391,    73, ...,     0,     0,     0],
        [10000,   509,   536, ...,     0,     0,     0]], dtype=int32)>,
 <tf.Tensor: shape=(64, 42), dtype=int32, numpy=
 array([[10000,   153,    83, ...,     0,     0,     0],
        [10000,  3077,     5, ...,     0,     0,     0],
        [10000,  4104,   284, ...,     0,     0,     0],
        ...,
        [10000,  3077,  6331, ...,     0,     0,     0],
        [10000,    19,   115, ...,     0,     0,     0],
        [10000,   249,  1503, ...,     0,     0,     0]], dtype=int32)>)
'''

# You can see that each row of the batch corresponds to a sentence. 
# The first row, and its decoded sentence in English. 
sample_sentence_en = sample_data_pair[0][0]
print(sample_sentence_en)
sample_sentence_en = sample_sentence_en.numpy()
sample_sentence_en= sample_sentence_en[np.where(sample_sentence_en!=0)][1:-1]
print(bpemb_en.decode_ids(sample_sentence_en))

'''
tf.Tensor(
[10000  5451  1616  9937  9915  1220  4451   352    42  3687   756  6110
  9967 10001     0     0     0     0     0     0     0     0     0     0
     0     0     0     0     0     0     0     0     0     0     0     0
     0     0     0     0     0], shape=(41,), dtype=int32)
why don't you want me to tell anybody?
'''
# The first row, and its decoded sentence in German. 
sample_sentence_de = sample_data_pair[1][0]
print(sample_sentence_de)
sample_sentence_de = sample_sentence_de.numpy()
sample_sentence_de = sample_sentence_de[np.where(sample_sentence_de!=0)][1:-1]
print(bpemb_de.decode_ids(sample_sentence_de))

'''
tf.Tensor(
[10000   153    83   865  3077   234  2377   632  8005    50  9223  9974
 10001     0     0     0     0     0     0     0     0     0     0     0
     0     0     0     0     0     0     0     0     0     0     0     0
     0     0     0     0     0     0], shape=(42,), dtype=int32)
warum soll ich es denn niemandem sagen?
'''

#endregion


