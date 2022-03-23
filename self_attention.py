import collections
import logging
import os
import pathlib
import re
import string
import sys
import time

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf




def create_padding_mask(seq):
  seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

  # add extra dimensions to add the padding
  # to the attention logits.
  return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


def create_look_ahead_mask(size):
  mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
  return mask  # (seq_len, seq_len)


def scaled_dot_product_attention(q, k, v, mask):
  """Calculate the attention weights.
  q, k, v must have matching leading dimensions.
  k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
  The mask has different shapes depending on its type(padding or look ahead) 
  but it must be broadcastable for addition.

  Args:
    q: query shape == (..., seq_len_q, depth)
    k: key shape == (..., seq_len_k, depth)
    v: value shape == (..., seq_len_v, depth_v)
    mask: Float tensor with shape broadcastable 
          to (..., seq_len_q, seq_len_k). Defaults to None.

  Returns:
    output, attention_weights
  """

  matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)    
  #([1, 8, 9, 64]) x ([1, 8, 9, 64])T = [1, 8, 9, 9]

  # scale matmul_qk
  dk = tf.cast(tf.shape(k)[-1], tf.float32)  #64
  scaled_attention_logits = matmul_qk / tf.math.sqrt(dk) # [1, 8, 9, 9]

  # add the mask to the scaled tensor.
  if mask is not None:
    scaled_attention_logits += (mask * -1e9)  

  # softmax is normalized on the last axis (seq_len_k) so that the scores
  # add up to 1.
  attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  
  # (..., seq_len_q, seq_len_k)   # bs, heads, h, w <--- CALCULATE Softmax along with W axis (sum=1).

  output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v) # [1, 8, 9, 64] 

  return output, attention_weights # [1, 8, 9, 64] [1, 8, 9, 9]



class MultiHeadAttention(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads):
    super(MultiHeadAttention, self).__init__()
    self.num_heads = num_heads
    self.d_model = d_model

    assert d_model % self.num_heads == 0

    self.depth = d_model // self.num_heads

    self.wq = tf.keras.layers.Dense(d_model)
    self.wk = tf.keras.layers.Dense(d_model)
    self.wv = tf.keras.layers.Dense(d_model)

    self.dense = tf.keras.layers.Dense(d_model)

  def split_heads(self, x, batch_size):  # x:(1,9,512) , batch_size: 1
    """Split the last dimension into (num_heads, depth).
    Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
    """
    x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
    return tf.transpose(x, perm=[0, 2, 1, 3]) #(bs,tokens,heads,dims) 1,9,8,64 --> (bs,heads,tokens,dims)  1,8,9,64

  def call(self, v, k, q, mask):
    print("Inside 'MultiHeadAttention' class...")
    batch_size = tf.shape(q)[0]

    print()
    print("The shape of 'q' is " + str(q.shape))
    print("The shape of 'k' is " + str(k.shape))
    print("The shape of 'v' is " + str(v.shape))
    
    q = self.wq(q)  # (batch_size, seq_len, d_model)
    k = self.wk(k)  # (batch_size, seq_len, d_model)
    v = self.wv(v)  # (batch_size, seq_len, d_model)
    
    print()
    print("After passing 'q', 'k', 'v' through densely connected layers....")
    print("The shape of 'q' is " + str(q.shape)) # (1, 9, 512)
    print("The shape of 'k' is " + str(k.shape)) # (1, 9, 512)
    print("The shape of 'v' is " + str(v.shape)) # (1, 9, 512)

 
    q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth) ([1, 8, 9, 64])
    k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth) ([1, 8, 9, 64])
    v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth) ([1, 8, 9, 64])
    
    print()
    print("After splitting the heads....")
    print("The shape of 'q' is " + str(q.shape))
    print("The shape of 'k' is " + str(k.shape))
    print("The shape of 'v' is " + str(v.shape))

    # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
    # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
    scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask) # 1,8,9,64    [bs, heads, tokens, dims]

    
    print()
    print("The shape of 'attention_weights' is " + str(attention_weights.shape))


    print("The shape of 'scaled_attention' is " + str(scaled_attention.shape))
    scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  
    # [bs, heads, len_tokens, dims]([1, 8, 9, 64]) ---> (batch_size, seq_len_q, num_heads, depth)   ([1, 9, 8, 64])
    
    print()
    print("After transposing....")
    print("The shape of 'scaled_attention' is " + str(scaled_attention.shape))
    concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))  
    # (batch_size, seq_len_q, d_model)  ([1, 9, 8, 64])---> ([1, 9, 512])
    
    print()
    print("The shape of 'concat_attention' is " + str(concat_attention.shape)) #([1, 9, 512])
    
    output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model) #([1, 9, 512])
    print()
    print("The shape of 'output' is " + str(output.shape))

    return output, attention_weights


if (__name__ == "__main__"):
    # temp_mha = MultiHeadAttention(d_model=512, num_heads=8)
    # sample_sentence = tf.random.uniform((1, 9, 512))  # (batch_size, encoder_sequence, d_model)
    # out, attn = temp_mha(v=sample_sentence, k=sample_sentence, q=sample_sentence, mask=None)


    # x1,x2 = out.shape, attn.shape
    # sample_query = np.arange(1*9*512).reshape((1, 9, 512)) + 1
    # sample_query = tf.convert_to_tensor(sample_query)
    # print(sample_query)

    # sample_query = tf.reshape(sample_query, (1, 9, 8, 64))
    # sample_query = tf.transpose(sample_query, perm=[0, 2, 1, 3])
    # print(sample_query)

    # Let's take a sample sentence as an example. 
    # As you can see in the output below, 'sample_sentence' is virutally a (9, 64) sized 
    # matirix, whose elements are simple sequential integers. 
    # *The first axis denotes the index of bathes. In this example the batch size is 1.


    # As I mentioned, "queries" can be in different language from "keys" or "values."
    # * They are supposed to be different in translation tasks. 

    # In this case you compare "quries" in the target language, with the "keys" in the original language. 
    # And after that you reweight "values" in the original language. 

    # Usually, the numbef or "queries" is different from that of "keys" or "values." because 
    # translated sentences usually have different number of tokens. 

    # Let's see an example where the number of input sentence is 9 and that of the translated sentence is 12. 
    temp_mha = MultiHeadAttention(d_model=512, num_heads=8)
    sample_sentence_source_lang = tf.random.uniform((1, 9, 512))  # (batch_size, encoder_sequence, d_model)
    sample_sentence_target_lang = tf.random.uniform((1, 12, 512))  # (batch_size, encoder_sequence, d_model)
    out, attn = temp_mha(v=sample_sentence_source_lang, k=sample_sentence_source_lang, q=sample_sentence_target_lang, mask=None)









