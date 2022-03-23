
#region ----------------------------------  3  ---------------------------------------

import tensorflow as tf
import time
import numpy as np



from c_encoder import Encoder
from d_decoder import Decoder


class Transformer(tf.keras.Model):
  def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, target_vocab_size, pe_input, pe_target, rate=0.1):
    super(Transformer, self).__init__()

    self.encoder = Encoder(num_layers, d_model, num_heads, dff, input_vocab_size, pe_input, rate)

    self.decoder = Decoder(num_layers, d_model, num_heads, dff, target_vocab_size, pe_target, rate)

    self.final_layer = tf.keras.layers.Dense(target_vocab_size)
    
  def call(self, inp, tar, training, enc_padding_mask, look_ahead_mask, dec_padding_mask):

    enc_output = self.encoder(inp, training, enc_padding_mask)  # (batch_size, inp_seq_len, d_model)
    
    '''
    The output of the last layer of the encoder is passed to all the layers of the decoder. 
    '''
    #dec_output.shape == (batch_size, tar_seq_len, d_model)
    dec_output, attention_weights = self.decoder(tar, enc_output, training, look_ahead_mask, dec_padding_mask)
    
    '''
    Tee final part of Transformer model. In case of machine translation, you predict a 
    'target_vocab_size' dimensional vector at every potition of the target sentence. 
    '''
    final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)
    
    return final_output, attention_weights




def test3():
  num_layers = 4
  d_model = 128
  dff = 512
  num_heads = 8

  input_vocab_size = 10000 + 2
  target_vocab_size = 10000 + 2
  dropout_rate = 0.1

  sample_transformer = Transformer(num_layers, d_model, num_heads, dff,
                            input_vocab_size, target_vocab_size, 
                            pe_input=input_vocab_size, 
                            pe_target=target_vocab_size,
                            rate=dropout_rate)

  # Let's put in sample inputs and targets in the sample Transformer model. 
  # In this case, the max length of the input sentences is 38, and that of targets is 37. 
  # In practice, all the elements of 'sample_input' and 'sample_target'  are integers. 
  sample_input = tf.random.uniform((64, 38), dtype=tf.int64, minval=0, maxval=200)
  sample_target = tf.random.uniform((64, 37), dtype=tf.int64, minval=0, maxval=200)


  fn_out, _ = sample_transformer(sample_input, sample_target, training=False, 
                                enc_padding_mask=None, 
                                look_ahead_mask=None,
                                dec_padding_mask=None)

  print(fn_out.shape)  # (batch_size, tar_seq_len, target_vocab_size)
  #(64, 37, 10002)
  # As you can see, each target entences is a (37, 10002) sized matrix. 
  #endregion
                    