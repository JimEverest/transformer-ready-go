

import tensorflow as tf
from utils import positional_encoding

from h_multihead import scaled_dot_product_attention,MultiHeadAttention,point_wise_feed_forward_network
from utils import positional_encoding,create_padding_mask, create_look_ahead_mask, CustomSchedule, create_masks

#region----------------------------------  4 The encoder ---------------------------------------

class EncoderLayer(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads, dff, rate=0.1):
    super(EncoderLayer, self).__init__()

    self.mha = MultiHeadAttention(d_model, num_heads)
    self.ffn = point_wise_feed_forward_network(d_model, dff)

    self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    
    self.dropout1 = tf.keras.layers.Dropout(rate)
    self.dropout2 = tf.keras.layers.Dropout(rate)
    
  def call(self, x, training, mask):
    
    '''
    Very importantly, in encoder layers, you generate "query", "key", and "value" from the same input sentences. 
    That is why the three inputs of the MultiHeadAttention() class below are all 'x.'

    The part 'self.layernorm1(x + attn_output)' means you apply a layer normalization with 
    an input through the residual connection. 
    
    You should also keep it in mind that the outputs of all the parts have the same shape. 
    '''
    
    attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
    attn_output = self.dropout1(attn_output, training=training)
    out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)
    
    ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
    ffn_output = self.dropout2(ffn_output, training=training)
    out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)
    
    return out2


class Encoder(tf.keras.layers.Layer):
  def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
               maximum_position_encoding, rate=0.1):
    super(Encoder, self).__init__()

    self.d_model = d_model
    self.num_layers = num_layers
    
    self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
    self.pos_encoding = positional_encoding(maximum_position_encoding, 
                                            self.d_model)
    
    
    self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate) 
                       for _ in range(num_layers)]
  
    self.dropout = tf.keras.layers.Dropout(rate)
        
  def call(self, x, training, mask):

    seq_len = tf.shape(x)[1]

    '''
    Fisrst you convert integers which denote words into d_model dimensional vectors
    with an embedding layer, as I explained in the first article. 
    
    I don't know why, but you multiply the embeddnig layer by √d_model, according to the original paper. 
    
    You just add positional encodng to the input x, depending on the length of input sentences so that 
    Transformer can learn relative and definite positions of input tokens, as I explained in the last article.
    That is equal to cropping the heat map in the last article and adding it to the each (input_seq_len, d_model)
    sized matrix. 
    
    You also apply a dropout to mitigate overfitting. 
    '''
    
    x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
    x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    x += self.pos_encoding[:, :seq_len, :]
    x = self.dropout(x, training=training)
    
    
    '''
    You put the input through all the encoder layers in the loop below. 
    After each loop, you can keep the shape (batch_ size, input_seq_len, d_model). 
    '''
    
    for i in range(self.num_layers):
      x = self.enc_layers[i](x, training, mask)
    
    return x  #(batch_ size, input_seq_len, d_model)





# num_layers = 4
# d_model = 128
# dff = 512
# num_heads = 4
# batch_size = 64
# vocab_size = 10000 + 2

def test41():
  sample_encoder_layer = EncoderLayer(d_model, num_heads, dff)

  '''
  Let the maximum length of sentences be 37 . 
  In this case, a sentence is nodenoted as a matrix with the size of (37, d_model=128). 
  '''
  sample_input = tf.random.uniform((batch_size, 37, d_model))
  sample_encoder_layer_output = sample_encoder_layer(sample_input, False, None)

  print(sample_encoder_layer_output.shape)  # (batch_size, input_seq_len, d_model)
  # (64, 37, 128)



def test42():
  sample_encoder = Encoder(num_layers=num_layers, d_model=d_model, num_heads=num_heads, dff=dff, input_vocab_size=vocab_size, maximum_position_encoding=10000)
  temp_input = tf.random.uniform((64, 37), dtype=tf.int64, minval=0, maxval=200)

  sample_encoder_output = sample_encoder(temp_input, training=False, mask=None)

  '''
  You can see that the shape of the output of the Encoder() class is the same as that of the
  EncoderLayer() class. 

  In this case, all the input sentences are denoded as a matrix with a size of (37, d_model=128), 
  And Transformer model keeps converting input sentences, layer by layer, keeping its original 
  shape at the end of each layer. 
  '''

  print (sample_encoder_output.shape)  # (batch_size, input_seq_len, d_model)
  # (64, 37, 128)

#endregion


