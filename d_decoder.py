
import tensorflow as tf
# from self_attention import MultiHeadAttention
from h_multihead import scaled_dot_product_attention,MultiHeadAttention,point_wise_feed_forward_network
from utils import positional_encoding,create_padding_mask, create_look_ahead_mask, CustomSchedule, create_masks
#region ----------------------------------  5 DecoderLayer  + Decoder---------------------------------------

class DecoderLayer(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads, dff, rate=0.1):
    super(DecoderLayer, self).__init__()

    self.mha1 = MultiHeadAttention(d_model, num_heads)
    self.mha2 = MultiHeadAttention(d_model, num_heads)

    self.ffn = point_wise_feed_forward_network(d_model, dff)
 
    self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    
    self.dropout1 = tf.keras.layers.Dropout(rate)
    self.dropout2 = tf.keras.layers.Dropout(rate)
    self.dropout3 = tf.keras.layers.Dropout(rate)
    
    
  def call(self, x, enc_output, training, 
           look_ahead_mask, padding_mask):
    # enc_output.shape == (batch_size, input_seq_len, d_model)
    
    '''
    You can generate all the "queries", "keys", and "values" from the same target sentence. 
    And you apply look ahead mask to the first multi-head attention of the decoder part. 
    '''
    attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)  # (batch_size, target_seq_len, d_model)
    attn1 = self.dropout1(attn1, training=training)
    out1 = self.layernorm1(attn1 + x)
    
    '''
    Very importatnly, you generate only the "queries" from the outputs of the encoder part. 
    You apply normal padding mask to the second multi-head attention of the decoder part. 
    '''
    attn2, attn_weights_block2 = self.mha2(
        enc_output, enc_output, out1, padding_mask)  # (batch_size, target_seq_len, d_model)
    attn2 = self.dropout2(attn2, training=training)
    out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, d_model)
    
    ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
    ffn_output = self.dropout3(ffn_output, training=training)
    out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, d_model)
    
    return out3, attn_weights_block1, attn_weights_block2


class Decoder(tf.keras.layers.Layer):
  def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size,
               maximum_position_encoding, rate=0.1):
    super(Decoder, self).__init__()

    self.d_model = d_model
    self.num_layers = num_layers
    
    #self.pre_embedding = tf.keras.layers.Dense(target_vocab_size, 100)
    
    self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
    self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)
    
    self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate) 
                       for _ in range(num_layers)]
    self.dropout = tf.keras.layers.Dropout(rate)
    
  def call(self, x, enc_output, training, 
           look_ahead_mask, padding_mask):
    #print("The shape of 'x' is " + str(tf.shape(x)))

    seq_len = tf.shape(x)[1]
    #print("'seq_len' is " + str(seq_len))
    
    attention_weights = {}
    
    #x = self.pre_embedding(x)

    x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
    x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    x += self.pos_encoding[:, :seq_len, :]
    
    x = self.dropout(x, training=training)

    for i in range(self.num_layers):
      x, block1, block2 = self.dec_layers[i](x, enc_output, training,
                                             look_ahead_mask, padding_mask)
      
      attention_weights['decoder_layer{}_block1'.format(i+1)] = block1
      attention_weights['decoder_layer{}_block2'.format(i+1)] = block2
    
    # x.shape == (batch_size, target_seq_len, d_model)
    #print("The shape of 'x' is " + str(tf.shape(x)))
    
    return x, attention_weights

# num_layers = 4
# d_model = 128
# dff = 512
# num_heads = 4
# batch_size = 64
# vocab_size = 10000 + 2

# # You need an encoder output for the decoder.
# sample_encoder = Encoder(num_layers=num_layers, d_model=d_model, num_heads=num_heads, 
#                          dff=dff, input_vocab_size=vocab_size,
#                          maximum_position_encoding=10000)
# temp_enc_input = tf.random.uniform((64, 37), dtype=tf.int64, minval=0, maxval=200)

# sample_encoder_output = sample_encoder(temp_enc_input, training=False, mask=None)



# sample_decoder = Decoder(num_layers=num_layers, d_model=d_model, num_heads=num_heads, 
#                          dff=dff, target_vocab_size=vocab_size,
#                          maximum_position_encoding=10000)

# temp_dec_input = tf.random.uniform((64, 39), dtype=tf.int64, minval=0, maxval=200)

# output, attn = sample_decoder(temp_dec_input, 
#                               enc_output=sample_encoder_output, 
#                               training=False,
#                               look_ahead_mask=None, 
#                               padding_mask=None)
# '''
# You can see that the decoder alaso gives out outputs like those of the encoder.
# '''
# output.shape
# # TensorShape([64, 39, 128])

# #endregion
