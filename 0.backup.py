

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


num_layers = 4
d_model = 128
dff = 512
num_heads = 4
batch_size = 64
vocab_size = 10000 + 2

sample_encoder_layer = EncoderLayer(d_model, num_heads, dff)

'''
Let the maximum length of sentences be 37 . 
In this case, a sentence is nodenoted as a matrix with the size of (37, d_model=128). 
'''
sample_input = tf.random.uniform((batch_size, 37, d_model))
sample_encoder_layer_output = sample_encoder_layer(sample_input, False, None)

print(sample_encoder_layer_output.shape)  # (batch_size, input_seq_len, d_model)
# (64, 37, 128)


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




#region----------------------------------  translator_ffn_wp ---------------------------------------
import tensorflow as tf

def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
      tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
  ])

d_model = 128
dff = 512
batch_size = 64
input_seq_len = 37

sample_input = tf.ones((batch_size, input_seq_len, d_model))
print(sample_input.shape)

# (64, 37, 128)

sample_ffn = point_wise_feed_forward_network(d_model, dff)
sample_ffn.build(sample_input.shape)
sample_ffn.summary()
# Model: "sequential"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #   
# =================================================================
# dense (Dense)                (64, 37, 512)             66048     
# _________________________________________________________________
# dense_1 (Dense)              (64, 37, 128)             65664     
# =================================================================
# Total params: 131,712
# Trainable params: 131,712
# Non-trainable params: 0
# _________________________________________________________________

# You can see that the number of parameters only depends on 
# 'd_model' and ''dff
512 * (128 + 1)
# 66048
128 * (512 + 1)
# 65664

#endregion



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

num_layers = 4
d_model = 128
dff = 512
num_heads = 4
batch_size = 64
vocab_size = 10000 + 2

# You need an encoder output for the decoder.
sample_encoder = Encoder(num_layers=num_layers, d_model=d_model, num_heads=num_heads, 
                         dff=dff, input_vocab_size=vocab_size,
                         maximum_position_encoding=10000)
temp_enc_input = tf.random.uniform((64, 37), dtype=tf.int64, minval=0, maxval=200)

sample_encoder_output = sample_encoder(temp_enc_input, training=False, mask=None)



sample_decoder = Decoder(num_layers=num_layers, d_model=d_model, num_heads=num_heads, 
                         dff=dff, target_vocab_size=vocab_size,
                         maximum_position_encoding=10000)

temp_dec_input = tf.random.uniform((64, 39), dtype=tf.int64, minval=0, maxval=200)

output, attn = sample_decoder(temp_dec_input, 
                              enc_output=sample_encoder_output, 
                              training=False,
                              look_ahead_mask=None, 
                              padding_mask=None)
'''
You can see that the decoder alaso gives out outputs like those of the encoder.
'''
output.shape
# TensorShape([64, 39, 128])

#endregion


#region ----------------------------------  6 Masking ---------------------------------------

en_padded = tf.keras.preprocessing.sequence.pad_sequences(en_encoded, padding='post')
de_padded = tf.keras.preprocessing.sequence.pad_sequences(de_encoded, padding='post')
def create_padding_mask(seq):
  seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

  # add extra dimensions to add the padding
  # to the attention logits.
  return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

def create_look_ahead_mask(size):
  mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
  return mask  # (seq_len, seq_len)

def create_masks(inp, tar):
  # Encoder padding mask
  enc_padding_mask = create_padding_mask(inp)

  # Used in the 2nd attention block in the decoder.
  # This padding mask is used to mask the encoder outputs.
  dec_padding_mask = create_padding_mask(inp)

  # Used in the 1st attention block in the decoder.
  # It is used to pad and mask future tokens in the input received by 
  # the decoder.
  look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
  dec_target_padding_mask = create_padding_mask(tar)
  combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

  return enc_padding_mask, combined_mask, dec_padding_mask
# When you make a sample batch with the first and the last sentence, the batch in English look slike below. 
en_sample_batch, de_sample_batch = np.stack((en_padded[0], en_padded[-1])), np.stack((de_padded[0], de_padded[-1]))
en_sample_batch
'''
array([[10000,   822,  9935, 10001,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0],
       [10000,   386,  8371,  2234,  5359,  1697,  1860,    26,  3285,
          754,  2161,  1601,  9934,   777,   662,   505,    86,  2383,
           42,  3690,   120,  1294,  1220,   309,  5481,    80,  3581,
           42,    86,  1515,    34,  2254,  9940,  9920,   359,    37,
         9935, 10001,     0,     0,     0]], dtype=int32)
'''
# You can make all the masks needed for the batch with the function create_masks() funciton. 
enc_padding_mask, combined_mask, dec_padding_mask = create_masks(en_sample_batch, de_sample_batch)
# Importantly enc_padding_mask and dec_padding_mask have the same shape. 
enc_padding_mask.shape, dec_padding_mask.shape
(TensorShape([2, 1, 1, 41]), TensorShape([2, 1, 1, 41]))
# The encoder padding mask and the decoder padding mask are the same, 
# and they are quite simple. You can see 
enc_padding_mask
'''
<tf.Tensor: shape=(2, 1, 1, 41), dtype=float32, numpy=
array([[[[0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
          1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
          1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]]],


       [[[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
          0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
          0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1.]]]], dtype=float32)>
          '''
dec_padding_mask
'''
<tf.Tensor: shape=(2, 1, 1, 41), dtype=float32, numpy=
array([[[[0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
          1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
          1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]]],


       [[[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
          0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
          0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1.]]]], dtype=float32)>
          '''
# Let's take a sample tensor with random elements as an example. 
# You just add -1e9 to the elements where quivalent elemets of masks are 1. 
# After applying a softmax functino you can zero pad positions of multi-head attention maps 
# where the input sentences are also zero padded. 
sample_tensor = np.random.rand(2, 8, 41, 41)
sample_tensor = sample_tensor+ (enc_padding_mask * -1e9)
attention_map = tf.nn.softmax(sample_tensor, axis=-1)
# When you print out averything, the attention map of the encoder part looks like this. 
tf.print(attention_map, summarize=-1)
#region output
'''
[[[[0.193826437 0.164843827 0.400757343 0.240572393 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.178208038 0.327145 0.175259694 0.319387227 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.35382843 0.24549368 0.213529289 0.187148675 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.247981757 0.216751724 0.276857913 0.258408546 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.148054138 0.198008567 0.319183141 0.334754139 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.175793216 0.219340697 0.280442446 0.324423611 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.190093711 0.345523119 0.216241658 0.248141512 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.188645929 0.277991205 0.369562477 0.163800374 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.221780017 0.262065 0.177053526 0.339101404 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.260317475 0.244840413 0.166880891 0.327961236 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.196715832 0.266652614 0.205538213 0.331093341 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.237098202 0.401204526 0.1692027 0.192494571 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.189820722 0.340569556 0.247390226 0.222219557 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.211889356 0.248726398 0.282896221 0.256488025 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.158152521 0.20069015 0.344577372 0.296579927 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.360075921 0.178577214 0.29358533 0.167761505 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.169235021 0.319603533 0.194800735 0.316360682 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.213894263 0.297523826 0.216623679 0.271958202 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.320925146 0.184400499 0.341716319 0.152958065 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.202220619 0.317958504 0.317782611 0.162038267 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.180766538 0.227917463 0.356372416 0.234943599 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.327570021 0.254392564 0.184539661 0.233497739 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.376062959 0.179396942 0.172152102 0.272387922 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.22893247 0.290472478 0.248389781 0.232205302 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.366674453 0.249080718 0.206156507 0.178088263 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.286835492 0.268473864 0.137619391 0.307071328 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.157542109 0.300579399 0.182868153 0.359010398 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.185019508 0.336738348 0.222429961 0.255812168 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.207654282 0.319681019 0.245229587 0.227435067 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.21732 0.227517232 0.226259261 0.328903586 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.219700098 0.25249207 0.346147746 0.181660131 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.274140626 0.261587352 0.311829567 0.152442396 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.270720363 0.226424292 0.286913961 0.215941355 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.24379012 0.276546925 0.2788665 0.200796455 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.359666 0.247774065 0.193206087 0.199353799 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.337088883 0.17057249 0.168617234 0.323721319 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.259568125 0.215739444 0.319830626 0.204861775 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.334671021 0.18812871 0.180928707 0.296271592 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.270633161 0.215560481 0.170545533 0.343260795 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.290897965 0.191284776 0.173578948 0.344238281 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.175714239 0.274780273 0.231987745 0.317517757 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]]

  [[0.259669721 0.235206619 0.169025391 0.336098313 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.252866477 0.188540161 0.343123317 0.215470061 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.161976233 0.339576 0.179019198 0.319428563 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.355734259 0.294493973 0.197188675 0.152583078 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.185338825 0.192065641 0.198716193 0.423879266 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.192598209 0.205208749 0.301890582 0.300302505 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.264013052 0.29386121 0.192640409 0.249485299 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.301548183 0.168642417 0.192949519 0.336859852 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.353108257 0.231347561 0.262384176 0.153159991 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.163719788 0.33223322 0.344306827 0.15974021 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.161719 0.229666188 0.367317 0.241297841 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.27450338 0.256369531 0.219643101 0.249484017 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.166295364 0.254947424 0.215870991 0.36288619 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.289236516 0.181542754 0.350148231 0.179072544 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.313597828 0.195779577 0.238204062 0.252418548 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.299737 0.281806767 0.160124362 0.258331835 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.149677753 0.174592957 0.292062104 0.383667171 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.232653275 0.269094884 0.242741808 0.255510032 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.326476067 0.278486222 0.263625264 0.131412446 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.353874683 0.204016238 0.212276965 0.229832157 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.357283324 0.162023783 0.147088811 0.333604068 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.323659867 0.19182761 0.185615808 0.29889673 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.323307931 0.290549487 0.148143932 0.237998664 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.172693729 0.243480638 0.333062351 0.250763267 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.186785 0.305000156 0.202346817 0.30586797 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.352506 0.260572314 0.15289475 0.234026894 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.231638581 0.215495452 0.241078839 0.311787128 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.234792784 0.290209 0.24714075 0.227857545 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.374836624 0.216190144 0.234614938 0.174358323 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.154307589 0.280392319 0.397467375 0.167832658 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.326155782 0.288944602 0.215025783 0.169873863 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.199731275 0.347683221 0.236201644 0.216383904 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.344131202 0.173491791 0.284494638 0.197882429 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.229349151 0.232093766 0.188256979 0.350300133 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.172328129 0.428442121 0.201630592 0.197599128 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.309615076 0.211817 0.216244072 0.262323916 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.24033384 0.34740603 0.178892642 0.233367458 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.307672173 0.326009065 0.214618832 0.151699916 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.230193093 0.264282674 0.358595461 0.146928743 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.322668165 0.171000689 0.284414202 0.221916988 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.329434305 0.176780105 0.284804404 0.208981141 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]]

  [[0.185493767 0.198122948 0.254213 0.362170339 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.367250562 0.223086312 0.20778805 0.201875106 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.205239519 0.27250582 0.284684122 0.237570584 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.317238331 0.163479015 0.274758279 0.24452439 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.185825706 0.353432417 0.271020979 0.189720944 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.320200533 0.167749032 0.171044558 0.341005802 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.244820386 0.191013619 0.32520631 0.238959745 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.343246371 0.303445131 0.185977519 0.167330876 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.291567534 0.167944238 0.279341459 0.261146784 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.339352518 0.242381677 0.210679069 0.207586795 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.307197452 0.241904676 0.224973798 0.225924015 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.125606164 0.325883418 0.256249636 0.292260766 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.276248306 0.203146428 0.188614309 0.331990898 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.331656694 0.133024216 0.234243214 0.301075906 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.163245 0.305247 0.342375129 0.189132795 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.258079052 0.312465698 0.219571158 0.209884062 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.174492866 0.202748835 0.296477228 0.32628113 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.32615298 0.180076569 0.309527278 0.184243098 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.231421798 0.202225372 0.348980397 0.217372447 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.280259252 0.236410409 0.182739928 0.300590396 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.197383165 0.334326833 0.231189832 0.237100139 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.290303469 0.161910549 0.307236463 0.240549535 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.236812264 0.240500048 0.328117758 0.19456993 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.232976452 0.139179364 0.343911707 0.283932507 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.260406137 0.192030743 0.204010978 0.343552142 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.278732032 0.250017434 0.279557943 0.191692606 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.31211248 0.337398052 0.19302465 0.157464698 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.26456666 0.238067195 0.286839336 0.210526869 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.264801383 0.32683596 0.189891145 0.218471542 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.347429752 0.187206045 0.24487102 0.220493138 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.228540942 0.238311172 0.320112288 0.213035569 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.323910534 0.259371191 0.216155186 0.200563118 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.26495865 0.179438218 0.2798253 0.275777847 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.261455297 0.281216 0.294451058 0.162877649 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.146935523 0.311382204 0.298888683 0.242793545 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.362185657 0.147564694 0.33732748 0.152922213 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.266961217 0.295136303 0.201143801 0.236758679 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.321737319 0.210978419 0.263777763 0.203506485 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.441667438 0.172719553 0.17165491 0.213958085 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.26312235 0.190824911 0.230581164 0.31547159 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.316617906 0.338549018 0.143145904 0.201687157 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]]

  [[0.191995382 0.373691201 0.243507877 0.190805584 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.222634867 0.289367586 0.230389968 0.257607549 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.280540258 0.205743372 0.168227822 0.345488459 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.240697876 0.248046815 0.252616704 0.258638561 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.200770885 0.305003732 0.212496191 0.281729192 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.35546577 0.347249836 0.13755469 0.159729704 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.284843534 0.255236983 0.301531911 0.158387527 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.279873669 0.310360193 0.241950542 0.167815685 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.180773377 0.243034676 0.217271239 0.358920693 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.274899423 0.269756287 0.277261525 0.178082734 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.196458191 0.215591982 0.205680609 0.382269174 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.241931468 0.294593811 0.244073123 0.219401553 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.278733343 0.268425 0.228208214 0.224633411 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.316123694 0.186051279 0.286471695 0.211353287 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.250476241 0.266146213 0.29839927 0.184978276 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.192685574 0.348659307 0.268079877 0.190575302 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.222761065 0.194677114 0.263121188 0.319440633 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.182313189 0.183386803 0.368129224 0.26617071 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.238366559 0.253968775 0.269413978 0.238250747 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.273926169 0.285747 0.258551836 0.181774959 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.305278391 0.285779834 0.177091137 0.231850624 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.215503335 0.264669299 0.195248649 0.324578762 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.181700826 0.251243919 0.337246 0.229809195 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.140187979 0.216120124 0.329070359 0.314621508 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.306119174 0.224070475 0.306931108 0.162879273 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.214446515 0.136201575 0.305329084 0.34402293 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.236740783 0.282268465 0.205977499 0.275013238 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.250356227 0.213744029 0.231287032 0.304612696 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.226845965 0.305746078 0.189782679 0.277625322 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.341901332 0.211551785 0.227120176 0.219426706 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.318821579 0.158723861 0.197137013 0.325317472 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.331131 0.203487903 0.189983264 0.275397867 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.331564903 0.212849438 0.273679346 0.181906298 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.331022233 0.223220944 0.188700244 0.257056564 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.227591544 0.24421446 0.246209696 0.281984359 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.335394263 0.170244336 0.218772486 0.2755889 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.185901344 0.251602829 0.319766343 0.242729515 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.17669417 0.314427763 0.280601054 0.228277087 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.164407268 0.213141456 0.311384201 0.311067134 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.258249849 0.206384316 0.30102545 0.234340385 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.30149138 0.221874848 0.289743543 0.186890215 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]]

  [[0.356900185 0.235246584 0.136630535 0.271222681 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.380523086 0.196947411 0.155761361 0.266768128 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.307009339 0.28261295 0.261987239 0.148390427 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.190799266 0.369779587 0.243389368 0.196031749 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.337057859 0.307743251 0.192662105 0.162536889 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.186457142 0.242254019 0.39003107 0.181257799 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.184216633 0.243672937 0.194000795 0.378109604 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.220524 0.236480147 0.223521546 0.31947431 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.238402426 0.264458179 0.307132095 0.190007329 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.161353722 0.160068706 0.390809685 0.287767947 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.195849299 0.418947786 0.213124558 0.172078386 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.363900751 0.284917474 0.180725545 0.170456216 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.236847728 0.270187765 0.141224578 0.351739913 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.21496655 0.254655838 0.264422953 0.265954584 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.245622531 0.174524739 0.248783424 0.33106932 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.253387392 0.263304025 0.162041 0.321267545 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.312977016 0.312826 0.206552207 0.167644694 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.274649709 0.17965661 0.389400095 0.156293586 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.235322833 0.229267478 0.243822172 0.291587502 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.170723557 0.27908209 0.24440743 0.305786878 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.217759117 0.279944897 0.22646904 0.275826961 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.348282427 0.208268672 0.23118791 0.212261051 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.245893687 0.301111042 0.26691407 0.186081141 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.374022484 0.20099394 0.209124252 0.215859294 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.3753483 0.191602781 0.200447 0.232601941 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.207393229 0.37330246 0.208254144 0.211050212 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.351905 0.194843307 0.265278965 0.187972814 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.239225462 0.138383374 0.291076094 0.33131516 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.294001639 0.239438623 0.266964674 0.199595019 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.218297645 0.194701612 0.409008741 0.177992031 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.228231177 0.275666177 0.240025952 0.256076723 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.317287743 0.250996917 0.304328263 0.127387106 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.233993724 0.236966908 0.259471446 0.269567907 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.297250658 0.2321309 0.310274094 0.160344437 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.226933807 0.30641371 0.321087718 0.145564809 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.186763987 0.195810661 0.218665779 0.398759544 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.194278196 0.414192438 0.169073209 0.222456172 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.420706153 0.191105455 0.226401642 0.16178681 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.195123404 0.326400042 0.184221476 0.294255108 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.274518907 0.250750035 0.272686899 0.202044189 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.167599648 0.194754675 0.27298528 0.364660412 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]]

  [[0.189324766 0.416704893 0.202563286 0.191407 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.305300981 0.322520316 0.154247329 0.21793139 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.343405783 0.206425011 0.212102935 0.238066196 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.296425879 0.14732118 0.293549478 0.262703538 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.311299324 0.16992408 0.244659483 0.274117142 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.277965903 0.1800358 0.161976561 0.38002184 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.179694414 0.194878429 0.374340951 0.251086265 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.1507871 0.338818073 0.334711164 0.175683677 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.177961498 0.367230684 0.276964813 0.177843 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.33652702 0.206530377 0.270892829 0.186049834 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.176293939 0.303497881 0.188384876 0.331823289 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.251048803 0.304710358 0.243308261 0.200932562 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.18856521 0.199536294 0.19359 0.418308496 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.186262041 0.256156743 0.278927296 0.27865389 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.306074202 0.16669035 0.228392959 0.29884249 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.141623333 0.173125461 0.32773298 0.357518256 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.206618562 0.358873 0.170003057 0.264505416 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.293888897 0.266784191 0.153237239 0.286089748 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.218887985 0.298317492 0.236442432 0.246352091 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.216209 0.248441398 0.351795822 0.18355374 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.257241666 0.194354683 0.186634839 0.361768782 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.345099211 0.282558858 0.176973239 0.195368692 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.3294608 0.323633879 0.134738892 0.212166533 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.25500226 0.205603272 0.225579321 0.313815147 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.320496112 0.259855837 0.225856513 0.193791568 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.319859475 0.159985453 0.307960391 0.212194666 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.32159403 0.304810673 0.230020747 0.143574491 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.240560576 0.26992327 0.18487452 0.304641575 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.348336339 0.152682662 0.307482481 0.191498429 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.244167253 0.266051471 0.257901847 0.231879473 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.3422167 0.22432515 0.232139081 0.201319039 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.293084 0.35797 0.172346443 0.176599547 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.170029044 0.261442691 0.260917127 0.307611138 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.144309238 0.253568828 0.318481177 0.283640772 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.226188764 0.24988237 0.226723805 0.297205031 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.314582407 0.228127941 0.237046912 0.220242724 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.249863327 0.351317167 0.214000851 0.184818655 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.164351344 0.411845893 0.264082193 0.159720585 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.174134463 0.257311016 0.345411718 0.223142877 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.356755167 0.239080325 0.180569634 0.223594874 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.189709321 0.192281798 0.435766816 0.182242081 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]]

  [[0.378544122 0.151877359 0.31087476 0.158703774 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.343129188 0.2013347 0.144325152 0.31121096 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.252508044 0.233921334 0.270209312 0.243361354 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.282831401 0.243714675 0.276972324 0.196481586 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.142333746 0.37317571 0.219657108 0.264833391 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.156090334 0.334780365 0.233610585 0.275518745 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.30108127 0.228190839 0.228111774 0.242616087 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.187179759 0.169044912 0.197376192 0.446399122 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.201744691 0.196889579 0.300062954 0.30130282 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.25792715 0.240178198 0.170256376 0.331638247 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.199127972 0.297274917 0.232657701 0.270939469 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.219393015 0.322586685 0.215580359 0.242439941 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.205509052 0.168436 0.306209952 0.319844961 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.358370394 0.241135418 0.167628288 0.232865915 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.25998795 0.232777938 0.318544298 0.188689828 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.22592403 0.229448661 0.283049703 0.261577636 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.167790696 0.261650503 0.266213566 0.30434528 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.289708734 0.291490197 0.194304317 0.224496812 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.187712342 0.335632771 0.237569615 0.239085272 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.204437599 0.392537683 0.187395528 0.215629146 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.231386691 0.210474566 0.24628222 0.311856538 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.210435495 0.222287104 0.282207549 0.285069764 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.156751066 0.183247089 0.297416717 0.362585098 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.188815579 0.314122766 0.324722469 0.172339141 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.221904323 0.281913638 0.288296521 0.207885534 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.300674409 0.233905867 0.218169928 0.247249797 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.267456144 0.291831642 0.219189689 0.22152254 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.221504882 0.209539682 0.350060165 0.218895316 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.219006494 0.179564953 0.322316736 0.279111803 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.313881516 0.205118224 0.216315299 0.264684975 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.181631729 0.264983803 0.226173505 0.327211 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.209841162 0.243686244 0.370415509 0.176057115 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.317377031 0.178609535 0.263479292 0.240534186 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.313636869 0.207836613 0.187652752 0.290873706 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.258125216 0.215043709 0.186901391 0.33992964 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.298632562 0.149369583 0.352520585 0.19947733 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.220128536 0.182892755 0.372920305 0.224058464 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.285872102 0.191990077 0.227903262 0.294234544 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.167964607 0.264172494 0.267473936 0.300388962 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.168259114 0.275817573 0.323117107 0.232806221 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.411852628 0.181079343 0.193660527 0.213407442 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]]

  [[0.184938699 0.317889571 0.213919029 0.283252716 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.152803466 0.213712618 0.274611 0.35887289 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.297453791 0.295639575 0.226470977 0.180435658 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.243089646 0.209877774 0.207634792 0.339397848 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.278788984 0.174227148 0.32465449 0.222329363 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.354245842 0.27486369 0.144018054 0.226872444 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.223745123 0.298847765 0.287982911 0.189424217 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.171004176 0.287606329 0.28578335 0.255606145 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.286007911 0.352951616 0.175631523 0.18540898 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.19344452 0.389170229 0.193798661 0.223586559 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.260139734 0.195870638 0.288570881 0.255418777 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.187754303 0.292045325 0.324892 0.195308298 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.202291846 0.173126653 0.395638973 0.228942573 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.307245225 0.24832572 0.140638188 0.303790867 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.21439676 0.197541103 0.297676831 0.290385365 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.203514054 0.198614448 0.275333792 0.32253775 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.276739 0.17478393 0.281211913 0.267265201 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.303609133 0.145644277 0.334708244 0.216038391 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.153907463 0.319982827 0.279112667 0.246997103 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.225657165 0.253708959 0.231434911 0.289198965 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.309643805 0.17707774 0.257525176 0.255753309 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.260412812 0.205373242 0.254902363 0.279311597 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.345669419 0.183705732 0.266389191 0.204235643 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.258573622 0.211326733 0.228811726 0.301287919 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.177184105 0.273786843 0.264447153 0.2845819 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.152850777 0.378729761 0.265792459 0.202627048 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.215337783 0.216715232 0.299675792 0.268271208 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.213419139 0.372716844 0.226479724 0.187384382 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.279869 0.194580227 0.262465835 0.263084978 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.365791649 0.264292598 0.223075956 0.146839797 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.184166119 0.400081038 0.237619534 0.178133264 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.235228464 0.356623024 0.166108042 0.242040455 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.300988793 0.27935648 0.248000026 0.171654761 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.183164731 0.149537951 0.320677578 0.346619695 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.292884946 0.126094192 0.276720405 0.304300398 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.294261664 0.273148507 0.268519044 0.164070696 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.157458797 0.192998543 0.271585405 0.377957255 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.147685 0.268719167 0.292884856 0.290710956 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.286976814 0.130143315 0.291486114 0.291393697 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.1730593 0.203256279 0.425795048 0.197889373 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   [0.262093484 0.243824169 0.317093581 0.17698881 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]]]


 [[[0.0280612092 0.0197229497 0.0203888547 0.0160829518 0.0168010518 0.0202312078 0.016892219 0.0193095189 0.0235506948 0.0307361279 0.041112259 0.0395015366 0.0177483223 0.0180304963 0.0173536558 0.030140534 0.0351663791 0.0259943623 0.0342159308 0.0244226139 0.0221323594 0.0159613322 0.0271643326 0.0276428536 0.0260293186 0.0274536852 0.0311508104 0.0403515138 0.0252486486 0.034678176 0.0345400795 0.0157480799 0.022336103 0.0175969657 0.0421690121 0.042298317 0.0205641203 0.0314714648 0 0 0]
   [0.0187140871 0.0265361927 0.0282851588 0.0292674042 0.0162479766 0.0364814922 0.0213388074 0.0369261578 0.0153697412 0.0362390094 0.0187740959 0.0185266454 0.0376937762 0.0246010162 0.0249350779 0.0345017761 0.0167427696 0.0271150284 0.0294584 0.0143181244 0.0149173532 0.0216135401 0.0180984139 0.0361131579 0.0197555255 0.0346441269 0.0268519633 0.0326022841 0.0338791274 0.0379413441 0.0194994658 0.0258453246 0.0385192223 0.0243808851 0.0213180501 0.0318844356 0.0303112324 0.0197518207 0 0 0]
   [0.0242921729 0.0185836628 0.0199735221 0.0351415537 0.0293380655 0.0232686438 0.0146664958 0.0386888422 0.029379189 0.0202066172 0.0165323671 0.0166155454 0.026651036 0.028580362 0.0214924887 0.018469777 0.0314735174 0.0300635602 0.0320116766 0.0262478217 0.0208966825 0.0229101889 0.0288093891 0.0329605788 0.0215835348 0.021055121 0.0256289784 0.021330161 0.0229250211 0.0375530422 0.0214390922 0.0344336182 0.03042957 0.0365837514 0.0232625064 0.0375551283 0.0222957954 0.0366710052 0 0 0]
   [0.0310652815 0.0362228788 0.0302253645 0.0297130849 0.0152634215 0.0169160347 0.0339329094 0.031989444 0.0282398239 0.0339160077 0.034580268 0.036145106 0.0297162142 0.0257714596 0.0270643495 0.0365871936 0.027322378 0.0262647234 0.0173237361 0.0234158952 0.0332543962 0.0368248597 0.0260062013 0.0254193619 0.0253463872 0.0159496497 0.0227824654 0.0166056287 0.017591387 0.0192949194 0.0293150265 0.0214374363 0.0176181216 0.0310223531 0.0236396231 0.0150090745 0.0270103887 0.0241972059 0 0 0]
   [0.0153843742 0.015872743 0.0338052772 0.0368484259 0.0347744748 0.0228991117 0.0295353904 0.038977161 0.0315207876 0.026170589 0.019269824 0.0275390223 0.0230250861 0.019940963 0.0274295267 0.0289166532 0.0413261615 0.0193072837 0.015544246 0.0266862381 0.0233444609 0.0209307279 0.0246098675 0.0173890796 0.0387497917 0.0273365062 0.0229543373 0.0341012664 0.0187597424 0.0230420195 0.0182210281 0.0311398413 0.0399882607 0.0330337435 0.0192570258 0.0242936909 0.031350296 0.0167248975 0 0 0]
   [0.041000884 0.0226645637 0.0291280374 0.0356356911 0.0230206177 0.0186293628 0.0214849375 0.0234157853 0.0240355451 0.0382793322 0.0169876907 0.0254018381 0.0174990073 0.0422158167 0.0347734019 0.0265968423 0.0203352757 0.0231840983 0.0269044191 0.0158942267 0.0293248668 0.0242864974 0.0326288827 0.0166667402 0.041592218 0.0194975939 0.0268860329 0.0265994612 0.0259883888 0.0193619914 0.0269664526 0.0178232118 0.0286585763 0.0385520533 0.0234629456 0.0212691203 0.0231153872 0.030232152 0 0 0]
   [0.0218825657 0.0171894748 0.0171558913 0.0154255908 0.0319027305 0.0405059792 0.0187173393 0.0270310324 0.0346363261 0.0407829545 0.0237054713 0.031619437 0.0332578234 0.0163612 0.0163857024 0.0404741913 0.0182797536 0.0308058299 0.0166773666 0.0322063118 0.0303659029 0.0376757197 0.0175056048 0.0369855911 0.0232512169 0.0266008805 0.0204729084 0.0264937952 0.02645896 0.0313443504 0.0346083567 0.0254273526 0.0155203892 0.0195858218 0.0182078667 0.0321423486 0.0280678496 0.0242821164 0 0 0]
   [0.0282949936 0.0244577508 0.0260326099 0.0328404 0.0248551201 0.0345999151 0.0259069558 0.0212665759 0.0397326872 0.0287781395 0.0213547982 0.0250474978 0.0231546629 0.0215805341 0.0158614013 0.0321737677 0.0188433882 0.0359748043 0.015868837 0.0230029859 0.0358589403 0.0246788245 0.0200167298 0.0341735072 0.0310240258 0.0384026133 0.0221543945 0.0297967643 0.0380295217 0.0262975041 0.027402306 0.0237040278 0.0167435091 0.0283355098 0.0227809809 0.0226217378 0.0214611385 0.0168901682 0 0 0]
   [0.0203569084 0.0202379096 0.0409209207 0.0257273261 0.0200277101 0.02360099 0.0264072698 0.0200900026 0.0361884572 0.0365515202 0.0270371027 0.0376429632 0.0207574889 0.0170554779 0.0409988202 0.0181871466 0.0388851 0.0264839884 0.0237956103 0.0406130888 0.0396645553 0.0173715763 0.0284624919 0.0238742437 0.0170340445 0.0343571082 0.0346517526 0.0221917257 0.0221114941 0.0170767158 0.021153437 0.0190622415 0.01924669 0.0355142057 0.0191728622 0.0189975556 0.0203295015 0.0281619895 0 0 0]
   [0.0172464699 0.0238810908 0.0452022851 0.0321676 0.0219158959 0.0201198049 0.0247152764 0.0192694869 0.0230441708 0.0281988326 0.0294426121 0.0238487031 0.0234154351 0.017151067 0.0178340971 0.0237977356 0.0253182668 0.0347594023 0.0272137877 0.0222356338 0.0183157101 0.035556633 0.0243791379 0.0334474593 0.0254397243 0.0182788018 0.0307610668 0.0220210329 0.0245105885 0.0288971271 0.0430335067 0.0316768847 0.018814357 0.0265962575 0.0229252093 0.0343145803 0.0377692692 0.0224849768 0 0 0]
   [0.0253064651 0.0313218534 0.033406321 0.0159596372 0.0429093391 0.0177279245 0.0177105609 0.0427780785 0.0366709344 0.0187023617 0.0200291164 0.0426750034 0.0180976484 0.0283433385 0.0352748074 0.0247887354 0.0190220363 0.0249243099 0.0265295617 0.0189595167 0.0229261108 0.022074528 0.038958665 0.0209504515 0.0214272235 0.0274249483 0.0182040725 0.02467921 0.0256848149 0.0255820155 0.0309444405 0.0211745147 0.0407009684 0.0361142 0.0301920213 0.0176121555 0.016479969 0.0177321862 0 0 0]
   [0.0224856287 0.0222625956 0.0171717051 0.0181706902 0.0237432346 0.0179174803 0.0268516615 0.025814252 0.0194638297 0.016989775 0.0412051827 0.036226768 0.0423628576 0.0220908 0.0287618414 0.0172643233 0.0417300053 0.0374272205 0.028292641 0.037736237 0.0198921822 0.0193145294 0.0261671506 0.0358173 0.0251102671 0.026146125 0.0172216855 0.033736445 0.0241549332 0.0288541 0.0322636105 0.016823547 0.0285156667 0.0254031494 0.0182640068 0.0181389935 0.0309052803 0.0293022841 0 0 0]
   [0.022427367 0.0374653488 0.0196384 0.0265451018 0.0276004765 0.0162689835 0.0287725441 0.0169721302 0.0299118031 0.0409166403 0.0215467624 0.0255800616 0.0239053536 0.034781117 0.0414382033 0.0355782211 0.0317863449 0.0247154292 0.0250837579 0.0190194622 0.0392913371 0.021804817 0.0367809609 0.0267335977 0.0236110911 0.0168253295 0.0179691296 0.0291865859 0.03022472 0.0185204633 0.0354454927 0.0161904097 0.033619795 0.0153858364 0.0286371242 0.0165817831 0.0263932142 0.0168448333 0 0 0]
   [0.0181796458 0.024465289 0.0396128036 0.0295651164 0.0384036452 0.0172350984 0.0163211878 0.0318988338 0.0349477939 0.0171665121 0.0162399095 0.0215943288 0.0398661941 0.0185327623 0.0200679079 0.0321712382 0.0183348656 0.0181851611 0.0204315223 0.0191337 0.0317707285 0.0340508446 0.0322120339 0.0396556742 0.0324791037 0.0179526974 0.0344070606 0.0240735188 0.0188220385 0.0223930478 0.0401388668 0.0238826536 0.0181410778 0.0342072137 0.0264685787 0.024748629 0.0320448279 0.0201979205 0 0 0]
   [0.0308623556 0.0203879904 0.0269630142 0.031531658 0.0328926221 0.0279915202 0.0188028719 0.0161957834 0.0285725556 0.0222554449 0.0171211585 0.0364643112 0.0227640513 0.0216150917 0.0284455735 0.0282624811 0.0301003885 0.025099121 0.040318843 0.0174938366 0.0382392481 0.0245383121 0.0388127677 0.0253477748 0.0280764196 0.0183907766 0.0379295573 0.0344737172 0.0160283931 0.0322999768 0.0200272948 0.0193139743 0.0185126495 0.0202306416 0.0213539414 0.0286489148 0.0202015564 0.033433456 0 0 0]
   [0.0331565477 0.0241718609 0.0249268822 0.0328256711 0.0194985792 0.0168171842 0.0335710198 0.0225671232 0.016178824 0.0204243287 0.0314347707 0.029324742 0.0217621494 0.0157934055 0.0161716882 0.0211320668 0.0285475105 0.0186289754 0.0411981642 0.0193997 0.0167949423 0.031523712 0.0281967726 0.0301973242 0.0259796865 0.0214837044 0.0302314553 0.0312363282 0.0421424173 0.0254626498 0.0317757167 0.0248872656 0.0349730588 0.0212617125 0.017066462 0.0366507508 0.0423199572 0.0202848222 0 0 0]
   [0.024636656 0.0305111017 0.0309279375 0.0174950343 0.0219344273 0.0175120216 0.0187825654 0.0445933305 0.0233126637 0.0195515 0.0374753214 0.0342694856 0.0317958966 0.0245610885 0.0168541316 0.0356226861 0.0340675414 0.0172426961 0.0409867913 0.0176963601 0.0166411065 0.0284473859 0.0212387238 0.0214226283 0.0329119675 0.0358235464 0.0274686832 0.0275710505 0.0233983956 0.0166625865 0.0306043811 0.0179129932 0.0296779107 0.0290353727 0.0337010063 0.0193498563 0.0308634602 0.017439751 0 0 0]
   [0.0237777419 0.0321823768 0.0251336247 0.0258965474 0.0178235807 0.0151803764 0.0170391183 0.0365880877 0.0178894699 0.0378710181 0.032340534 0.0245511085 0.0377240553 0.039467413 0.0206220988 0.0398923866 0.0366741568 0.0235339254 0.0187040605 0.0199115891 0.02235833 0.0293609854 0.0199224874 0.0193132218 0.0299574863 0.0273504592 0.0196892861 0.0158207081 0.0256749652 0.0208222829 0.0207090061 0.0321469717 0.0397117138 0.0261122845 0.0359140523 0.0261636283 0.0218878109 0.0242810939 0 0 0]
   [0.0311155319 0.027970504 0.0201677512 0.0209976304 0.0241595246 0.0156354494 0.0356835872 0.0316746272 0.0297952425 0.0188351236 0.0230022781 0.0356175974 0.0170388725 0.0381916575 0.0259027015 0.0221014 0.0335526206 0.0332428701 0.0207416546 0.0154859889 0.0397905558 0.0265910234 0.0168896504 0.024482986 0.0155267883 0.0347111151 0.0275752917 0.0329435505 0.0188843627 0.0213835016 0.0369169749 0.0153189115 0.0349454433 0.0302115176 0.0291591212 0.0233641546 0.0287887659 0.0216036718 0 0 0]
   [0.015891647 0.0236326437 0.014889908 0.0148466267 0.0344630592 0.0297353882 0.0375203304 0.0345091 0.0284016207 0.0225087162 0.018352326 0.0249210671 0.0213457849 0.0244785938 0.0310106408 0.0173223801 0.018434586 0.02897745 0.0356837325 0.0214404613 0.0201842766 0.0363090858 0.0272934195 0.028347984 0.0379955061 0.0280525312 0.0315139331 0.0393913686 0.015402656 0.0232737623 0.0247826762 0.0147125125 0.0316997021 0.0240720864 0.0204934888 0.031057613 0.0315644518 0.035486903 0 0 0]
   [0.0381524526 0.0276029222 0.0392218791 0.0192205384 0.0243793521 0.0368533842 0.0169485714 0.0319770649 0.018249657 0.0294705182 0.0238194093 0.0263103452 0.0201512892 0.0331283435 0.0299924165 0.0297887418 0.0226034187 0.026889991 0.0178322513 0.0367587507 0.0209293235 0.0287939757 0.0264342986 0.0156458132 0.0272286572 0.0396679454 0.0340799242 0.0203219689 0.037344154 0.0171149429 0.0231878515 0.0377809778 0.0155703798 0.020962147 0.019597644 0.0233135335 0.016199369 0.0264758132 0 0 0]
   [0.0219782535 0.0321364366 0.0377465114 0.0184262041 0.0336266384 0.0279543679 0.0320858508 0.0293029 0.0178605895 0.0328136757 0.0297106728 0.0166116394 0.0251480341 0.0193895176 0.0340125263 0.020390885 0.0161680616 0.0236695595 0.0212828703 0.0319892503 0.0360204503 0.0229470488 0.0276192259 0.0315770768 0.0192744844 0.0242111199 0.0197025612 0.0321862064 0.0158447176 0.0214673262 0.0350565352 0.0285849553 0.0332594328 0.0299345143 0.0351964533 0.0160070844 0.0154622151 0.0333441235 0 0 0]
   [0.022896491 0.0190402195 0.0323250033 0.0308286175 0.0196797345 0.0179283954 0.0288757533 0.0174861196 0.0150554106 0.0306758359 0.0369188786 0.0219878163 0.0226834472 0.0359698832 0.0275570415 0.0373856835 0.0330549628 0.0185158569 0.0367516764 0.0210788902 0.0226891153 0.0224853847 0.0195577685 0.0304622408 0.0256696492 0.01934563 0.0288953111 0.030288348 0.0172704589 0.0373473652 0.032434117 0.0240163393 0.0204455331 0.0256361067 0.0374320112 0.0267261211 0.0311727561 0.0214301161 0 0 0]
   [0.0214802027 0.0361392833 0.0345565714 0.0154289557 0.0183498431 0.0187975764 0.0227675885 0.0287790336 0.0153980181 0.0396494195 0.0324217342 0.01867323 0.0225376878 0.0291172359 0.026702622 0.0315727629 0.0306447893 0.0292654615 0.0247748308 0.0266538933 0.0254405122 0.0178736113 0.0390264876 0.0362865515 0.0315622501 0.0266958605 0.021810757 0.0198979378 0.0225001238 0.0174869299 0.0168880317 0.0396113694 0.0194767807 0.0215763729 0.0268484019 0.033017233 0.0300990306 0.030190954 0 0 0]
   [0.035555616 0.020898439 0.0346833169 0.0164048132 0.0373804867 0.0350298695 0.016149126 0.0216027685 0.0294966269 0.018600069 0.0174737535 0.0331039317 0.0381581597 0.0236329306 0.0356613696 0.0319446027 0.0342647023 0.0162093025 0.0307649709 0.0360272564 0.0166490935 0.0258699767 0.0209195409 0.0189663358 0.0312231593 0.0173860379 0.0379175656 0.0231836494 0.043597579 0.0285471492 0.0223369896 0.0167560428 0.0339823067 0.0215339214 0.0200980175 0.0169011708 0.0232008863 0.0178884789 0 0 0]
   [0.026677154 0.0280740149 0.0324797705 0.0197452 0.0347579978 0.0280039348 0.0160835702 0.0180057157 0.0352573171 0.0380125754 0.0174611639 0.0299454574 0.0167417079 0.0208177734 0.0383648612 0.01683148 0.0222592838 0.038335871 0.0252669677 0.0283701587 0.0215891 0.0352515727 0.0274594743 0.0156948194 0.0218209606 0.0312027875 0.0326847471 0.0208929274 0.0239210948 0.0196468644 0.0332517289 0.0258697942 0.0357751846 0.0189667214 0.0214644596 0.0299994759 0.0200283062 0.0329880267 0 0 0]
   [0.0312634632 0.0286795124 0.031942416 0.0183181632 0.0164547507 0.0285141356 0.0185009539 0.016963603 0.0244879257 0.0349925123 0.0344360843 0.0279212799 0.0357601978 0.0230121501 0.0251421183 0.0291601364 0.0168414488 0.0189329535 0.0251566321 0.0212030038 0.0171670076 0.0238487106 0.0168332402 0.0200099107 0.0419979729 0.0393406302 0.0407507867 0.0161735099 0.0185218267 0.0386183262 0.0202525333 0.0382741317 0.0234250575 0.0415345021 0.0293142274 0.020019995 0.0208617169 0.02537244 0 0 0]
   [0.0220020451 0.028530458 0.0347004607 0.0382093675 0.0176272281 0.0263871811 0.0211178195 0.0221421029 0.0188840069 0.0207717903 0.0336857215 0.0283431429 0.0199779775 0.0169388857 0.0407554582 0.0177552458 0.0176284071 0.0385914855 0.0352395624 0.0299915336 0.0327539518 0.0323659293 0.0190440845 0.0284701176 0.0228590816 0.0299162325 0.0204065684 0.0193270836 0.0170331467 0.0229880363 0.0168865882 0.0414892733 0.0186669379 0.0312273297 0.0289773177 0.0370260291 0.0269849189 0.0242974646 0 0 0]
   [0.0272296574 0.0203913953 0.0188472942 0.024391422 0.0398513153 0.0244898032 0.0211939 0.0307530183 0.0416680351 0.024027409 0.0326640047 0.0174605902 0.0300592165 0.0323088802 0.0191120468 0.0189948156 0.0252308268 0.022077987 0.0181962363 0.0223475397 0.0341081545 0.0215060674 0.022164626 0.0230255816 0.0190360863 0.0388039388 0.0343499035 0.017272057 0.0264867079 0.0243558325 0.0201222599 0.0443436429 0.0263289139 0.0398934446 0.0177972317 0.0227216054 0.0255490877 0.0308394823 0 0 0]
   [0.0261918511 0.019777583 0.0338145271 0.0306488946 0.0193304308 0.0177325085 0.0255056154 0.0237156264 0.0272099357 0.0288818553 0.0341515541 0.0343821235 0.0335062481 0.0343698449 0.0202757902 0.0290218405 0.0181073919 0.0266789775 0.0264796317 0.0268121734 0.0242367946 0.0292417947 0.0246252324 0.0151067106 0.0325117782 0.0364741385 0.0184337609 0.0328550339 0.0397327431 0.0212844573 0.0189934485 0.0231265035 0.019193707 0.0320678577 0.0187284667 0.0199066345 0.0242356602 0.0326508731 0 0 0]
   [0.0171695352 0.0218483508 0.0170912612 0.0179644804 0.026099598 0.0290208105 0.0225689877 0.0381765142 0.0186151937 0.0321737863 0.0219170656 0.0325209536 0.0169712286 0.01639135 0.037729241 0.0258539375 0.0284458604 0.0262607671 0.0253207032 0.0192737449 0.0369464606 0.0215581153 0.0377692692 0.0170277376 0.016416084 0.0220344644 0.0435940325 0.0204085167 0.0384790264 0.0355312079 0.0291397311 0.0188464336 0.0187501162 0.0300997049 0.0202945601 0.0391607583 0.03550696 0.0270234495 0 0 0]
   [0.0330668502 0.0347369276 0.0304691 0.0365534797 0.0303976405 0.0370763093 0.0278344378 0.0155782178 0.0229400825 0.028426962 0.0178498011 0.0290516093 0.0323349535 0.0199921336 0.0209581796 0.0383258238 0.0329571478 0.01873005 0.0175117049 0.0187060442 0.0238591414 0.0258200131 0.0304945596 0.0363986231 0.0187615026 0.0178785846 0.0188429765 0.0225505512 0.0386897102 0.0204459913 0.0328123495 0.0190426409 0.0310651846 0.0259309802 0.0380121209 0.018727148 0.015851805 0.0213186722 0 0 0]
   [0.0157173183 0.0208879448 0.0318075716 0.0328804515 0.0378330834 0.0343388021 0.0148078324 0.0199322831 0.0301532317 0.020068435 0.0232185293 0.0229923613 0.0176378377 0.0144246062 0.0224704389 0.0192468055 0.0371409245 0.0374985449 0.0370572172 0.0215041339 0.0346682891 0.0326919667 0.0297289453 0.0189849027 0.0342974812 0.0277172457 0.0222905446 0.0361077935 0.0316756852 0.0281100534 0.0181765798 0.0248469189 0.0228276663 0.0373086408 0.021443082 0.0330206119 0.0184493922 0.0160358697 0 0 0]
   [0.0305412468 0.0217122771 0.0200653151 0.0181163922 0.0412429422 0.0312593654 0.0255940519 0.0234874245 0.0312994979 0.0178563967 0.0195901301 0.0346218646 0.0377255976 0.0181031488 0.0213961694 0.0175368488 0.0178524237 0.0328524597 0.024244668 0.0198463928 0.0272433609 0.0188252348 0.0417622402 0.0244032647 0.0235807225 0.0394172631 0.0312757604 0.0332652926 0.0207948517 0.0278396048 0.0238405857 0.0283932947 0.0300090238 0.0370427854 0.0213537887 0.0259441677 0.0181653816 0.0218988266 0 0 0]
   [0.0249319971 0.032542035 0.024363894 0.0191380978 0.0229901578 0.034871541 0.0229848847 0.0207143649 0.0365864597 0.0312866084 0.0333461426 0.0311247986 0.0275723487 0.0282388031 0.0223937761 0.0285747349 0.0176536925 0.0304946452 0.033015471 0.0251511149 0.0279390104 0.0271968693 0.0279935505 0.0214250479 0.0180711988 0.0331477933 0.0226952955 0.0390472189 0.0311702918 0.0202629212 0.0303471815 0.0178525634 0.019894639 0.0179007016 0.0220646858 0.0233696662 0.0254729073 0.0261729788 0 0 0]
   [0.0398550406 0.0265165865 0.0226703901 0.0345558 0.0259929691 0.0226982981 0.0237211622 0.0163823478 0.0306826644 0.0160104949 0.0291556306 0.0284628905 0.0341216139 0.0349901766 0.0412263162 0.0171245206 0.0327472091 0.0210624132 0.0388602763 0.023341 0.0368731767 0.0183856133 0.019508699 0.0257859882 0.0299583469 0.0167404152 0.0159259457 0.0292425696 0.0181041509 0.0167141333 0.0406019203 0.0298147704 0.020398546 0.030227527 0.0237482563 0.0238484554 0.026476007 0.0174677279 0 0 0]
   [0.0200454798 0.0316760577 0.0240555611 0.0342460535 0.0291427132 0.0234357025 0.0331403688 0.0380101576 0.018824093 0.0229505 0.0221246071 0.0223759916 0.017614251 0.0296570398 0.029724773 0.0359692946 0.0376550034 0.0269606952 0.0386952534 0.0194042586 0.0286758319 0.0154489167 0.0221828632 0.01745506 0.0247335974 0.0213074423 0.0396038629 0.022921104 0.0161912963 0.0216404106 0.0153133078 0.0274456926 0.0336314365 0.0177568272 0.0359385 0.0218929555 0.0240766034 0.0380763672 0 0 0]
   [0.0277150571 0.0345643 0.0194096249 0.0278713759 0.0269760378 0.017710885 0.0167611726 0.0207992252 0.0189552139 0.0179242697 0.0277699437 0.0234855227 0.0188766811 0.036901962 0.0234827474 0.0289305169 0.0185026769 0.0289537199 0.0324750394 0.0189423934 0.040993385 0.0206410307 0.0390669629 0.02675483 0.0194762051 0.0318329073 0.035725154 0.025967462 0.0391750857 0.0266945641 0.0311256256 0.0310074557 0.0213183369 0.0195249692 0.0205907337 0.0414976142 0.0222216379 0.019377606 0 0 0]
   [0.0183447413 0.02888483 0.0371931642 0.0233950596 0.0230933372 0.0250537451 0.0375914648 0.0369112752 0.0240651 0.0175654683 0.0249202717 0.0249371473 0.034809012 0.0375838578 0.026549 0.0275152195 0.0337773785 0.0337231569 0.0172006059 0.0171502568 0.0217850711 0.0170235578 0.0217893738 0.0222203545 0.0226501469 0.0196123105 0.0345453918 0.0373846479 0.0284840818 0.0290812403 0.0419355929 0.024194859 0.0308530256 0.0156625118 0.0298491884 0.0158477314 0.019752508 0.0170643553 0 0 0]
   [0.0393277183 0.0280879177 0.0379942842 0.0218401551 0.032719288 0.0323253237 0.0338132493 0.0168410819 0.0217537154 0.0236403923 0.017514972 0.0204911027 0.0170089938 0.0323291384 0.0315440781 0.016433388 0.0197270345 0.036714986 0.0395493619 0.0338331535 0.0232996698 0.0163951237 0.0268492214 0.0331075229 0.0236533191 0.030467106 0.0303153135 0.0171739981 0.0249418914 0.0423140228 0.0244071335 0.0174021106 0.0190197937 0.0193971414 0.021486301 0.0179101508 0.0281784683 0.0301923472 0 0 0]
   [0.0385992676 0.034833502 0.0381102785 0.0157577768 0.0338450819 0.0153367985 0.0320466384 0.0219288673 0.0365929157 0.0188097078 0.027333688 0.0275291707 0.0182177164 0.0250503905 0.0181590412 0.0225692037 0.02301508 0.0344023816 0.0168749355 0.0176169816 0.0226071086 0.0161089897 0.0184779987 0.0380967297 0.0314058363 0.0225567557 0.0235526171 0.0377099 0.0254601128 0.0333713777 0.0166574735 0.0299588554 0.0380121134 0.0188734531 0.0350026414 0.0250842478 0.0199814457 0.0304528661 0 0 0]]

  [[0.0205017868 0.0319312885 0.0315385833 0.0244478602 0.0158154406 0.0388138443 0.0276507549 0.0179643594 0.0321318284 0.0292302463 0.023456255 0.0163361877 0.025421964 0.016885303 0.0203448534 0.030510623 0.0383130275 0.0397598594 0.0359213464 0.0175359752 0.0205570478 0.0254540499 0.0390634723 0.0165326558 0.0176492576 0.017181538 0.0172966402 0.0301277023 0.0180798676 0.0275770426 0.0170942973 0.0339681618 0.0274330843 0.0382283814 0.0326094814 0.029153889 0.0394294523 0.0180525519 0 0 0]
   [0.0224296656 0.0236444417 0.0299812183 0.0312881246 0.0181064904 0.0175752211 0.0224564895 0.0369461142 0.0234595593 0.0241933037 0.0251746904 0.0164616592 0.0297642965 0.0388591 0.0351537 0.0260737669 0.0189234745 0.0299583133 0.0382707939 0.0212169476 0.0328647047 0.0246884748 0.0375714786 0.0255686454 0.0215542037 0.0263694208 0.0206942763 0.0188483857 0.0186859164 0.0197364818 0.0413972251 0.0372835808 0.0214557648 0.0180456489 0.0200759787 0.0251187738 0.0258195437 0.0342841707 0 0 0]
   [0.037340764 0.0263549369 0.0374109745 0.0163079519 0.0406586528 0.0193873 0.0354996845 0.0355513208 0.0156069119 0.0230679959 0.0174876675 0.0165922567 0.0170165189 0.0245872047 0.0181089658 0.0231692865 0.0406610854 0.0299602728 0.0158385728 0.0310962275 0.0361581184 0.0194767248 0.0191410333 0.0315241851 0.0160814691 0.028290594 0.0330972821 0.0349894837 0.0233246405 0.0170267541 0.0177797824 0.0336277522 0.027815111 0.0219281558 0.0200780649 0.0383074172 0.0385449491 0.0211038925 0 0 0]
   [0.0181825776 0.0171752796 0.0254387576 0.0192478877 0.0282115638 0.0219627768 0.0231271647 0.0306180604 0.016117448 0.0262956023 0.0318753533 0.0173555091 0.0342890546 0.0235725231 0.0407303 0.0270528421 0.031565357 0.0157870185 0.0408280492 0.0370346531 0.0178777929 0.025564516 0.0250025541 0.0284456108 0.0213883594 0.0190489944 0.0237946175 0.0221533719 0.0308592804 0.0377747491 0.0219681952 0.0157108661 0.0396838523 0.0247000717 0.0351907052 0.031386435 0.0273680948 0.0256141108 0 0 0]
   [0.0386732 0.020777585 0.0262758955 0.0154119395 0.0187115055 0.023109518 0.0364539549 0.0179500859 0.0279958025 0.0357538089 0.0281906333 0.0297823511 0.030360857 0.0156938415 0.0404748134 0.0348075777 0.0192879755 0.0272479523 0.0174844898 0.0160651505 0.028828403 0.0205089878 0.0271045696 0.0170634817 0.0230640825 0.0406482145 0.015988633 0.0278096031 0.0404242724 0.0347683541 0.0172862373 0.0355018601 0.0220166817 0.0243424308 0.0350094922 0.0327680968 0.0155788604 0.020778805 0 0 0]
   [0.0163630173 0.035625685 0.0252626631 0.0380055644 0.0206555221 0.0238146968 0.0180414878 0.023630118 0.0246709585 0.018571239 0.0170918275 0.0188712794 0.0345593728 0.0346185081 0.016309984 0.0230215043 0.0200732853 0.0171360299 0.0179656297 0.01760385 0.0201864652 0.041264791 0.0267814174 0.0341421291 0.0265881382 0.0381904468 0.0381101854 0.0319570266 0.0378865339 0.0218913592 0.0386978425 0.0339336544 0.0228017122 0.0157690402 0.0393603295 0.0222403537 0.0222017951 0.0261044893 0 0 0]
   [0.0418648869 0.0179064255 0.024839079 0.0176398568 0.0265101958 0.0288801845 0.0167927258 0.0363944396 0.0206546057 0.0165954418 0.0192046128 0.0392275266 0.0375165232 0.0239586066 0.0189564545 0.0237325635 0.0218163636 0.0447009206 0.0349636562 0.0184228774 0.0179496948 0.0299043413 0.0195341576 0.0300029051 0.0244826432 0.0399383 0.0278652851 0.0201149229 0.0185789391 0.0193949919 0.0404264256 0.0248443522 0.0281848926 0.0220640805 0.0266348291 0.0200058669 0.0240807235 0.0354146212 0 0 0]
   [0.0374612771 0.0188809 0.01784 0.0182752199 0.0268571842 0.0217380654 0.0179486256 0.0378772542 0.0184360109 0.0361108 0.0241784379 0.0326600671 0.033095587 0.0297424477 0.021886 0.0323254056 0.0195845757 0.0231420416 0.0154528441 0.0303761493 0.0227956 0.019977184 0.0390118845 0.015483967 0.0156668127 0.0185052063 0.0223179832 0.0222564265 0.0319034345 0.0277812649 0.0294680651 0.0329955071 0.0326985344 0.0295783728 0.0267636739 0.0363643914 0.0221540481 0.0404086858 0 0 0]
   [0.0334829874 0.0152835175 0.031092165 0.0268535931 0.0355577357 0.0172232334 0.0239202641 0.0362099931 0.0178327169 0.0233606137 0.0271559358 0.0323251821 0.0208203215 0.0164786614 0.0212248527 0.0239256527 0.0361148156 0.0329474472 0.036093533 0.0284315031 0.0242597703 0.0369261801 0.0181294046 0.0142710134 0.0354419872 0.0341809504 0.0212957039 0.0339337662 0.0154022481 0.01434437 0.0195795726 0.0319846347 0.0157412719 0.0290703308 0.0267632585 0.0342588536 0.0319407843 0.0261411853 0 0 0]
   [0.0313975587 0.0219114125 0.015905777 0.0359022953 0.0207288098 0.0169566 0.0189029071 0.0273293946 0.0383671857 0.0189359076 0.0351130404 0.0196130686 0.0281436685 0.016270062 0.0407174341 0.0157203637 0.0242489353 0.0354606323 0.0397110321 0.025014909 0.0361628905 0.033706069 0.0215138942 0.0217366405 0.0178998262 0.0403884351 0.0258773789 0.0396627933 0.0166990012 0.0234663319 0.0224139579 0.0207626838 0.0250629708 0.0272069294 0.0409513153 0.0192853827 0.0236489307 0.0172035061 0 0 0]
   [0.0280889887 0.0340617076 0.0165894777 0.0284476038 0.0262719542 0.028629439 0.0399533659 0.0157711543 0.022317728 0.0263784863 0.0174881462 0.030780416 0.0329692066 0.0180931 0.0320864618 0.0189755205 0.0354786217 0.0296587721 0.0259074792 0.0207024906 0.020775618 0.0406806692 0.0258539356 0.0165121686 0.0164724756 0.0191704929 0.0359446183 0.0396689139 0.0155090699 0.0346145667 0.0245735776 0.0160161816 0.0302197542 0.0296664629 0.0200408399 0.0343945548 0.0209348965 0.0303010922 0 0 0]
   [0.0205765907 0.0312745199 0.0246309694 0.0160757434 0.0183281489 0.0210020468 0.018549826 0.0238797721 0.0348173045 0.0303964615 0.042206265 0.018399572 0.0377073884 0.0290479101 0.0403094143 0.0250295643 0.0425206795 0.0173212346 0.0338243805 0.0170870032 0.0251067691 0.0183159094 0.0212471895 0.042245023 0.0183330346 0.0165933184 0.0227307882 0.0282741468 0.0392406546 0.0180111378 0.0286411885 0.0379952677 0.0174469259 0.0176121332 0.0161696207 0.0376173034 0.0284832399 0.0229515471 0 0 0]
   [0.0266597327 0.0400900729 0.0173927862 0.019886991 0.0279255453 0.0222407207 0.0329912938 0.0422822535 0.0189122334 0.0208418649 0.029295478 0.0256494302 0.0196007565 0.0201951358 0.0187902059 0.0208993237 0.0272563063 0.0177756045 0.0184116475 0.0240122396 0.0265292618 0.0308503527 0.0163826048 0.0216827113 0.0316651613 0.0264970697 0.02324396 0.0338255167 0.0413813032 0.0171607025 0.031758368 0.0338611901 0.0372889787 0.0344111249 0.0186880864 0.0268768277 0.0300961193 0.0266910307 0 0 0]
   [0.0228085984 0.0379293151 0.0167705026 0.0315336883 0.0243894029 0.039597448 0.0219229404 0.0263730399 0.0162684247 0.0251527466 0.0342373773 0.0150406659 0.0363653563 0.0224841591 0.038987387 0.0293502342 0.0211516619 0.0192595329 0.0323693492 0.0161002856 0.0327057801 0.0355964489 0.0362730548 0.0245113112 0.0156855397 0.0205787346 0.0264576972 0.0304700378 0.0354878753 0.0197805408 0.0399276353 0.0358095579 0.0169049501 0.0174655933 0.0294132903 0.017683709 0.0221884772 0.0149676595 0 0 0]
   [0.0226970743 0.0321058 0.0230175424 0.0296030231 0.0194166787 0.0182827208 0.0159909856 0.0327899903 0.0305950623 0.0239578318 0.0369919538 0.0185535438 0.0181793198 0.0256838 0.019817153 0.0396371298 0.0203179177 0.0189810395 0.0293360744 0.0275451373 0.0195583086 0.0259209946 0.0322171859 0.0369407162 0.0375437886 0.0163452 0.0305894725 0.0269295052 0.0204273816 0.0361344665 0.0239241682 0.0369967259 0.0163568594 0.0230123829 0.022120975 0.0174928643 0.0409082249 0.0330809094 0 0 0]
   [0.0175165981 0.0171552207 0.016566759 0.0291288886 0.037504632 0.0344938487 0.0168626476 0.0195299685 0.0350910835 0.0323331393 0.04272696 0.0249418207 0.0204723552 0.0379488282 0.0169362891 0.018952189 0.0293678883 0.0297728274 0.0411245525 0.0395964198 0.0189128071 0.0352265425 0.0174566638 0.0275463071 0.0161557645 0.0331415795 0.0246142875 0.0304392502 0.0206354242 0.0230711717 0.0175280757 0.0218846668 0.0293701645 0.0178988203 0.023681879 0.0234803706 0.019168 0.041765254 0 0 0]
   [0.0359397866 0.0269538276 0.0375302322 0.025669504 0.0357641689 0.0149994623 0.0226700976 0.0161266308 0.0353846736 0.0249180011 0.0165323932 0.0351903476 0.0379569866 0.0147070782 0.0281927157 0.0274947975 0.0229291432 0.0352374911 0.019076325 0.0357994102 0.0159700885 0.0363348871 0.0142906215 0.0349065587 0.0378470533 0.0285635665 0.0272316169 0.0143405264 0.0350010656 0.0147056263 0.0152754923 0.0286224037 0.0169607103 0.0326309577 0.0210409276 0.0203324314 0.0341575146 0.0227149203 0 0 0]
   [0.029107064 0.0166798774 0.0314258039 0.0145896925 0.0160875116 0.0352631472 0.0242936555 0.0368960127 0.0249308813 0.0310852267 0.0372413 0.0187591966 0.0157599952 0.0219972581 0.0326308534 0.0300041921 0.028627662 0.0312852338 0.0186726227 0.0330110975 0.0348269828 0.0262437221 0.0254316628 0.0333903767 0.0145611977 0.0156098027 0.0160662159 0.0338774398 0.0192162097 0.0386311673 0.0181902107 0.0310468245 0.0207543112 0.0147632686 0.0344770886 0.0326665565 0.0308793243 0.0310193244 0 0 0]
   [0.0311265923 0.032043986 0.0287881531 0.0274199955 0.025363503 0.0153841944 0.0244181082 0.0166983381 0.0263728313 0.0239257775 0.035716027 0.0352814905 0.0289481767 0.0166307166 0.0286686942 0.0179977659 0.0177231506 0.027840646 0.0276033673 0.0363540351 0.0252058823 0.0265944619 0.0390342 0.0153481429 0.0200854968 0.0206679218 0.031927973 0.0209967699 0.0243400205 0.0187134016 0.0318884812 0.02745465 0.0342236534 0.0193412434 0.0316799767 0.028394768 0.0353990197 0.0243984032 0 0 0]
   [0.0323269144 0.0161396153 0.0275590159 0.0179652348 0.0218128022 0.0352843925 0.017926814 0.0279334262 0.0254529435 0.0242195353 0.0412394814 0.0202817544 0.0241083428 0.0374530256 0.0267343186 0.0333104 0.0348939821 0.0355686694 0.0190473646 0.0287460648 0.0234529395 0.0156865977 0.0256077629 0.037050195 0.0234286953 0.0356000625 0.0195649192 0.0160865225 0.0178415403 0.0168247912 0.0349636786 0.0181823596 0.0413144343 0.0297893304 0.0185227655 0.0157765821 0.0278028063 0.0344998576 0 0 0]
   [0.033697281 0.0220578276 0.0277835336 0.0348055437 0.0194014907 0.0300066564 0.0249208473 0.0389271 0.0340029597 0.0338716805 0.0354598612 0.0353786238 0.017172521 0.0216702968 0.0221799295 0.0295366794 0.0210205428 0.0294436552 0.0306808893 0.0216744319 0.0272862725 0.0324996375 0.0229181908 0.0172365177 0.031305071 0.0196311679 0.0194568802 0.0179863181 0.0201118477 0.0180846136 0.0369361155 0.0202202089 0.0265247934 0.015905099 0.0286098607 0.0234531146 0.0244941264 0.0336478315 0 0 0]
   [0.0389814861 0.0166097954 0.0240304396 0.0250147209 0.0204810947 0.0391864665 0.0180015545 0.0229340419 0.0412448719 0.0315109231 0.0186210126 0.0327666029 0.0262082815 0.0285414066 0.0293144211 0.0237717293 0.0185752232 0.0358846448 0.0186390392 0.0224790443 0.028451534 0.0372289121 0.0184923355 0.0185663495 0.0154829947 0.0339271314 0.0244577657 0.0161138531 0.0166016016 0.0413744785 0.0395391472 0.0205181148 0.0237052683 0.0164090749 0.0384015925 0.0234613027 0.0230299514 0.0314418 0 0 0]
   [0.0287191588 0.0170511808 0.0200677 0.0347109549 0.0207530726 0.0276348013 0.0310843028 0.03328472 0.0297506582 0.0191266444 0.0267509073 0.0405327678 0.0313434303 0.027382737 0.0267930571 0.0312979147 0.0158584379 0.0324896127 0.0170025956 0.0253772791 0.0181169305 0.036043182 0.0215581376 0.0270263292 0.0334944092 0.0372383632 0.0312100854 0.0255183931 0.0153575921 0.0280454811 0.0237179901 0.0178077519 0.0158913173 0.0199520588 0.0307592656 0.035202954 0.0301298965 0.0159179028 0 0 0]
   [0.0192323 0.0273341462 0.0210417025 0.0275014341 0.0334697589 0.0290148817 0.0259744376 0.016665332 0.0225992892 0.0211201217 0.0419770256 0.021207355 0.0206189808 0.0207450222 0.0263809767 0.0177860055 0.0174317509 0.0284097586 0.0366608873 0.0253145136 0.0247439649 0.0289666466 0.0198593345 0.0212654136 0.0325436294 0.0224186331 0.0408069 0.0318493359 0.0346396975 0.0243792553 0.0188765042 0.022614982 0.0196337588 0.032397639 0.0229150876 0.0262752026 0.0353102945 0.0400181487 0 0 0]
   [0.0355761647 0.0321620442 0.0206038654 0.0225352421 0.0358679444 0.0284488797 0.0235360507 0.0186846368 0.0354541838 0.0292460229 0.0212899819 0.0246215314 0.0388181061 0.021304423 0.0414990112 0.017375892 0.0246478133 0.0211605188 0.016684683 0.0272925738 0.0237577595 0.0386232324 0.0233423375 0.0319835208 0.0318765976 0.0161803719 0.0241880324 0.0415175073 0.0214422531 0.0170247536 0.0238104574 0.02936792 0.0375556275 0.0170185622 0.0237878095 0.0167287178 0.0184182581 0.0265667047 0 0 0]
   [0.0307410769 0.0192669556 0.023582276 0.0216264371 0.0292069521 0.015839681 0.0167266596 0.0259000156 0.0289011318 0.0286640432 0.0294420328 0.0153139075 0.0160436854 0.0362450033 0.0353415906 0.0209987 0.0211426467 0.0195596218 0.0395511799 0.0277304817 0.0284605753 0.0260722805 0.0369474627 0.0282683671 0.0281886868 0.0161730424 0.0317682847 0.0313117839 0.023979798 0.0263224225 0.0378069654 0.0164763611 0.0215851776 0.0273373723 0.034247756 0.0213765875 0.0382316262 0.0236214176 0 0 0]
   [0.0279118847 0.0198464543 0.0223743077 0.0193218365 0.0382363461 0.0287966896 0.0227001067 0.0170740336 0.0206664968 0.0232608244 0.0370782 0.0209466834 0.0193804167 0.0234150887 0.0168538895 0.0152422599 0.0273573771 0.0345954783 0.0307130422 0.0320744589 0.017470561 0.0238260329 0.0192551315 0.0284838974 0.0214470495 0.0314096026 0.0218681637 0.0227658078 0.0331992395 0.0360328145 0.0232864935 0.0330723785 0.0334943123 0.0263175573 0.0323248617 0.0361339971 0.0291842371 0.0325819813 0 0 0]
   [0.0252539124 0.0334488191 0.0309098959 0.0199816171 0.0370438322 0.0350012742 0.018653689 0.0241074618 0.0218862128 0.0345880315 0.021684844 0.0426519327 0.0372828804 0.0335752405 0.0260950048 0.0203475375 0.0334902816 0.0195752885 0.0167889874 0.0237451307 0.0264806133 0.0173888225 0.0221776646 0.0242108181 0.0198156182 0.022396123 0.0338866413 0.0292239916 0.0188800301 0.0237649456 0.0408063754 0.0266415011 0.0201026052 0.0222199373 0.0307950135 0.0225438755 0.0216881558 0.020865405 0 0 0]
   [0.0347075 0.0169469211 0.0332990773 0.0285538025 0.0216813795 0.0185486935 0.0407971814 0.042373009 0.0164266583 0.0210822839 0.0233421661 0.0328340307 0.0242071562 0.038525302 0.0239932854 0.0173975676 0.0203947052 0.0324093774 0.0161787756 0.0242483709 0.0250278469 0.0270414036 0.03304502 0.0200099312 0.0293947197 0.0252118185 0.0168473814 0.0332171358 0.0337045901 0.0264751706 0.020938186 0.0328660533 0.0167277362 0.0176121891 0.0354639851 0.0260364376 0.0159536637 0.0364795 0 0 0]
   [0.0380267501 0.025314264 0.0209693443 0.0292200483 0.0289827511 0.0237913709 0.0293036811 0.0181301739 0.0329577923 0.0219007917 0.0402030535 0.0206684936 0.0174902063 0.0194317251 0.0256404113 0.0262909923 0.0164207034 0.0336750224 0.0385804065 0.0180468634 0.0292763 0.0243395939 0.0180808213 0.0179441869 0.0338967592 0.0339297391 0.0166361164 0.0164003819 0.017482575 0.0364980549 0.039653711 0.0308097173 0.0350602381 0.0177351534 0.0323810652 0.0232276898 0.0256876703 0.0259154402 0 0 0]
   [0.0236066859 0.0183432959 0.0188175235 0.0199233722 0.030430194 0.0186328646 0.0207260512 0.0340753794 0.0292660277 0.0153894471 0.0365386829 0.0307227969 0.0390497297 0.0228932 0.0335247107 0.0379402898 0.0370854363 0.0364376977 0.0235451926 0.0292627495 0.0356106311 0.0319549404 0.0162118226 0.019998882 0.0376538 0.0255173687 0.0380061492 0.0181789957 0.0203140732 0.0233405922 0.0162728503 0.040799737 0.0154943867 0.0274621658 0.0207280833 0.0152766537 0.025471013 0.0154965613 0 0 0]
   [0.022516584 0.0349111706 0.0210680403 0.0192404501 0.028672399 0.0249777939 0.0204861909 0.0297163874 0.018559048 0.0221275948 0.0432363823 0.0253848024 0.0169703346 0.0444238894 0.0188911613 0.0177343823 0.0171730071 0.0221053325 0.0195709541 0.0297643635 0.0357389301 0.0218262281 0.0280045904 0.0331120864 0.0322974473 0.0278674662 0.0293692648 0.0228667874 0.0225145668 0.0164601281 0.0290260073 0.0235929899 0.037121281 0.0337158814 0.0274387617 0.0185662061 0.0238515381 0.0390995741 0 0 0]
   [0.0250744801 0.0170292519 0.0358538292 0.0192776863 0.0326639302 0.0309658404 0.024351012 0.033783406 0.031194441 0.0254567042 0.0266226027 0.0367216095 0.0196362343 0.0409935303 0.0254786313 0.0219915789 0.0311802626 0.0178402085 0.0232888926 0.0301469825 0.0355318561 0.0239593014 0.0350159369 0.0259001423 0.0186302327 0.0184696428 0.0322604552 0.0173086 0.018119676 0.0196749866 0.0374372862 0.0224294625 0.0232939404 0.0271787886 0.0213766769 0.0176260862 0.0157216 0.0405141897 0 0 0]
   [0.0300592948 0.0184366386 0.0276352726 0.0240486972 0.0306597035 0.0194841251 0.0196169447 0.0176828988 0.0303891581 0.0421007089 0.0268275607 0.0279210787 0.031955272 0.0208192449 0.0253695436 0.0425976887 0.0310032163 0.0404814631 0.0249046739 0.0283087939 0.0240957513 0.0203873087 0.0182834119 0.0179208741 0.0328352563 0.0434607826 0.0194655061 0.016494453 0.016796438 0.0168171134 0.026065819 0.032603655 0.0217023101 0.0323107503 0.0161599685 0.0348821692 0.0291952454 0.0202211812 0 0 0]
   [0.0166930817 0.0166595355 0.0323262103 0.0170446523 0.0180927478 0.0236923657 0.0266509168 0.0217351709 0.0373100527 0.0409482345 0.0249764211 0.0401667021 0.0281310733 0.041341193 0.0229205135 0.0182378236 0.0222353023 0.0237484667 0.0266593043 0.0424599648 0.0210440718 0.0166271981 0.0181138963 0.0210591331 0.0230949149 0.0354440585 0.0409094691 0.0246434901 0.0214454588 0.0259505138 0.03472111 0.0216233116 0.0352007598 0.0197197609 0.0195142422 0.0302659329 0.0216488019 0.0269441977 0 0 0]
   [0.0285594985 0.0303755552 0.0158772562 0.0302782841 0.0219072923 0.0355355181 0.031832356 0.0290884431 0.017713014 0.0307431314 0.0305852871 0.0363313034 0.0402964093 0.015861582 0.025418574 0.0194434077 0.0299343672 0.0248304345 0.0377775729 0.0205508657 0.0237192791 0.0293684769 0.0308614 0.0271335114 0.028282959 0.0191801898 0.0204730295 0.0301089976 0.0211105198 0.0193692576 0.033891011 0.0164351407 0.023220377 0.0181655828 0.0287515111 0.0258389153 0.031841971 0.0193077065 0 0 0]
   [0.018771166 0.0161655415 0.021804316 0.0355004221 0.0287183933 0.032505773 0.0337034166 0.0300226901 0.0350434035 0.0172608253 0.027261328 0.0204761606 0.0236017685 0.0361491777 0.0180164129 0.0246932022 0.0300456267 0.0178130399 0.0265424978 0.0314329937 0.0312521495 0.0370501727 0.0148414215 0.016837826 0.0219620652 0.0359896757 0.0285375193 0.0230215974 0.0233843885 0.0175215 0.0367280506 0.0208173282 0.034737356 0.0254177079 0.0158112161 0.0180322304 0.0356623307 0.036867246 0 0 0]
   [0.01703359 0.0362997353 0.0244833268 0.0320627652 0.0318641253 0.0357886553 0.022573771 0.0332987048 0.023359241 0.0192091521 0.0244667679 0.0405416191 0.0414639637 0.0227481294 0.0202126876 0.0328432024 0.0275286231 0.0205226224 0.0212084316 0.0180653557 0.0297513567 0.0298850909 0.0192107968 0.0200401358 0.0300528649 0.0171758551 0.0252870359 0.0303318985 0.0185472053 0.0330051556 0.0295249671 0.0192726832 0.0314428844 0.0282927807 0.0293526929 0.0168378856 0.0224828813 0.0239314176 0 0 0]
   [0.0183286313 0.0251365043 0.0262683053 0.0213146899 0.018991733 0.0186728 0.0357854106 0.0170118753 0.0215941034 0.0328015462 0.0224423539 0.0326939262 0.0193512086 0.0397462025 0.0174718965 0.0408272147 0.0274586976 0.0169146042 0.0225845054 0.0355950482 0.0216619521 0.0361977294 0.0252235644 0.0201202203 0.0170734785 0.0289676562 0.0189307239 0.0243375786 0.0189881828 0.0265091751 0.0193807185 0.0267285258 0.0379325114 0.0202644952 0.0404471867 0.0403688066 0.0312834196 0.0345928185 0 0 0]
   [0.0255684871 0.0280260313 0.0232178885 0.0219909027 0.0249415822 0.0189539883 0.0261788964 0.0266318377 0.0240860116 0.0182098709 0.0365344286 0.0251700226 0.0361391045 0.0174486078 0.0378707 0.0166232679 0.0166269969 0.0288345255 0.021436505 0.0207772739 0.0263430029 0.0400970057 0.0261559971 0.0362655967 0.0259619 0.0177347772 0.0247419924 0.0254836194 0.0405857563 0.0288931672 0.0355520472 0.0154862413 0.0321202539 0.0264087785 0.0213864483 0.0255289655 0.0202749483 0.0357126333 0 0 0]
   [0.0287792254 0.0235182364 0.0391960032 0.0201712176 0.0179099441 0.0403672457 0.0190079361 0.0164164621 0.033302132 0.0233966187 0.0320031457 0.0266269743 0.0173016395 0.0241599679 0.0373818353 0.0368418768 0.0303399414 0.015885802 0.0236004833 0.0319097452 0.0331733152 0.0257708132 0.0273298621 0.0220970828 0.0188850723 0.0310257562 0.0315951183 0.0330123678 0.0262251142 0.0208621081 0.0371685438 0.0159876533 0.0196472649 0.0215562545 0.0201123804 0.0289701968 0.0321901441 0.0162745919 0 0 0]]

  [[0.0177014116 0.0381489918 0.0175699312 0.0224222206 0.0166343637 0.0252469517 0.0159749221 0.0173411686 0.0167646091 0.0356508158 0.022897983 0.0374230444 0.0302028302 0.0235516839 0.0155487079 0.0205541607 0.0328814164 0.0386742428 0.0385882407 0.0322328471 0.0260941386 0.0244091544 0.0231621917 0.0364392139 0.0205119308 0.0204168521 0.0370254107 0.0253329091 0.0170870423 0.0217852127 0.0356534719 0.0359754376 0.0341476314 0.0277181901 0.0195864253 0.0147244576 0.0332921781 0.0306275506 0 0 0]
   [0.0208194144 0.0246301033 0.0237448495 0.0165838208 0.0271120332 0.0338865817 0.0246463045 0.0405740961 0.0325829387 0.0233875904 0.0217573047 0.0391301587 0.0346597321 0.0287159178 0.021359954 0.0172408819 0.0186251123 0.0187951829 0.034625385 0.0323855206 0.0218705926 0.0327898338 0.0170674808 0.0164702535 0.0157086793 0.0328811407 0.0162679 0.0229731705 0.0183464214 0.0285895616 0.0317254253 0.0281234682 0.0218169577 0.0241070408 0.0401975662 0.0338184945 0.0264238529 0.0355592221 0 0 0]
   [0.0369885676 0.0378032364 0.0240067244 0.016491862 0.0360018499 0.0348128863 0.0171919353 0.0335353576 0.0210657474 0.0298821311 0.0170435198 0.0158263817 0.0171405822 0.018539099 0.0205614921 0.0332852192 0.0303934347 0.0300701018 0.0161721446 0.0208238456 0.0275654681 0.0248061325 0.0241283085 0.0422521979 0.0292219501 0.0374169908 0.0197304413 0.0230195038 0.0231349394 0.0187465195 0.0349766165 0.0263689384 0.034963578 0.0354297385 0.0326286964 0.0170676708 0.0221324395 0.0187737849 0 0 0]
   [0.016052613 0.0206075199 0.0259496085 0.0265474711 0.0381022915 0.036032 0.0203207862 0.0313995406 0.0233740062 0.0162775498 0.0174763259 0.0205793045 0.0367054716 0.0249026492 0.0294798724 0.0413784198 0.0205164719 0.0256892294 0.0262699053 0.0163978338 0.0238982681 0.0250987485 0.0240462981 0.0241346639 0.0194433071 0.0407970436 0.0281069595 0.0173196439 0.0293526035 0.0250381865 0.0416274481 0.0228019785 0.0378802679 0.0265872721 0.0217030738 0.017985709 0.0168114714 0.0433081798 0 0 0]
   [0.0354096852 0.0272156261 0.0366985388 0.0259751752 0.0303032 0.0169908553 0.0307212733 0.0275173634 0.0339359716 0.0241459925 0.0174486842 0.0310495719 0.029462561 0.0364141427 0.0198651124 0.0231238659 0.0305792559 0.0254614241 0.0229531936 0.0271879788 0.0357371978 0.0217096414 0.0347462781 0.0191577263 0.0282643232 0.0166421439 0.0138756931 0.0351245217 0.0193832945 0.0167715587 0.0239233598 0.0317504629 0.0304381624 0.0193412639 0.0157060325 0.0185223315 0.0323669389 0.0340795927 0 0 0]
   [0.017084226 0.0203732084 0.0438795686 0.0193352494 0.0195432976 0.0181081165 0.0338370055 0.0227975938 0.0200325139 0.0387540348 0.0317256525 0.0237455778 0.0182576217 0.0201358665 0.0208575726 0.0330303535 0.0445788093 0.0269685704 0.0205841754 0.0239946526 0.0372129343 0.024539493 0.0197085775 0.022518279 0.019219324 0.0214360617 0.026378775 0.0251939725 0.0421351492 0.0418481417 0.0397491977 0.0252338685 0.0348399729 0.0228752717 0.0171994846 0.0206950102 0.0194468983 0.0221458357 0 0 0]
   [0.0266827364 0.016376378 0.0162833948 0.0187449064 0.0406275615 0.024166001 0.029956406 0.0412929878 0.0182573125 0.016114939 0.0191398934 0.0406745784 0.0165730566 0.0184303354 0.0307758916 0.017289836 0.0202234928 0.0211837068 0.0288612191 0.023611417 0.0255489908 0.0217454042 0.0286079422 0.0176612418 0.0363741294 0.0288353469 0.0280737169 0.0386442915 0.0287358817 0.0386244841 0.0230308659 0.0260564331 0.0265512187 0.0289176237 0.0363997668 0.0263533071 0.0157478619 0.0388253704 0 0 0]
   [0.0283496361 0.0206201598 0.0227701385 0.0363534167 0.0176678821 0.018858349 0.0318548568 0.0158031862 0.0168142393 0.0407701656 0.019209696 0.0408447236 0.0160614 0.0279940367 0.0326360725 0.0411168858 0.0359938368 0.0295821019 0.0185814202 0.0251092296 0.0300686266 0.0397040583 0.028564699 0.0246139951 0.0171496384 0.019640781 0.0304050688 0.0163743626 0.0407524817 0.0213185754 0.0363051966 0.0331284441 0.0232075714 0.0164817404 0.0190289635 0.0328087434 0.0170941465 0.0163614303 0 0 0]
   [0.0246776529 0.0307129212 0.0333863683 0.0385585539 0.0274127088 0.0169522259 0.0173250344 0.023820255 0.0261665508 0.0207248684 0.0325105861 0.0206531566 0.0185816474 0.0193781797 0.0324018486 0.0251041576 0.0233240426 0.0311952252 0.0375139 0.0167072508 0.0204864126 0.0228728317 0.0296952296 0.0218213517 0.0169338882 0.0165870767 0.0183742177 0.0442354158 0.0171041526 0.0302469619 0.0227778479 0.0358566977 0.0330875516 0.0278076269 0.0373567604 0.0329538621 0.037368495 0.0173265487 0 0 0]
   [0.033339072 0.0237935074 0.0379559956 0.0159510095 0.0326175839 0.0234991033 0.0238907821 0.031758666 0.029219754 0.0155478446 0.0181986056 0.0236823447 0.0182453748 0.0216065682 0.0273973867 0.0382507518 0.0151263364 0.018759368 0.0317647457 0.0218516681 0.0187893063 0.0306716599 0.0351360217 0.0338648148 0.0261145122 0.0263731591 0.0395661443 0.0270070452 0.0228862893 0.017020883 0.0397247225 0.0152763091 0.0392625518 0.0350832976 0.0189826153 0.0149224093 0.0281239264 0.0287378486 0 0 0]
   [0.0255905353 0.0156073738 0.0183046181 0.0281819459 0.0237261765 0.023361709 0.0165423788 0.0164833013 0.0259814933 0.0307017621 0.0375264101 0.0367562473 0.0202748179 0.0170774385 0.0176696908 0.0215018708 0.0269855317 0.0149981752 0.0389052853 0.0157648399 0.0380511545 0.038987726 0.0296556484 0.0214696191 0.0235095266 0.0212669726 0.0358713642 0.0376743861 0.0239170752 0.0291516595 0.0219097901 0.0272926129 0.0375501327 0.0320920758 0.0274663661 0.0249903444 0.0221439488 0.0350580104 0 0 0]
   [0.0389092602 0.0277712904 0.0308702718 0.0315351188 0.027227534 0.0307514127 0.0290900748 0.018126566 0.015600373 0.0235947706 0.0235932805 0.0167584512 0.0383120403 0.0226046033 0.0222187154 0.0315594748 0.0205809176 0.0363954417 0.0193758644 0.0311172102 0.0327222869 0.0168912839 0.0344549641 0.0236566179 0.0391963944 0.025868142 0.0265388861 0.0276652575 0.0201176889 0.0170300622 0.0239041969 0.0354837216 0.0239625052 0.0185444783 0.0245711599 0.0175880566 0.0186511613 0.0371604972 0 0 0]
   [0.0170333646 0.030654788 0.0312263407 0.0187879223 0.020565385 0.0285587255 0.018094182 0.0339717679 0.0278836805 0.0390178636 0.0329084 0.0269252788 0.0325010158 0.016615767 0.0179367308 0.0248064417 0.0352024585 0.0400802158 0.0374651924 0.0160674937 0.0205108896 0.0177209042 0.0213446841 0.0406100862 0.0237157736 0.0341788232 0.0167949926 0.0220298283 0.0178525504 0.0233726501 0.02398929 0.0194062702 0.0230569802 0.0171356406 0.0378243029 0.0374098271 0.0394910276 0.0172524434 0 0 0]
   [0.0242789574 0.0370140597 0.0236668047 0.0169440601 0.0233556144 0.0254823714 0.0295324959 0.0252477396 0.0233735759 0.0204653442 0.0236780252 0.0368406847 0.0335299782 0.0204705708 0.0188675299 0.0238077287 0.0327304527 0.0215995945 0.0249353498 0.0282187406 0.0166004077 0.0295831 0.0164906029 0.016353935 0.0406085849 0.0300988648 0.0244959146 0.0356482305 0.0247159638 0.0278263912 0.0272064488 0.0288448241 0.0357704796 0.0186869092 0.0316478387 0.0199017804 0.0339344554 0.0275456421 0 0 0]
   [0.0175962299 0.0170773361 0.0252588801 0.0181217 0.0177626368 0.029553758 0.031144632 0.0369852595 0.0196654964 0.0241616573 0.0378650166 0.0178007297 0.0161088668 0.0273055714 0.0188365616 0.036494866 0.0239278898 0.0221916437 0.0340062566 0.0219931044 0.0214490481 0.0317413 0.0357336551 0.0332530215 0.0385552 0.0261150748 0.0299795363 0.022285752 0.0370190293 0.0153417112 0.0370852426 0.0210610833 0.0164196435 0.036484208 0.0191449281 0.0270190351 0.0167649463 0.0406895 0 0 0]
   [0.0380451 0.0197113808 0.0206174534 0.0383627713 0.0343540087 0.0306603555 0.0345617868 0.024941843 0.0389016718 0.0314712077 0.0154749667 0.0153602911 0.0317609422 0.0335892625 0.0168252941 0.037875779 0.0228235535 0.0265488 0.0148119545 0.0253732931 0.0289022513 0.0241469033 0.0298322681 0.0178106371 0.0230431501 0.0337279961 0.0170070082 0.0221869275 0.0289457738 0.0188307632 0.0379864834 0.0319648273 0.0188470706 0.0202272311 0.0259244777 0.0150212916 0.0204141047 0.0331090949 0 0 0]
   [0.0162047893 0.0339714438 0.0205723308 0.0417970642 0.0285813026 0.0213697366 0.0159030166 0.0256359912 0.0379522853 0.0292821173 0.0188741069 0.0410518162 0.0414689034 0.0353365727 0.0417365171 0.0413795039 0.0182603803 0.0216061417 0.0188062545 0.0172647573 0.0161514729 0.0401231796 0.0235918984 0.0407361612 0.0169415083 0.0218020249 0.0206608605 0.0156206228 0.0247130152 0.0326564424 0.0199810695 0.016746603 0.0193343479 0.0228003748 0.040404968 0.0226308871 0.0164810643 0.0215685442 0 0 0]
   [0.0381864756 0.023253927 0.0333390199 0.0294997189 0.016810067 0.0362397432 0.0332379825 0.0322584249 0.0353448279 0.0302368589 0.0225101076 0.0318533033 0.0360431485 0.0190967023 0.0266707391 0.0195594 0.0201721471 0.0169195812 0.02035941 0.0283557121 0.0233587846 0.0277421325 0.022044545 0.0186762251 0.0408180952 0.035538733 0.0357998274 0.0166697539 0.016368391 0.026786631 0.0169307366 0.0241581723 0.029287288 0.0247726422 0.0182224717 0.0223683808 0.0294758808 0.0210339818 0 0 0]
   [0.0302264914 0.0386092179 0.0194719844 0.0396846533 0.0189437028 0.0158104245 0.0191481113 0.0249629673 0.0361391529 0.0239827819 0.021350028 0.0189446416 0.024057053 0.0363197848 0.0297819134 0.0168886483 0.0227232035 0.0173526965 0.0333874486 0.0234363694 0.0172236376 0.0182575788 0.0354970098 0.0196425207 0.0420211591 0.0164289698 0.0191117041 0.0258535426 0.041565191 0.0262541622 0.0254547019 0.0337644257 0.0192939211 0.0185905807 0.0410974883 0.0295927972 0.0193348285 0.0397945382 0 0 0]
   [0.0385248512 0.0151317464 0.0363917165 0.0183242951 0.0381552838 0.0229886435 0.0211253632 0.024962917 0.0251339022 0.035981819 0.0296617858 0.0261427332 0.0371723212 0.0288869347 0.0213650167 0.0242921617 0.0369089618 0.0301539581 0.0174067374 0.0197555088 0.0149670113 0.0290804058 0.0163079221 0.0398652069 0.0173870437 0.0282000042 0.0279883202 0.0196300261 0.0185438022 0.0254151747 0.0352990031 0.017228296 0.0159721244 0.0294386633 0.0242288541 0.026696343 0.0256396681 0.0396454893 0 0 0]
   [0.0183798503 0.0160424467 0.0341101 0.0326511636 0.0189073607 0.0211814623 0.031694904 0.0176857244 0.0239787437 0.0181408562 0.0369718745 0.0161756556 0.0260518659 0.0195094701 0.0308940317 0.0369430073 0.0358636715 0.026849 0.0342397466 0.0216767918 0.0318545513 0.0198480934 0.0309215076 0.0308833122 0.0295319669 0.0150075611 0.0374148823 0.0368827768 0.0326144248 0.0367421731 0.0186744984 0.0245993566 0.0154365301 0.0256989915 0.0290054362 0.0145886149 0.0271188281 0.0252288096 0 0 0]
   [0.0212329235 0.0254840963 0.0297523178 0.0163709559 0.0172198955 0.0197333507 0.0362317674 0.0281225387 0.0224065538 0.0238072239 0.0180303361 0.0353190377 0.0236012898 0.026693061 0.0223465096 0.0271322392 0.0186636597 0.0394749567 0.0176565293 0.0285508074 0.0313523151 0.0268291719 0.0272736773 0.0176894329 0.0248378776 0.020368576 0.0180804506 0.0318567492 0.0198758356 0.0392713696 0.0352623165 0.0428788401 0.0332998596 0.0432645679 0.0226357114 0.02194128 0.0284316894 0.0170202479 0 0 0]
   [0.0161975455 0.0370374732 0.0368086621 0.0395416468 0.0224099755 0.0169358924 0.0200041682 0.0147904055 0.0379681848 0.0223759152 0.0235891975 0.0274049044 0.0213291887 0.0218212213 0.0371983089 0.0282983072 0.0273387823 0.0202362873 0.0375662521 0.0228079725 0.0241804179 0.0208280664 0.0199284628 0.0180061832 0.0392877907 0.0152775506 0.0264532529 0.0158427116 0.0306501482 0.0315339305 0.0213392973 0.0208333135 0.0297058728 0.0392474644 0.0167713724 0.033216048 0.0338864364 0.0313513391 0 0 0]
   [0.0339850858 0.0214119647 0.0169033539 0.0321778283 0.0259424914 0.0361901708 0.0177059211 0.0370116197 0.0207705535 0.0283014197 0.0265686736 0.03706811 0.0360403657 0.0393296815 0.0340983495 0.0179190729 0.019086102 0.0226586275 0.0320388265 0.0299206432 0.0296425819 0.0157149043 0.023833327 0.0172458962 0.0175099913 0.0387230925 0.01771391 0.0212229136 0.036469236 0.0394403152 0.0178642366 0.0327101313 0.0166740697 0.0183896068 0.0345294736 0.0211714171 0.0198981147 0.0161179453 0 0 0]
   [0.0224345345 0.0228466317 0.0344580859 0.0376983657 0.0196318533 0.0322627425 0.0195605811 0.019020807 0.0182668101 0.0317114219 0.0252220649 0.0154651 0.0355277359 0.0208926611 0.0385391787 0.0315284766 0.0198897291 0.0170921478 0.0265575964 0.0359033719 0.0232252963 0.0232219622 0.024015611 0.0257128365 0.0375248194 0.0290937275 0.0239833854 0.0340328701 0.0319811739 0.0161858257 0.0270105563 0.0293126646 0.0272596311 0.0275381058 0.01525022 0.0186937638 0.0258028097 0.0356447548 0 0 0]
   [0.0261172429 0.0219198838 0.0372963 0.038111452 0.023282174 0.0207956135 0.0178536046 0.0276645627 0.0312140938 0.0148141412 0.0151205957 0.016587453 0.0326871164 0.0209805779 0.0390317179 0.0342226624 0.018149808 0.0263053104 0.0360520221 0.0246450864 0.0335302204 0.0315136276 0.0168055296 0.0337310471 0.0244007483 0.0355647802 0.0188168231 0.027677549 0.0196666662 0.027042171 0.0353260525 0.0224447399 0.0274311863 0.0210566 0.0170634575 0.0242973175 0.0268275924 0.033952456 0 0 0]
   [0.0297175329 0.0202823523 0.0312343594 0.0395155624 0.0186489765 0.021604538 0.018934885 0.0280330256 0.0212588087 0.0248376038 0.016172329 0.0348415636 0.0287900697 0.0284624 0.0355005972 0.0243882015 0.0369992219 0.027283052 0.0192509498 0.0192680433 0.0171549749 0.0259331688 0.034124542 0.0156276468 0.0217075273 0.0261597708 0.0226820167 0.0297853369 0.0228718612 0.0291075669 0.0182922371 0.0257681031 0.0171331856 0.0406433679 0.0407595783 0.0311916843 0.0331668556 0.0228664875 0 0 0]
   [0.0172178242 0.0403187871 0.0399348214 0.0393075123 0.0245142467 0.0368601494 0.0240278412 0.0242867526 0.0399669521 0.0396421514 0.0201598313 0.0378591456 0.0238005 0.0272330828 0.0356838629 0.0185142811 0.0279456489 0.0241156537 0.0190539118 0.0259214714 0.0356505513 0.0216075443 0.0344812572 0.0308190733 0.0174708646 0.0150052384 0.0277937427 0.0323184095 0.0221263971 0.0160929989 0.0205340255 0.0157908183 0.0194536056 0.0156748947 0.0188563745 0.0176393799 0.0261946917 0.0261257235 0 0 0]
   [0.0329858139 0.024335904 0.0241698902 0.023405347 0.020124672 0.0192933977 0.026609892 0.0336520225 0.0320778638 0.0147733605 0.0342104211 0.0227802824 0.0307518411 0.0205130763 0.0352900587 0.0330103189 0.0352241248 0.0305560194 0.0278106648 0.0288881157 0.0295830797 0.0212820563 0.0163939744 0.035700012 0.0238921139 0.0316650383 0.0370671228 0.0189601723 0.0241755638 0.0320970975 0.0204126891 0.0264737252 0.0171342101 0.0389426537 0.0179146975 0.0148600563 0.0242486838 0.0187339336 0 0 0]
   [0.0349515527 0.0264206622 0.032261055 0.030744141 0.0344264023 0.0178364534 0.0197945219 0.0166600551 0.0303557161 0.0172095504 0.0197332762 0.0404430814 0.0160325728 0.0307796039 0.0249170829 0.0172977243 0.0289130546 0.0291031618 0.0271582771 0.0366285592 0.0199330896 0.0217619706 0.0190756395 0.0279426612 0.0216230303 0.026652243 0.0193421952 0.0186493266 0.0198902916 0.0392368175 0.0247137509 0.0247907881 0.041146677 0.0282047186 0.0377707817 0.0270848162 0.0320585594 0.0184560586 0 0 0]
   [0.0302531123 0.0273777395 0.0248933323 0.0248304103 0.0194579158 0.0187457241 0.0155491801 0.0146685122 0.0302086603 0.0199444816 0.0150028141 0.0371861272 0.0156795792 0.0366419293 0.0264204703 0.0196803715 0.0260544606 0.0372882672 0.0271574855 0.0239948835 0.0312049892 0.0328096561 0.038403362 0.0223081335 0.0322699957 0.034560781 0.0167443641 0.0367903896 0.0213370249 0.0156339686 0.0186663326 0.018643396 0.0389589407 0.0372804292 0.0331387 0.032749258 0.0212776791 0.0261871908 0 0 0]
   [0.0351739712 0.0277847853 0.0260121915 0.0160048101 0.0309660584 0.0398037918 0.0316670351 0.0250964295 0.0401179753 0.0242263898 0.0214378182 0.022071626 0.0225081481 0.0241095144 0.0200459547 0.0248350836 0.0185584296 0.0186446719 0.018473709 0.030423956 0.0416426472 0.0291824564 0.0207284521 0.0298661068 0.0168275349 0.016662091 0.0414054096 0.0271703787 0.0318898074 0.0199345797 0.0237741284 0.0206576884 0.0420582518 0.0170659423 0.0345940292 0.0321749263 0.0180195011 0.0183836818 0 0 0]
   [0.0311850719 0.0306677651 0.0224438831 0.0271828324 0.0162466057 0.0167032443 0.0168374702 0.0364835672 0.0259303376 0.0213733688 0.0321857519 0.0251495596 0.0300298855 0.0246357322 0.0233576167 0.0149296122 0.0339879282 0.0152793368 0.0197372604 0.0312060807 0.0300142467 0.0263913386 0.0226520747 0.0234497301 0.0282629151 0.0272525456 0.0155707896 0.0309570059 0.0367052853 0.0197377633 0.0167983435 0.0395238549 0.0355397463 0.0353318863 0.0300229676 0.0215381272 0.0256423857 0.0390560776 0 0 0]
   [0.0271438584 0.0328936465 0.0195171572 0.0201854929 0.0305219386 0.0159028471 0.0174971912 0.0235051308 0.0166979656 0.0330132358 0.0208553113 0.0184083041 0.019036077 0.0170652531 0.0330569893 0.03686985 0.037608102 0.0299715344 0.0276838895 0.0157015603 0.0177805014 0.0225629378 0.0184263233 0.0360583141 0.0272466894 0.0382590406 0.0282963347 0.0386146978 0.015464142 0.0179131906 0.0315399691 0.0414635 0.0257915203 0.035292428 0.035271246 0.0222182069 0.0331208035 0.0215448439 0 0 0]
   [0.034218803 0.0219278578 0.0284710228 0.0201187171 0.035033796 0.0258726776 0.0164929442 0.0367733948 0.0206614845 0.0306960791 0.0255272444 0.0175329633 0.0214708038 0.0351878554 0.01629759 0.0162199363 0.0401205868 0.0423804671 0.0171827953 0.0172238816 0.0257338081 0.0162932221 0.0188303199 0.0198029038 0.0160243791 0.0204562675 0.016921116 0.0308542829 0.0400670916 0.0270636 0.0426476412 0.0310129207 0.0385636352 0.0334461704 0.0409117378 0.016571857 0.0221174248 0.0232707821 0 0 0]
   [0.0170852263 0.0200397968 0.0237271804 0.0269565824 0.0370635428 0.0326259546 0.0364268199 0.0362539664 0.0245039109 0.0365673266 0.0209900867 0.0269986391 0.0345949084 0.024376 0.0172538459 0.0348935425 0.0297073796 0.0199997704 0.0378611162 0.0213245 0.0258057 0.0161229875 0.0258476101 0.019362608 0.0281967148 0.0236191843 0.022558704 0.0299834907 0.0419904441 0.0221326556 0.0164643936 0.0158702 0.0162280947 0.0170601942 0.0299556684 0.0318198539 0.0228942037 0.0348371491 0 0 0]
   [0.0159471221 0.0260283388 0.0255241077 0.015982151 0.0192523263 0.0203360766 0.0357483551 0.0182846431 0.0168673862 0.0235063024 0.0358812772 0.0187493712 0.0334821567 0.0162625462 0.0202917438 0.0286651086 0.0300677735 0.0216657501 0.0315455161 0.0358626097 0.0237627197 0.0165205058 0.0237807855 0.0369493887 0.035432931 0.0264949091 0.0317807682 0.0163874254 0.0296430755 0.0197601262 0.0346697 0.0208301544 0.0388288 0.0376714431 0.0298291296 0.0234564692 0.0382373221 0.0260136519 0 0 0]
   [0.0376881287 0.0162585601 0.0271602627 0.0163932499 0.0242156684 0.0209671222 0.0204471815 0.0283504911 0.0310500506 0.0185372178 0.0170173515 0.0284990277 0.0365271345 0.030286083 0.0247823894 0.0269684717 0.0235374421 0.0179014858 0.0405494943 0.0336649083 0.0291437786 0.0374440812 0.0160678979 0.0172647685 0.0315137357 0.0295881946 0.0307809971 0.0345899574 0.0165249929 0.0262702275 0.01549011 0.0229234174 0.0177101 0.0398724303 0.0327427424 0.0236151535 0.0392213948 0.018434301 0 0 0]
   [0.0368787088 0.0216910243 0.0393191501 0.0237534158 0.041650869 0.0218920279 0.0204034876 0.0313070677 0.0309660435 0.0337612443 0.0193857234 0.0196411684 0.0233093873 0.0258738305 0.0299053956 0.0268317 0.0177300777 0.0215357337 0.0285764411 0.0218469147 0.0313300528 0.0239374917 0.0245674103 0.0299727134 0.0295285694 0.016793834 0.0180751011 0.0319214873 0.0167274326 0.0175379105 0.02062531 0.027505463 0.0338571 0.0269712433 0.0253668819 0.0270018131 0.0319525711 0.0300681237 0 0 0]
   [0.0224079918 0.0199459754 0.0208957102 0.0303528775 0.0208328236 0.0282401424 0.0181439333 0.0369755179 0.0252490938 0.0367116705 0.0281099509 0.0297091957 0.0292675365 0.0162599105 0.0148518281 0.0357946157 0.0333921053 0.0357035622 0.0294301771 0.027006546 0.0259033758 0.0249035843 0.0313721597 0.0225881841 0.0289315805 0.0172342956 0.0251391102 0.0169318169 0.0276227426 0.0249416307 0.0236114915 0.0305766389 0.0319840163 0.0211415142 0.0363877118 0.0267985854 0.0181714687 0.0264789276 0 0 0]
   [0.0305745285 0.033888936 0.040134307 0.037514627 0.0284965169 0.017489668 0.0206376221 0.0172704328 0.0307426155 0.0368601605 0.0311468262 0.0169752911 0.0162050072 0.0152007071 0.0385983 0.0226991624 0.0161430202 0.0176511351 0.0280897189 0.015015508 0.0248436853 0.0261534899 0.0292629171 0.0209805965 0.032538306 0.0311627705 0.0235121287 0.037716832 0.0273004305 0.0244048089 0.0359281227 0.0162689481 0.0153414952 0.0249110945 0.033907976 0.0217053387 0.0310623907 0.0316646174 0 0 0]]

  [[0.0355856568 0.0251841694 0.022505166 0.0197376069 0.0283007193 0.0326078832 0.0219414104 0.0405518189 0.0179727916 0.0323084407 0.0296744537 0.0245632622 0.0314655192 0.0153368702 0.0158196148 0.0396923907 0.0172472801 0.0292986892 0.0310373083 0.0184323452 0.0291483905 0.0230618604 0.0341679268 0.0381126888 0.0405702032 0.0221719202 0.022989409 0.0153380195 0.0387149714 0.0222059246 0.0394629091 0.038295567 0.0161590092 0.0190630071 0.0188731272 0.015543838 0.02140297 0.0154549256 0 0 0]
   [0.0320544913 0.0182060506 0.0198551621 0.039430622 0.0263543595 0.0218237694 0.0281195641 0.0286265742 0.034711726 0.0172162279 0.0253975727 0.021874262 0.019299712 0.0221680664 0.0370108709 0.0170970932 0.0260865558 0.0176269375 0.0394460149 0.0215388685 0.0293674357 0.0241492819 0.0317974538 0.0272936039 0.0215269737 0.0181871634 0.0291716717 0.028366901 0.0368050784 0.033740826 0.0278937854 0.0178928114 0.0215811878 0.0221273042 0.0188138261 0.0278300773 0.042420961 0.0270891562 0 0 0]
   [0.0352576412 0.0400834 0.0346678682 0.0380708836 0.0297087114 0.017070543 0.0274593644 0.0333947949 0.0245193467 0.0252282284 0.0186025333 0.0231500193 0.0354872234 0.01970529 0.0305523 0.0327736214 0.033490289 0.0270004086 0.0305146407 0.0183484647 0.0242412817 0.0157135278 0.0195970647 0.039953202 0.0304605979 0.0336390883 0.0242654383 0.025504319 0.0168706346 0.0163332205 0.0220071375 0.0209936537 0.029308537 0.017345991 0.020680394 0.0173199289 0.0270095095 0.0236708838 0 0 0]
   [0.0272972323 0.0340116434 0.0358558558 0.0185502805 0.0382254906 0.0339465626 0.0242157653 0.0178227779 0.0156691503 0.0241490193 0.0170910545 0.0291587263 0.0158119425 0.0300755519 0.0335714668 0.0235242303 0.0194517076 0.0158498082 0.015856456 0.0257295836 0.0256733671 0.0188472215 0.0289853346 0.0396913663 0.0236684084 0.0320353359 0.0254832674 0.0340796039 0.0244188923 0.0341149531 0.0268629547 0.0181470551 0.0393436179 0.0335261 0.0383086205 0.0151231103 0.0295590572 0.0162674114 0 0 0]
   [0.0307125896 0.0226778612 0.0375136808 0.021069 0.018435875 0.0202989876 0.0426581167 0.017066909 0.0406029113 0.0392575487 0.0392920598 0.0293571278 0.0309575554 0.0213272199 0.0280343536 0.0265302565 0.0388967954 0.0240286626 0.0214692149 0.0284920055 0.0210171901 0.0241183378 0.0179789383 0.0180373304 0.0258012395 0.0170631837 0.0253921375 0.0201604161 0.0285198521 0.0208645072 0.023689637 0.0230396558 0.0174157936 0.0294040218 0.0264735203 0.0312361233 0.02880219 0.0223072246 0 0 0]
   [0.026322 0.0196633656 0.0353881679 0.0294999816 0.0364663638 0.0362067968 0.0177597292 0.0187197216 0.0171194039 0.0362008251 0.0170058832 0.035527885 0.0154679529 0.0189496372 0.034578681 0.0380857699 0.0200633612 0.02089094 0.0168414041 0.0169782192 0.0262135286 0.040326111 0.0389800258 0.0270683467 0.020683635 0.0161138568 0.0211595651 0.0330714695 0.0193937253 0.0375557281 0.0279468577 0.0367780551 0.0168351568 0.0176248662 0.0230831895 0.0396155342 0.0217279848 0.028086314 0 0 0]
   [0.0383068249 0.0420310907 0.0404220708 0.0358081125 0.025761487 0.0184036531 0.0233942736 0.0199638885 0.0173259769 0.0345971808 0.0169500355 0.0301667955 0.0222743433 0.0319754891 0.0192828793 0.016400503 0.0194073655 0.021832753 0.0395144299 0.0309480634 0.0381954797 0.0371394455 0.0290713236 0.026944546 0.0307010934 0.017573826 0.0289676785 0.0224934909 0.0166500397 0.0198762193 0.0189964697 0.0229210798 0.0170467533 0.0322744846 0.0193511117 0.0228178594 0.0209396705 0.0332722515 0 0 0]
   [0.0198499337 0.0226770844 0.01562934 0.0384877659 0.0192590673 0.0194311664 0.0254050139 0.016558107 0.0203763433 0.0182855688 0.0415545 0.0420073643 0.0169982594 0.0387015 0.0343914293 0.0253530163 0.0244867243 0.0390245691 0.0181522537 0.0279592797 0.0192403141 0.0190800484 0.0311097298 0.0270061027 0.0312932618 0.0250077024 0.0295222 0.0357853658 0.0176845472 0.0421555527 0.0177293 0.0225331802 0.025887927 0.0200785 0.020648228 0.0392311923 0.0297878925 0.0216306932 0 0 0]
   [0.0223048478 0.0182933211 0.0204815865 0.0300579369 0.0267303716 0.0403095894 0.0289385431 0.0248326417 0.029898772 0.0189568922 0.0269470904 0.0239388049 0.0432983786 0.0266767461 0.0317564234 0.0246345121 0.026785247 0.0255179 0.0254679546 0.0178299453 0.0220333599 0.0260923244 0.017689677 0.0247252807 0.0298040789 0.0250519328 0.0171241146 0.0274737924 0.0223091766 0.0421703905 0.0306357965 0.031052392 0.0185556691 0.0216467455 0.0236022621 0.0307861324 0.0349538364 0.0206354782 0 0 0]
   [0.0284063276 0.0179219451 0.0322704799 0.0214759279 0.0166752581 0.024346631 0.0379197448 0.0271577984 0.0296028219 0.0279925801 0.0326449946 0.0223810542 0.0232101716 0.0225628149 0.0174979083 0.0349377692 0.0193145182 0.0381295606 0.023846142 0.0162771624 0.0338502377 0.0220649876 0.023949746 0.0361465625 0.023803601 0.0192656182 0.0309488177 0.0246372912 0.0357378535 0.0313199721 0.0295980666 0.0286125615 0.0294727907 0.0238771923 0.0284858737 0.0259245168 0.0220384356 0.015694214 0 0 0]
   [0.0367063768 0.0217469782 0.0189172253 0.0290988013 0.0266469102 0.0347083174 0.0293386262 0.0316105969 0.021473093 0.0223557875 0.02602 0.0192967579 0.0463252924 0.0434012413 0.0276851505 0.0213899184 0.038178049 0.017480975 0.0191136617 0.0427895412 0.0246417448 0.0289257672 0.0203805342 0.0261970479 0.0175783467 0.0223309547 0.0224236101 0.0172822233 0.02014 0.0433385558 0.0174515396 0.0221349839 0.0217721909 0.0353079401 0.021962218 0.0256256443 0.0177129824 0.020510355 0 0 0]
   [0.0201480519 0.0223210268 0.0207430683 0.0233574267 0.0350769311 0.0276484024 0.0157829318 0.0383731499 0.0300869625 0.0399700403 0.0339274853 0.0219432283 0.0400203504 0.0244864449 0.0303304922 0.0155052403 0.0154083287 0.0223805364 0.0157942902 0.0267401151 0.0236604586 0.0168846399 0.0196231026 0.0158591624 0.0183332339 0.0401493311 0.0257145111 0.0195035059 0.0210405551 0.0294860266 0.0307247117 0.0352033935 0.0381911248 0.0366544202 0.0366921909 0.0197417028 0.0195674561 0.032925956 0 0 0]
   [0.026628701 0.0241157357 0.0195643976 0.0274693016 0.0261433069 0.0209923591 0.0155261764 0.0185957234 0.0226965677 0.0206005611 0.0377951264 0.0292210374 0.0323858 0.0227615871 0.0252847467 0.0354166664 0.0202403907 0.0218733717 0.031491 0.0313153677 0.0168624595 0.0159439873 0.0177560803 0.0315781161 0.0198489353 0.0373247117 0.0283861309 0.0277836751 0.0284125544 0.0359985754 0.016688237 0.0391678028 0.0273969099 0.0166795887 0.0380037464 0.0252344925 0.0360580571 0.0307580177 0 0 0]
   [0.0220289025 0.024369115 0.0173961744 0.0230567213 0.0375131816 0.0280405711 0.0203296617 0.0253527984 0.0200489908 0.0172349513 0.0170028768 0.0321773849 0.0340812467 0.0343198255 0.0266582705 0.0379926153 0.0191984829 0.0225331485 0.0378121287 0.0292978212 0.0207411405 0.0238435213 0.03955 0.0293378849 0.0325909704 0.0310598575 0.0164212044 0.0362837091 0.0203244071 0.0235907603 0.018142106 0.0295658428 0.0177700073 0.0380379073 0.0290587116 0.0276366416 0.0226734653 0.0169269666 0 0 0]
   [0.017722588 0.0184425563 0.0173787605 0.0261386912 0.0312099457 0.0251624733 0.0350046381 0.028429633 0.0314149 0.0169264 0.0211239196 0.0264974739 0.0237670653 0.016361475 0.0237523559 0.022965271 0.0282226373 0.0371035784 0.025873458 0.0357697494 0.027787121 0.01634847 0.0183558706 0.032366246 0.0268436279 0.0190915205 0.0388822183 0.0164087228 0.0389710553 0.0379768126 0.0191850662 0.0165017415 0.0366860516 0.0245406218 0.0239934865 0.0217575878 0.035573978 0.0394622572 0 0 0]
   [0.0248305518 0.0230061803 0.0280068759 0.01609952 0.04141232 0.0258181226 0.0426375717 0.0163653214 0.0353076197 0.0238072556 0.0329306088 0.02034254 0.0246804729 0.0182880498 0.0431770757 0.022256529 0.0191696584 0.0243597031 0.0305933058 0.0279478468 0.034613207 0.0220547505 0.0200597029 0.0193568114 0.0230418835 0.0394100659 0.0162925776 0.0182638522 0.0333828591 0.0209492892 0.0178257097 0.0275059734 0.0266612936 0.031483721 0.018700866 0.038608823 0.0281148199 0.0226365887 0 0 0]
   [0.0275236201 0.0209273342 0.0191351492 0.0283253528 0.0265471917 0.0404226631 0.0225345176 0.0369350202 0.0233806856 0.0268438831 0.0188614745 0.0258389078 0.0240949802 0.0190772284 0.0205120221 0.0343021341 0.0319180712 0.0284682866 0.0401792638 0.0291870199 0.030189395 0.0363337398 0.0174000524 0.0295482706 0.0388214141 0.0164949074 0.0213519894 0.0205652956 0.0221428331 0.0223306622 0.0364969559 0.0166430473 0.0153445434 0.0381058902 0.0226911101 0.0222001895 0.0294205528 0.0189042911 0 0 0]
   [0.0366100557 0.0208687112 0.0444899425 0.0173894893 0.0424526148 0.0269635208 0.0194810964 0.0292071402 0.0211211611 0.0176715162 0.0405381694 0.0239206366 0.0265459828 0.0367345102 0.0252327509 0.0187044647 0.0395961516 0.0359268971 0.0237187538 0.0249526743 0.0181130227 0.0263669062 0.0174245909 0.0450929217 0.0256940331 0.017428726 0.0185361821 0.0203146264 0.0282614939 0.0202990305 0.0193189122 0.0197059456 0.0226231609 0.0247064959 0.0198201612 0.0239664298 0.0367130823 0.023487987 0 0 0]
   [0.0181615017 0.0324290842 0.0298131537 0.0367035307 0.0175048262 0.0198843461 0.0351156294 0.023166731 0.0286276396 0.0349741094 0.0294732209 0.0162175335 0.0356038176 0.0141769061 0.0277205482 0.0290606413 0.0255386885 0.0138345649 0.0265409071 0.0249130242 0.0257258303 0.0266979598 0.0256176367 0.0262154844 0.0300684664 0.0243395288 0.0257972963 0.0157486256 0.0368489847 0.0285095759 0.0336486362 0.0359014384 0.0233171433 0.0355896093 0.0220070668 0.0146267032 0.0301550254 0.0197245982 0 0 0]
   [0.0369388536 0.020295009 0.0252150036 0.0248855483 0.0295977835 0.0149772605 0.0294777025 0.0309461635 0.0360009521 0.0374320596 0.0163993761 0.0177244525 0.0181158148 0.0343870856 0.0348004587 0.0292066094 0.0218624733 0.0382781662 0.0193948317 0.021232985 0.0329820067 0.0269851722 0.0228496362 0.0310892314 0.0172413494 0.028723795 0.0206952784 0.0167595446 0.0242531449 0.031724371 0.0282327496 0.0229940172 0.019597102 0.0402441137 0.0373939089 0.0262189824 0.0166644882 0.0181824826 0 0 0]
   [0.0239345245 0.0284124855 0.0205065813 0.0167384427 0.0322000235 0.0169495773 0.0159157924 0.016359847 0.0416673347 0.0190792382 0.0232133809 0.035399802 0.0196416155 0.0344012789 0.0412268415 0.019885404 0.0382674448 0.0201400202 0.0261206571 0.0392738506 0.0175925121 0.0210876055 0.0249449667 0.0392127745 0.0344164036 0.0398476794 0.0333943889 0.0197531432 0.0168202575 0.0212861858 0.0211230144 0.0166529566 0.0409529 0.0274417624 0.0209034551 0.0202223081 0.0186731834 0.0363403633 0 0 0]
   [0.0233546756 0.0201566499 0.0365979783 0.0263790023 0.0178173333 0.0272079874 0.0423492528 0.0219370872 0.0364667699 0.0222292952 0.0192276072 0.0167299714 0.0338002741 0.0404599197 0.0393898487 0.0213730112 0.0290476047 0.0204971 0.0170970522 0.0352026969 0.0280033518 0.0246339682 0.0179246813 0.024900984 0.0201284532 0.0341629796 0.0184510238 0.0236822665 0.020255696 0.032124266 0.0167226251 0.0314883962 0.0409532972 0.0184396803 0.0319119804 0.0198584665 0.024986431 0.0240503177 0 0 0]
   [0.0350036472 0.039633397 0.0189372 0.0264758617 0.0339373462 0.0196347144 0.0165990088 0.0197543688 0.0358408 0.0269342754 0.0231907256 0.0320633613 0.0396633968 0.0350946374 0.0177162662 0.0157237109 0.0183534659 0.0169942547 0.039753 0.0239442959 0.0345261507 0.0181703605 0.0220945552 0.0343488976 0.0174325015 0.0197210182 0.0185150485 0.026615316 0.0198918972 0.0227057319 0.0255890917 0.0308192428 0.0316678882 0.0289786756 0.0290602837 0.0369938761 0.020057559 0.0275641419 0 0 0]
   [0.029276114 0.0270095263 0.038223777 0.0267280228 0.0287578497 0.0163270701 0.0368873142 0.0154394312 0.0397670977 0.0188125353 0.0362867415 0.0264006518 0.0202055182 0.0184341893 0.0217012912 0.032462921 0.0177052468 0.0169818792 0.0199582614 0.0360202342 0.0278221108 0.0152731799 0.0350727327 0.0236853771 0.0336875394 0.0187962446 0.0390781499 0.0292517859 0.0294597317 0.0230552759 0.027471967 0.0310200267 0.0166278258 0.0175510403 0.0341320224 0.0209010728 0.0323285237 0.0213998128 0 0 0]
   [0.0195483714 0.0264899712 0.0233827624 0.0228392743 0.0392039 0.0296137836 0.0329894535 0.0216856655 0.0243624318 0.0348837376 0.031240439 0.0382266119 0.0184731428 0.0271482989 0.0168596897 0.023581773 0.0229891557 0.0183197632 0.029126469 0.0302741304 0.020781599 0.0328724459 0.0315567963 0.0189571101 0.0186275188 0.0322166495 0.0313253552 0.0284154508 0.0238037128 0.0163489915 0.0279957671 0.0188713968 0.0319494493 0.0259448867 0.0216885488 0.0290686972 0.0395487472 0.018788062 0 0 0]
   [0.0182226859 0.0165483505 0.0319390297 0.0267190132 0.0238535535 0.0283161011 0.0190419257 0.0184378829 0.0211410522 0.0338321552 0.0203577392 0.0319940709 0.021932533 0.0172495302 0.0281738043 0.0330619514 0.0342819095 0.0387020595 0.0173039306 0.0380470082 0.0159333367 0.0183769409 0.0384092629 0.0224232301 0.0292726085 0.032972239 0.0211240817 0.0315159447 0.0334568024 0.0221038852 0.0339347869 0.0263079628 0.017337719 0.0319967233 0.0334472544 0.0187255722 0.0264537409 0.0270516556 0 0 0]
   [0.0260888375 0.0234035924 0.0347787924 0.0341469161 0.0165198334 0.0199899413 0.0164267197 0.0220998898 0.0322710387 0.0189588852 0.0388230346 0.0245875064 0.015759835 0.0386874676 0.0157757234 0.0167578049 0.0257861596 0.0308783241 0.0192156211 0.0336659439 0.0348983407 0.0205586366 0.0306548029 0.0184649527 0.0270532016 0.0398120619 0.0206078887 0.0220488403 0.0282481369 0.0257482268 0.0254508313 0.0346880741 0.0195808578 0.0222995486 0.0359088518 0.0248903744 0.0281213764 0.0363430791 0 0 0]
   [0.0183985345 0.0162383988 0.0351682752 0.0239150934 0.0273699742 0.0163017884 0.0293462444 0.0314370729 0.0224768426 0.0259136055 0.0184606202 0.0158090964 0.0237308443 0.0250641275 0.0387893505 0.0378231145 0.0293446183 0.0214420799 0.038153924 0.0337904952 0.0421036 0.017513277 0.0163412355 0.0320943817 0.0160604045 0.0228586234 0.0218952186 0.0281208269 0.0288121309 0.0365052633 0.0258268211 0.0209600944 0.0275444873 0.0179262813 0.03923136 0.0184574947 0.0335797742 0.025194576 0 0 0]
   [0.0411142632 0.0226403512 0.0409425683 0.0324031077 0.0199415162 0.0258578192 0.0322867855 0.0278945621 0.0319544077 0.0282851756 0.0286404099 0.0313433893 0.0371871665 0.0176354535 0.0180282872 0.021844741 0.0201849975 0.0348535217 0.0183277708 0.029633237 0.035623204 0.0236065052 0.0182960629 0.0352535918 0.0165055171 0.0335543379 0.0257485788 0.0184639059 0.0296190642 0.0160084292 0.0163054634 0.0256506372 0.0198872536 0.0245162863 0.025529854 0.0200530477 0.0331521779 0.021226557 0 0 0]
   [0.0304926541 0.0170509983 0.0413552336 0.0166435391 0.0216902215 0.0207323581 0.0199234467 0.0164883435 0.0301487036 0.0319566317 0.020908514 0.0349150188 0.0240989868 0.0158147439 0.0230790041 0.0235611815 0.0248641949 0.0195681918 0.0221721549 0.0157338325 0.0291195028 0.0382154 0.0409649275 0.0278792307 0.0383482315 0.0168337729 0.0184136797 0.0391598418 0.0319901071 0.0199753493 0.0227055233 0.037259277 0.0272839759 0.0279987194 0.0252082404 0.0292457398 0.028707074 0.0294934846 0 0 0]
   [0.0174536891 0.0259611793 0.0406491123 0.0314458683 0.0318696909 0.0180173982 0.0216167 0.019679781 0.0314191245 0.0254739467 0.0257628318 0.0255689584 0.0408444963 0.0272323713 0.0231345128 0.0328889191 0.0277459342 0.0187832303 0.0200484972 0.0206194427 0.0286164191 0.0223034527 0.041506514 0.0158622954 0.0227490794 0.0194164272 0.0181705821 0.0375978909 0.0294921175 0.0250332821 0.0270697568 0.0200417172 0.0406525284 0.0185364038 0.0248426534 0.0256543048 0.0185868572 0.037652 0 0 0]
   [0.03718514 0.0232267808 0.0164161678 0.0369700082 0.0317636579 0.0380381979 0.0354743153 0.035373 0.015248091 0.0335746035 0.0163780432 0.023532249 0.0257724896 0.0366726331 0.0381596 0.0186600517 0.0213452447 0.01538077 0.020965727 0.0191652738 0.0338852406 0.0267237891 0.0295755602 0.0273641627 0.0153624788 0.0171732623 0.0384192802 0.0161305424 0.0216567088 0.0168360826 0.0261683818 0.0201126486 0.0242095292 0.0168150384 0.0401500575 0.0271450412 0.0334996916 0.0294704325 0 0 0]
   [0.0263083819 0.0291438 0.0384631157 0.0147230504 0.0341891088 0.0185546428 0.0212284382 0.0193778891 0.0222982857 0.0252946038 0.0181998182 0.0352099761 0.0226979721 0.0211396534 0.0167196151 0.0350427404 0.0208927933 0.0217125397 0.0254383106 0.0384477451 0.0145532321 0.0337173119 0.0258979 0.0314828493 0.0383328795 0.0262286551 0.0315163769 0.0330415443 0.0368296951 0.0383848697 0.0220322832 0.0144133596 0.0287207197 0.0255055632 0.0153101636 0.0214505419 0.0289489049 0.0285506807 0 0 0]
   [0.0195678845 0.0193869174 0.0238005873 0.0226055309 0.0280634016 0.0295544192 0.0293551609 0.039835725 0.024584176 0.0309954882 0.0167565402 0.0372552909 0.0409655496 0.0318057388 0.0260380786 0.0264776144 0.0267810114 0.036714863 0.0245193448 0.0196381509 0.0225001071 0.0192188788 0.018539099 0.0185985882 0.0273240209 0.0289895404 0.0343960524 0.0227452144 0.0207299888 0.0239102636 0.0300321933 0.0408511 0.0275098644 0.0162147228 0.038714204 0.0198346321 0.0180529971 0.0171370637 0 0 0]
   [0.019845942 0.0190733522 0.026185073 0.0165186357 0.0364643522 0.018745983 0.0209798422 0.0297088195 0.0159422811 0.0393261723 0.0338966362 0.0264137704 0.0282395128 0.0377171785 0.0241057146 0.0262492299 0.0370552577 0.0351796076 0.0231127217 0.0194039512 0.0166086741 0.0256054755 0.0209924374 0.0160207804 0.0285430849 0.0175930727 0.0336935632 0.0187179241 0.0216702186 0.0271805245 0.021815598 0.0306426361 0.018669432 0.0241426 0.0295381621 0.0410046391 0.0378432088 0.0355539545 0 0 0]
   [0.0349714272 0.023436984 0.0169800837 0.0257725529 0.0156781469 0.0229979958 0.0209957939 0.0396271162 0.0340615921 0.0164264012 0.0251064338 0.0304725319 0.0246214885 0.0408081226 0.0250568688 0.0380816609 0.0345961936 0.0406520143 0.0275162123 0.0281942524 0.0228397474 0.0358150639 0.0330924 0.0158657543 0.018763341 0.0170996524 0.0300712362 0.0172638018 0.0265725609 0.0366703942 0.026199976 0.0191630889 0.0205866434 0.0175266396 0.0378765799 0.018264709 0.024895601 0.0153788896 0 0 0]
   [0.0213055033 0.0294810291 0.0329778083 0.0242488738 0.0188766103 0.0396284312 0.0172472689 0.0188896954 0.0366811 0.0200355258 0.0237482265 0.0407192782 0.017516626 0.0195305794 0.0169084799 0.0310515445 0.0249541271 0.023860367 0.021052815 0.0157637112 0.0241106153 0.0278896037 0.0283741318 0.0256525893 0.0211916901 0.033098463 0.027012866 0.0399555713 0.0183088258 0.0396940522 0.0312575102 0.0329191536 0.0158262812 0.0216948465 0.020961877 0.0372808129 0.0247479938 0.0355455093 0 0 0]
   [0.022524761 0.0148771098 0.0276993569 0.0275602546 0.0216338262 0.0357285179 0.0335615166 0.0358586237 0.0385254845 0.0293831769 0.0179789755 0.0364155881 0.0166024063 0.0221638344 0.0315655 0.0156475157 0.0362513 0.0312296972 0.0264068712 0.0222449712 0.0198147614 0.0318057463 0.0168968458 0.0327791087 0.0304731112 0.036618039 0.0253815819 0.0189835522 0.030849617 0.0309448801 0.0201394297 0.0255843904 0.0181094762 0.0207265466 0.0156244216 0.0237294156 0.0234063044 0.0342735499 0 0 0]
   [0.0309452657 0.0305929389 0.0192012619 0.039185904 0.0177683886 0.0168554671 0.0193710458 0.0354951657 0.0201829597 0.040625751 0.0253525283 0.0287339855 0.0244910158 0.0258233156 0.030999722 0.0407328904 0.0161124058 0.0404941812 0.0189190879 0.0157418884 0.0252391193 0.0317903049 0.0201205611 0.0285101794 0.0174410362 0.0289549176 0.0305941217 0.0393383019 0.0231712405 0.0188457333 0.0172964111 0.0214127209 0.017437337 0.0295974016 0.0238933098 0.0342005603 0.0288191792 0.0257123876 0 0 0]
   [0.0337672234 0.0252775941 0.0279579647 0.0342594162 0.0281836893 0.0383865498 0.0162614062 0.0289042704 0.0184950754 0.0168210249 0.0421065129 0.0403971672 0.0249713175 0.0409708396 0.0198004041 0.0169163421 0.0388651602 0.0205197688 0.0214155335 0.0371580496 0.0188135449 0.0167753864 0.0267156344 0.0219395943 0.0212481022 0.0294668209 0.0256991088 0.01609632 0.0389090925 0.0255540125 0.0195307862 0.0221576169 0.0299629513 0.0185872205 0.0259031337 0.0159060452 0.0293388143 0.02596041 0 0 0]
   [0.0150044914 0.0309032798 0.0358270854 0.0183165874 0.0153580792 0.0154790878 0.0348164365 0.0165357795 0.0233414192 0.0251811408 0.0154480534 0.0405817479 0.035328351 0.024041906 0.0160995983 0.0161809027 0.0329521485 0.0375328846 0.0376637392 0.0387888774 0.0179164074 0.0186235607 0.0203637127 0.0281113368 0.0337021686 0.0383804888 0.0359451286 0.0332800895 0.0168031938 0.0335336104 0.0231914967 0.0240279939 0.0279999394 0.0165071171 0.0342474 0.0250453148 0.0206449814 0.0262945052 0 0 0]]

  [[0.0147417355 0.0245160796 0.0246034898 0.0177675299 0.0301432814 0.0206936095 0.0255105905 0.0376066193 0.0383572318 0.0359017588 0.0380247086 0.0318907239 0.0344650112 0.0177632477 0.0223369524 0.0388664827 0.0363673791 0.0217791721 0.0212059692 0.0145955645 0.0254217926 0.0327118747 0.0166659169 0.0145358276 0.0183608886 0.0346943811 0.0210482907 0.0217535794 0.0203323681 0.033622086 0.0384707078 0.0255059395 0.0197046809 0.0379051939 0.0175979845 0.0313390046 0.0226824246 0.020509949 0 0 0]
   [0.0282315761 0.0331846774 0.02274625 0.0246169586 0.0258468669 0.0215877593 0.0419783182 0.0210969299 0.0169114415 0.0236744825 0.0173018444 0.0212236606 0.0251916368 0.0267068371 0.0256581977 0.0407551564 0.0218211915 0.0397252254 0.0206633881 0.0194396116 0.0378592834 0.0290377848 0.0181440841 0.0262330752 0.016052682 0.0165716447 0.0300977416 0.0327610709 0.0324666351 0.0176717341 0.0393674038 0.0213557351 0.0249059703 0.0246667974 0.030901514 0.0189806633 0.0374953039 0.0270688813 0 0 0]
   [0.0271133892 0.0387195833 0.0237776767 0.0338866 0.018293187 0.0208670404 0.0231044721 0.0366937295 0.0243849754 0.0160754528 0.0357180461 0.0362282172 0.0219384171 0.0272445586 0.0285388436 0.0311384 0.0263024643 0.022811396 0.0294557568 0.0204973239 0.025630163 0.0209961943 0.0190275554 0.0284095146 0.0186222475 0.0361186825 0.0156000108 0.025708748 0.0342901163 0.0361372083 0.0343502834 0.0301238541 0.0262146052 0.0181582551 0.0162400752 0.0259760693 0.0190025214 0.0266043656 0 0 0]
   [0.0380991064 0.0161291659 0.0310312472 0.0149229905 0.0323165357 0.0319352336 0.0175061394 0.033043012 0.0337593295 0.0310976915 0.0306826923 0.0159293488 0.0220197756 0.0252774768 0.025833698 0.015265285 0.0328555033 0.0169388056 0.0364388712 0.0236213654 0.0382714719 0.0337510295 0.0278842673 0.0346386842 0.0148946606 0.0251217168 0.015000795 0.0165839549 0.0382592604 0.0300063435 0.0167956017 0.0296698287 0.0335130543 0.0377900079 0.0355456397 0.0155319814 0.0145757338 0.0174627025 0 0 0]
   [0.0238415916 0.0227340553 0.0408628806 0.0181624182 0.0342642739 0.0157449767 0.0282714497 0.0203752518 0.0179043841 0.0403563865 0.0376184545 0.0238102209 0.0161552373 0.0392271839 0.026517557 0.0175202172 0.0248175059 0.0372087508 0.0185091812 0.0294352267 0.020236386 0.0294953492 0.0215600282 0.0204790011 0.023461068 0.0369984806 0.0377920382 0.0194715858 0.0324926 0.0175794493 0.0212685298 0.0350156799 0.019430995 0.0182068646 0.023335129 0.034378171 0.0174515769 0.0380099192 0 0 0]
   [0.0287042223 0.0265313163 0.0168029647 0.0282714125 0.0160145741 0.0267977118 0.0390997417 0.0202367548 0.0229202453 0.0384579 0.0282850768 0.0204776358 0.0184636321 0.0183604024 0.0394378938 0.0173943844 0.0322346948 0.0373166688 0.0354698412 0.0355929881 0.0159306433 0.0181192756 0.0164105408 0.0277057327 0.0323085785 0.0351843238 0.0182575677 0.022163745 0.0189493131 0.0353498794 0.0156340282 0.036554873 0.0383585393 0.0329877548 0.016323233 0.0229670946 0.0239410046 0.0259838197 0 0 0]
   [0.0240784362 0.0161466375 0.0238786526 0.0254881959 0.0316162035 0.0323531292 0.0321493037 0.0395731367 0.0322351642 0.0264397394 0.0186380465 0.0377483964 0.0308347344 0.0243767407 0.0313836373 0.0160317291 0.0202931743 0.0161091 0.0289628636 0.0379268 0.0186867472 0.0290274583 0.031733565 0.0229286086 0.0232983902 0.0260095056 0.0414474234 0.0179833714 0.0377287269 0.0176333152 0.0209629945 0.0413919315 0.0203276519 0.0186247658 0.0326649025 0.0165263545 0.0193843 0.0173762105 0 0 0]
   [0.0307998508 0.0207715034 0.0335514396 0.0283401534 0.0166608654 0.0193688832 0.0196443498 0.0196476 0.0170449112 0.0305506773 0.0207919143 0.0227184594 0.0269058365 0.016875308 0.0192762222 0.0374607071 0.0403586477 0.0270662885 0.0373346843 0.0163350422 0.022746738 0.0211698562 0.0170544386 0.0213974323 0.0414380319 0.0352343582 0.0245112441 0.0164112207 0.0355736762 0.0192342084 0.0236027502 0.0408136062 0.0349422581 0.0398248658 0.0297801811 0.0220275335 0.0221184157 0.0306157861 0 0 0]
   [0.0161537752 0.016774293 0.0255625043 0.0212681983 0.0261615403 0.0253250636 0.0327023081 0.0214836523 0.0325270295 0.0208257195 0.0312984809 0.0302869119 0.0183087345 0.0344489478 0.0178941023 0.0214856379 0.0163225718 0.0188068915 0.0239385162 0.0314132161 0.015686404 0.035816513 0.0170436073 0.0262295678 0.0379134268 0.0219510123 0.0239666812 0.0392847136 0.0217726231 0.0397151895 0.0288182925 0.0349664614 0.0178117827 0.0333557166 0.0280323718 0.0377422273 0.0369434506 0.0199619159 0 0 0]
   [0.0251429472 0.0201010499 0.0233329833 0.0192458425 0.0162949581 0.042183537 0.0189782698 0.0293192752 0.0312545486 0.0175273344 0.0218818057 0.0163191799 0.0342632234 0.0363247506 0.0206850413 0.0173533279 0.0379215628 0.0286468621 0.0280687809 0.0337844715 0.0277757645 0.034723632 0.0205564108 0.0160773322 0.042162016 0.0216351971 0.0389840193 0.0260883756 0.0208524931 0.0347350352 0.0303476527 0.0164304655 0.0163943 0.0265162401 0.0180770811 0.026916245 0.0359401032 0.0271578785 0 0 0]
   [0.0243584383 0.0354573429 0.0255137235 0.0311455466 0.0167959258 0.023741886 0.0382926799 0.0291034542 0.0292329323 0.0205758158 0.0247679949 0.0147275822 0.0308707152 0.0178446639 0.0263667814 0.0158954151 0.0164153576 0.0143541973 0.0164906587 0.0378573202 0.0369771793 0.0246127564 0.037669044 0.0298879817 0.0298286863 0.0250345357 0.0254430976 0.0244552214 0.0353427194 0.0313165374 0.0230010673 0.0306066 0.0286244359 0.0165733434 0.0338680595 0.0249446929 0.0233965144 0.0286090318 0 0 0]
   [0.0230783224 0.0394624323 0.0266512651 0.0248239115 0.0225344598 0.0166331436 0.0202399082 0.028050283 0.0162711758 0.0344511233 0.0247153603 0.0417725332 0.0301470365 0.0327048935 0.018662272 0.0176207162 0.032479275 0.0377643742 0.0220324956 0.0295651853 0.0384361111 0.0284929909 0.0199780893 0.0228708684 0.0222308356 0.026753746 0.0360688716 0.0250846762 0.0278605614 0.0227345079 0.0189709 0.0205424652 0.0160838 0.0179764647 0.0336234719 0.0229911506 0.0219292399 0.037711028 0 0 0]
   [0.02813063 0.0269888248 0.0259372164 0.0169035494 0.0300134066 0.0314014331 0.0200018734 0.0338358618 0.0254208837 0.0186012518 0.0299435202 0.0410976522 0.0410880186 0.0232452042 0.0154452445 0.0262484495 0.0313987695 0.0345913 0.0305239055 0.0186344944 0.0239331722 0.0157106426 0.0194119737 0.0209606942 0.0230335202 0.0317773409 0.026556015 0.0169390831 0.0287529025 0.0230441224 0.0396720767 0.0155248484 0.0269836597 0.0214744322 0.0188857522 0.0377422 0.0414534099 0.0186926797 0 0 0]
   [0.0159732252 0.0186596718 0.0351580717 0.0222312547 0.0298168194 0.0274883416 0.0240447223 0.039029967 0.0360425748 0.0175590161 0.0292613506 0.0217843298 0.0267683025 0.0181973279 0.0393779352 0.0195593424 0.02564051 0.0194897335 0.0398802795 0.0202917065 0.0161372051 0.0316293128 0.0233777221 0.033383511 0.01643827 0.022052642 0.0218325369 0.0331949 0.0413241461 0.0215987246 0.0222504549 0.0163242817 0.0324749127 0.018248396 0.0373542383 0.0181104913 0.0316868685 0.0363268629 0 0 0]
   [0.0253696851 0.0324850045 0.0378627442 0.0183931421 0.040467672 0.0353550203 0.0290020593 0.0181445982 0.0244636964 0.0285419393 0.0250234492 0.0339632966 0.0236061439 0.0290656518 0.0275601894 0.0282623824 0.0223054346 0.0187015329 0.0348550715 0.0398514718 0.0311676171 0.015562783 0.02805808 0.0293006636 0.0217749327 0.0162725206 0.015273558 0.0280714501 0.0152807729 0.0311473329 0.021022588 0.0203263853 0.0158635136 0.0393857844 0.0294765923 0.0167167448 0.0184563585 0.0335621312 0 0 0]
   [0.0301102754 0.0310163684 0.02686668 0.0324913077 0.0156112155 0.0276256613 0.0361076929 0.0247127451 0.0275117792 0.0263593663 0.0366863087 0.0164363757 0.0163689 0.0282367896 0.0403182767 0.0216066819 0.0226155762 0.0170299392 0.0185257569 0.0162106138 0.0390388295 0.0242693983 0.0304036271 0.0278235693 0.025760714 0.021537561 0.0401070528 0.0178465284 0.0202861037 0.0381406806 0.0318229869 0.0391190648 0.0259193219 0.0206975564 0.0228467919 0.0259061959 0.0178912766 0.0181343537 0 0 0]
   [0.0286397394 0.0398536958 0.0262641832 0.0280830879 0.019905217 0.0363590904 0.0306396764 0.02069127 0.0287229847 0.021115642 0.0192663409 0.0305687506 0.0331928693 0.037873134 0.0227864143 0.0288537256 0.0189730916 0.0243954342 0.0211663693 0.0193958636 0.031512063 0.0406853 0.0244539212 0.0317347273 0.0162077099 0.0235098228 0.0204289146 0.0166060776 0.0167839061 0.0174040142 0.015767578 0.0177218132 0.0249515362 0.0410291366 0.0368835703 0.0299693532 0.0180363953 0.0395675562 0 0 0]
   [0.0171020348 0.0256513394 0.0343837626 0.0257825684 0.024728721 0.0367925838 0.0218478777 0.017935982 0.0241176169 0.0367898643 0.0284639299 0.0198128242 0.0356910974 0.0237807203 0.0285765678 0.0241063926 0.0228896383 0.0370659605 0.0279631149 0.0232312717 0.023785105 0.0299387313 0.0288047493 0.0190850366 0.018182287 0.0400611833 0.0287650116 0.0233365409 0.0308354869 0.0194293801 0.0361565202 0.0189549141 0.0278435219 0.0228665173 0.021563245 0.0337317847 0.0170466695 0.0228994433 0 0 0]
   [0.0189194679 0.015544055 0.0189013034 0.0315635279 0.0382619463 0.0351327918 0.0356534608 0.0315948799 0.0314749666 0.0267686956 0.033031527 0.0219233409 0.0149329538 0.0271445382 0.0212088656 0.0310185608 0.0299200714 0.0152082816 0.0368467793 0.0153659312 0.0335435458 0.0317343809 0.0374696627 0.0304155611 0.0346566327 0.0188843496 0.0300694462 0.0149457306 0.0170680583 0.0197712984 0.0236464087 0.0363425799 0.0258664284 0.0221172702 0.020151075 0.0147946235 0.0260888245 0.0320181884 0 0 0]
   [0.0165782738 0.0162582491 0.0184099842 0.0401013494 0.0159861539 0.0347210579 0.0173158292 0.0344106816 0.0326017626 0.0272196829 0.0218944978 0.0354757607 0.0217684 0.0188323855 0.0304228161 0.0326176398 0.0301258937 0.0271038394 0.033076033 0.020781368 0.0257286988 0.0383707918 0.0339403041 0.0264813211 0.0247328598 0.0261452254 0.0217890255 0.0430425517 0.0169126391 0.0183066409 0.0252233092 0.0312036648 0.0185251795 0.0181051716 0.0165760554 0.0187926609 0.0372594558 0.0331627615 0 0 0]
   [0.0189157128 0.019622298 0.0246911552 0.0292160194 0.01632265 0.0189643353 0.0250050947 0.0298954975 0.0280947387 0.0331254676 0.0356005952 0.0232756604 0.0311229564 0.0367067903 0.0270190164 0.0270874668 0.0233531836 0.0217830259 0.0208734497 0.0350995399 0.0155922566 0.0167673808 0.0410314053 0.030529134 0.0282223206 0.0213718116 0.0193481743 0.0397498757 0.0209677368 0.0391249396 0.0209700204 0.0213207901 0.0257151667 0.0194535386 0.0273619499 0.0324278362 0.0305089187 0.0237621292 0 0 0]
   [0.0342148 0.022732595 0.0372958742 0.0301101897 0.0238717366 0.0298386961 0.0188523121 0.0260809176 0.0372851118 0.0211622324 0.0182063486 0.0238908883 0.0239765123 0.0343197398 0.026842583 0.0145724509 0.0254820827 0.0272177141 0.0301031377 0.0346212909 0.0293343291 0.0254015904 0.0371825956 0.0196920484 0.0335318781 0.0178325828 0.0244969875 0.0382818207 0.018857494 0.033857774 0.0219495334 0.0177016519 0.0196293611 0.020410439 0.0230188854 0.0269072838 0.031866841 0.0193696693 0 0 0]
   [0.026180923 0.0345735848 0.0158725232 0.0238952618 0.0216408633 0.0325776897 0.0262140781 0.0156747531 0.0212898944 0.0333322249 0.0365654491 0.0361081362 0.0280148033 0.019607516 0.0231557898 0.0329198651 0.0370933451 0.0411921255 0.0191865545 0.0205888562 0.0231775194 0.0347939 0.0264111925 0.0175799038 0.0367093906 0.0162415672 0.0191512704 0.0239111893 0.0234200303 0.0265414231 0.038072288 0.0181803815 0.0188468751 0.0326533653 0.0406850874 0.0208064616 0.0158799905 0.0212539248 0 0 0]
   [0.0190096609 0.0282509811 0.0399257727 0.0303641204 0.020289978 0.0153943989 0.0374232903 0.0349343605 0.0152044604 0.0351402834 0.0176844578 0.0190063938 0.0213490538 0.0217024609 0.0172137711 0.0371026881 0.0296530183 0.0194451753 0.0386941209 0.0240042116 0.025090972 0.022080658 0.0177966729 0.0275844857 0.0291822404 0.023867866 0.0302429441 0.0321855098 0.0190779492 0.0185345095 0.0342386626 0.0268041883 0.0338066258 0.0357261188 0.0205180272 0.0186814331 0.0366808474 0.0261076353 0 0 0]
   [0.029703958 0.0281916521 0.0289021786 0.0231253896 0.0234449562 0.0345327705 0.0276241973 0.0159771163 0.0370750576 0.0347033516 0.0151592065 0.0328824483 0.0399332196 0.0302416347 0.0272344723 0.0305721853 0.0163187571 0.0228320695 0.0221714489 0.0272801258 0.0349527113 0.0163047016 0.0358425528 0.0314994864 0.0204505678 0.0166543964 0.0377164781 0.01552861 0.017428441 0.0204614848 0.0242370106 0.0376697034 0.0247638486 0.0201019906 0.0221906863 0.0285152942 0.0317499563 0.0160258487 0 0 0]
   [0.0394627675 0.0270791519 0.0169684347 0.0407197401 0.0225192048 0.0275792442 0.0243652258 0.0180522036 0.0212885793 0.0416235588 0.036162816 0.0406225733 0.0160027333 0.0196009893 0.0166096333 0.0160219613 0.0210712422 0.0291707944 0.0169224013 0.026363967 0.0293320231 0.0266529899 0.0226651654 0.0201056823 0.0395420268 0.0395053141 0.017984733 0.0203715824 0.036385756 0.0277885757 0.0296055786 0.0260717459 0.0306396261 0.0164616853 0.0296813436 0.0286854245 0.0198489223 0.0204646382 0 0 0]
   [0.0345487036 0.0396917164 0.0150840944 0.0283094328 0.0275602 0.0323521979 0.0201208238 0.028095033 0.0293371379 0.023247797 0.0323056504 0.0245299805 0.0184670109 0.0152132139 0.0287298691 0.0156541429 0.0222817417 0.0187464096 0.0267527755 0.0151618551 0.0212704837 0.0190058835 0.0396787 0.0218533818 0.0173782501 0.0363776162 0.0172375292 0.0364312828 0.0283672959 0.0243676212 0.0269180574 0.0337430052 0.033880908 0.0321008898 0.0236857608 0.0251854882 0.0392180942 0.027109934 0 0 0]
   [0.0174102969 0.0197247602 0.0249753185 0.0175279155 0.0338215828 0.0261616912 0.0248271246 0.0168131087 0.029301744 0.0337948278 0.0161341671 0.0273826607 0.0337154679 0.0312969722 0.01939377 0.0359981172 0.0241709705 0.0403232947 0.0306939706 0.0181829948 0.0171759091 0.0290486906 0.0371460356 0.0177466627 0.0196071025 0.0160531327 0.0277129225 0.0385355 0.040383 0.0208036602 0.0410757698 0.0295551401 0.0366062187 0.021245284 0.0206979383 0.0158922337 0.0299929902 0.0190710891 0 0 0]
   [0.0223771017 0.0201634392 0.0246851407 0.018302178 0.0172592625 0.0281972475 0.0170015022 0.0238839202 0.0186997596 0.0223474279 0.0399575531 0.0212743599 0.0193691403 0.02322313 0.0222733524 0.0393104777 0.0391538888 0.022980297 0.0348934345 0.0396423489 0.017900059 0.0266231541 0.0188177433 0.0223190207 0.0272744019 0.0336750224 0.0292687193 0.0248890482 0.026123587 0.0172746237 0.0256206859 0.0169485081 0.040871311 0.0377417505 0.0218302906 0.036822442 0.0236601718 0.0373444296 0 0 0]
   [0.0226116814 0.0184668042 0.0159354769 0.0203504451 0.0322122686 0.0281672366 0.017588038 0.031083649 0.0158474762 0.0344422534 0.033484064 0.0342234895 0.0171483532 0.0191505682 0.0222938955 0.0229136683 0.0283376779 0.0419000238 0.0249258708 0.0215111542 0.0232788 0.0262983218 0.0307304803 0.0328412354 0.0183865093 0.0237187222 0.0413620099 0.029395774 0.0165007915 0.0229421239 0.0223136544 0.0390496887 0.027397858 0.0210655406 0.0265836176 0.0308933631 0.0371191725 0.0275282189 0 0 0]
   [0.0397338793 0.0333353803 0.0210366882 0.0191588327 0.0162756667 0.030524198 0.0335772336 0.0248726569 0.0335468948 0.0235964376 0.0172181539 0.0204826035 0.0164222941 0.0348618403 0.0180875212 0.0360295847 0.0213900246 0.0390808657 0.0340238065 0.0288778991 0.0326593518 0.015136539 0.0154252099 0.0206166711 0.0224736966 0.0320303142 0.0329292305 0.0231586117 0.023181878 0.0351790041 0.0314921588 0.0395263508 0.0326609239 0.016460374 0.0158893 0.0167774465 0.0267175324 0.025552962 0 0 0]
   [0.0299014729 0.0157133695 0.0271115 0.038535431 0.0228242856 0.0184623431 0.0228012092 0.0155561082 0.0222351328 0.0230934042 0.0327477865 0.0206364114 0.0313345939 0.0385233872 0.0329819731 0.0184170678 0.0243143719 0.0279832371 0.0254037697 0.0152742742 0.0398344 0.0358877331 0.0190078728 0.0327485614 0.0153165059 0.0222161859 0.0191984922 0.0396979712 0.0277374852 0.0380723476 0.0271109957 0.0156744383 0.0327164158 0.017119376 0.0192090273 0.0187365469 0.0402514 0.0356130861 0 0 0]
   [0.0291160699 0.0362198874 0.0376728438 0.0303862654 0.0225099735 0.0300286226 0.0290581733 0.0262543652 0.0257700272 0.0267008748 0.0183779988 0.0368275493 0.0208811108 0.0301458985 0.0217724182 0.0406425223 0.0181318875 0.0412614979 0.0221024081 0.0352697149 0.0180669744 0.0156626925 0.0278904233 0.0229505096 0.015409125 0.0355112925 0.0194823164 0.0313160568 0.0184708536 0.0186892562 0.0172178615 0.0371809155 0.0157231893 0.0326556265 0.0309452452 0.0252161883 0.0189833101 0.0194979887 0 0 0]
   [0.0270517059 0.0192499869 0.0233703777 0.0266576074 0.0224910434 0.0192360319 0.0273179282 0.0381571911 0.0200817026 0.0251236595 0.025365252 0.042359 0.0202780943 0.0283126663 0.0171956643 0.0191292129 0.0275206864 0.034563724 0.0308295097 0.0168364104 0.0316632725 0.0274256393 0.0213992894 0.0216269512 0.0206275806 0.0423367359 0.019707337 0.0206667297 0.0204159506 0.0184451602 0.0205383729 0.0258735884 0.0295794215 0.0356339552 0.0300827958 0.040496815 0.0271320958 0.035220854 0 0 0]
   [0.0309879184 0.0205988083 0.0180455204 0.0372696258 0.0169751104 0.0245521609 0.0353911035 0.0335623957 0.016426865 0.0304936413 0.0163642261 0.0167690758 0.0162856895 0.0276231989 0.0276008528 0.0233506802 0.0279938281 0.0309167821 0.0214980673 0.0371651091 0.0391763449 0.0322558843 0.0221891087 0.0370943472 0.0246974733 0.0203521 0.0171069857 0.0228534322 0.0172111206 0.0318338647 0.0198699571 0.0222485643 0.0403736047 0.0234924518 0.0269250553 0.0329756364 0.0358872749 0.0235861205 0 0 0]
   [0.035951633 0.020080829 0.0154533358 0.0344342627 0.0207149293 0.0206875149 0.0140191931 0.0235805865 0.0147478478 0.0214348547 0.0193540342 0.0141575839 0.0232239328 0.0193657186 0.0285461675 0.0158181805 0.0357745737 0.0332214572 0.0376003273 0.0322325304 0.0316548981 0.0298271123 0.034006048 0.0259680673 0.0276639443 0.0269407388 0.0208858717 0.0327446088 0.0367863029 0.034500353 0.0239600204 0.0160934515 0.0356538668 0.0369618833 0.0225581732 0.0326828584 0.022857815 0.0278543942 0 0 0]
   [0.0213027857 0.0295919701 0.0255520176 0.0188143086 0.018914653 0.0206024405 0.0206422061 0.0368242785 0.0170531962 0.0341272056 0.0370087922 0.0330510549 0.0370724201 0.040950682 0.0317861848 0.0164176654 0.040574681 0.0283154435 0.0180108491 0.0305636693 0.0189169757 0.0394871756 0.0155172711 0.0160078164 0.0338415727 0.0199163593 0.0176831894 0.0236648861 0.0229879338 0.0284341797 0.0412949733 0.0180972628 0.0395370163 0.0261245 0.0177535154 0.0177859757 0.019230403 0.0265425034 0 0 0]
   [0.0387474447 0.0277367383 0.0287121385 0.0321391784 0.0342434384 0.025285881 0.0392745137 0.025035888 0.0287024695 0.0291453302 0.0180111807 0.0299346242 0.0306246039 0.0242589694 0.0369934626 0.040241193 0.0227796305 0.0168402623 0.0301131122 0.0156936236 0.0367283113 0.0158028733 0.0167406965 0.0255017392 0.0218870286 0.0321939066 0.0221610907 0.0267001037 0.0301294737 0.0162329972 0.0179464277 0.0321742557 0.0157889128 0.0199913234 0.024072472 0.0343570374 0.0197171587 0.01736046 0 0 0]
   [0.0168724582 0.0236769 0.0255048983 0.0362856947 0.0267513823 0.0193736404 0.0188020039 0.0243728347 0.0196493436 0.0292051025 0.0219087247 0.0307250787 0.0337292738 0.035154473 0.0216441676 0.0175079871 0.0158855785 0.0223531798 0.0214207973 0.0178565755 0.0379702449 0.0297351964 0.0244983304 0.0390434377 0.0184371211 0.0398441143 0.0178298987 0.0243006889 0.0159720387 0.0261089373 0.0299045537 0.0250176229 0.0330194272 0.0377335586 0.0285892989 0.0251304246 0.0268890727 0.0412958935 0 0 0]
   [0.0173709933 0.0349140503 0.0372016691 0.0218371116 0.0286522489 0.0193852633 0.0188079253 0.0290459525 0.0376420058 0.0177314039 0.0420794301 0.0202573054 0.0172598213 0.0290153157 0.0207171328 0.0221806504 0.0338067934 0.0202166066 0.0346472152 0.0210984256 0.0236059763 0.0359189808 0.0255945865 0.0295714419 0.0412789956 0.0224256925 0.0241990983 0.0293524358 0.0361304507 0.0196635071 0.0213078372 0.0394860208 0.0165000539 0.0169378985 0.0195210148 0.0273418818 0.0303985737 0.0168982539 0 0 0]
   [0.0244671442 0.0195093304 0.0283712391 0.0282074381 0.0314641781 0.0298755467 0.0388514735 0.0245966408 0.0231086276 0.0192706622 0.0241524521 0.0375443362 0.0383182243 0.0381989963 0.0217895266 0.0230587479 0.0201252289 0.0302412715 0.0233976804 0.0184105132 0.0257408265 0.0295177549 0.0281074829 0.0170172229 0.0304392129 0.0151610961 0.0179027785 0.0264744665 0.0313753635 0.0205782242 0.0349390358 0.0230794847 0.0185866319 0.0259074736 0.0399156585 0.0186989568 0.0228330661 0.0307660773 0 0 0]]

  [[0.0335563459 0.0188089758 0.0167805236 0.0292792264 0.0271871611 0.0240334291 0.0163161103 0.0286772903 0.0392497517 0.0376077816 0.0401850641 0.0217442755 0.0368042253 0.0282827634 0.0225394629 0.0245826989 0.0244136304 0.0186665319 0.022679504 0.0347045027 0.025286641 0.0223530699 0.0255037025 0.0314577073 0.0177041758 0.015273626 0.023981357 0.0242343508 0.0374986455 0.0166011155 0.0378835127 0.0292428192 0.026094364 0.03262062 0.0200816989 0.0217615925 0.0274805203 0.0188413076 0 0 0]
   [0.0290700886 0.0256317034 0.0276743751 0.0184548348 0.0318140909 0.0234303 0.0185720902 0.0174435787 0.031351611 0.0164079908 0.0297121163 0.0250839945 0.0370657668 0.0288092606 0.024844138 0.0271317922 0.0224839728 0.0381181054 0.0304870885 0.0167121608 0.0404176451 0.0388182029 0.0262448564 0.0165986028 0.0188435353 0.0185381863 0.0261035021 0.024975596 0.0324109569 0.0301503576 0.0222331658 0.0326134898 0.0419858471 0.0195008125 0.0187407453 0.0314188637 0.0176574737 0.0224490948 0 0 0]
   [0.0314795338 0.0154874511 0.025552474 0.0263885912 0.0174555648 0.0212715771 0.0162764043 0.0185517743 0.0264726542 0.033995036 0.0303344168 0.033005245 0.0192115903 0.0385579765 0.0154061364 0.0245045759 0.0240317248 0.0293782353 0.0153099839 0.0352288298 0.0165075604 0.0287027135 0.02668548 0.0227578841 0.0287956782 0.0222296715 0.0397064164 0.0386817344 0.0259639509 0.0215111524 0.0250173379 0.019905787 0.0280445758 0.0170190446 0.031187946 0.0371480957 0.0377262458 0.0345089771 0 0 0]
   [0.0331390835 0.0245042406 0.0176707897 0.0263013 0.023685867 0.0299197305 0.0226116814 0.0266250297 0.0167280659 0.0327582844 0.0225792583 0.0328669213 0.0332647 0.0321217552 0.0390171148 0.0335870236 0.0294268131 0.0374104865 0.0212228 0.0193675458 0.020641651 0.0190348942 0.0281883292 0.0367667601 0.0170187652 0.029374741 0.0261777081 0.0243146811 0.0240034889 0.0171042755 0.035317 0.017870605 0.0191034805 0.0302430838 0.0181263778 0.0415108055 0.0223578084 0.0180370584 0 0 0]
   [0.0206953324 0.036000248 0.023483647 0.0253171641 0.0211091228 0.0274876077 0.0277460404 0.0409590639 0.0222850908 0.0413598828 0.0226994548 0.0203684159 0.0256276187 0.025521772 0.0171143133 0.0378117189 0.0264522545 0.0287238043 0.0309883431 0.0196580403 0.0181041863 0.0203601737 0.0259018894 0.0222713836 0.0414821245 0.0244217031 0.0253782347 0.0220617279 0.0292733144 0.03112006 0.0189612191 0.0297493376 0.0167014617 0.0210172068 0.0185920671 0.0410333276 0.0166264921 0.0355351456 0 0 0]
   [0.0272508375 0.0274852067 0.0265050158 0.0330580175 0.0194215532 0.0193755664 0.0344563 0.0239430182 0.019534843 0.0255119521 0.0372027718 0.0263589974 0.0256578885 0.0231301803 0.0175980199 0.0266778171 0.032065887 0.017617 0.0377961844 0.0363272838 0.0232285112 0.0208052751 0.0312278029 0.0187406484 0.0256110039 0.0161629952 0.0381796136 0.0241506379 0.0160145164 0.0289960802 0.0168758687 0.0196150802 0.0334751494 0.0277994461 0.0337843783 0.0247742888 0.0301427804 0.0334415883 0 0 0]
   [0.0351034626 0.0281026978 0.0215559378 0.039436046 0.0231502224 0.017807 0.0227073822 0.0340349413 0.0188928768 0.0195310824 0.0194446705 0.0254396312 0.0216182712 0.03311662 0.02276529 0.0317084789 0.0174874216 0.0271214861 0.0323444046 0.028568048 0.0189880375 0.0203622635 0.0384222604 0.0214180164 0.0340404846 0.0245764125 0.0157911014 0.0406783931 0.0205800757 0.0168578811 0.0343089178 0.0274986923 0.0250299405 0.0411451384 0.0333762802 0.0286795385 0.0210574511 0.0172530618 0 0 0]
   [0.0221792273 0.0208611544 0.0224494077 0.02021599 0.0404495522 0.0164693296 0.0300062075 0.0274067111 0.037369743 0.0213556588 0.0178195424 0.0177110769 0.0350655355 0.016319098 0.0358271 0.038333 0.0291137155 0.0278025847 0.0420585908 0.0282785725 0.0391207449 0.030757647 0.0180235151 0.0189207904 0.0290794075 0.0313986242 0.0206209365 0.0157878 0.0191566572 0.0310321115 0.0277711824 0.0212538373 0.0157982558 0.0227900278 0.0343917981 0.0257134698 0.0166421235 0.0346492678 0 0 0]
   [0.0353948474 0.0341338068 0.0167679731 0.0354175456 0.0270640757 0.0190408546 0.04150711 0.0345988683 0.0180254392 0.0169972908 0.0191825964 0.0179629531 0.028306758 0.0336368233 0.035861928 0.0350383 0.0260698162 0.0279680341 0.0313446037 0.0328005515 0.0295248497 0.0232003052 0.0185980871 0.0252962578 0.0165443365 0.0351287946 0.0248975232 0.023579644 0.0231382865 0.0198071226 0.0263464134 0.0164465588 0.0236613043 0.0315502025 0.0166382957 0.0362088308 0.024962103 0.0173509289 0 0 0]
   [0.0173149612 0.0296176486 0.0405616276 0.0402535722 0.015804071 0.0244063307 0.0338881873 0.018244274 0.0214423575 0.0408224463 0.0322675556 0.0239544306 0.0214971863 0.0304385535 0.0314064659 0.0166599564 0.0163589865 0.0277400166 0.0227564182 0.0162505116 0.0174720902 0.029756749 0.0255221017 0.0180259 0.0260977838 0.0311497413 0.0179177653 0.0348122939 0.0229485426 0.0273627732 0.0222680867 0.0397714451 0.0271692332 0.0299801789 0.0211218651 0.0330586322 0.0224458873 0.0314333253 0 0 0]
   [0.016484011 0.015735 0.0315083712 0.0203712024 0.0281504616 0.0273823403 0.0186363328 0.042029649 0.0163509697 0.0296499 0.0204954315 0.0182264578 0.041311238 0.0389649868 0.0298391487 0.0301847011 0.016265586 0.0344922654 0.0261940528 0.0296698827 0.0198022649 0.0164244678 0.0182088073 0.020009879 0.0282214768 0.0193227436 0.0206165221 0.0392094254 0.0405920371 0.0269864 0.0399012975 0.0195287541 0.0177268144 0.0278599802 0.0312177502 0.0168928597 0.0246795863 0.0408570319 0 0 0]
   [0.0248101875 0.0372636244 0.0297601391 0.0279138777 0.0282678474 0.0271794461 0.0219885707 0.0349857621 0.0168066025 0.0298781637 0.0288011413 0.0173326284 0.0196864661 0.0172600634 0.0292470101 0.017718561 0.0233437978 0.0211065859 0.0267078262 0.0303958096 0.0284677073 0.0368396081 0.0386926085 0.0274789277 0.0375928544 0.0237629954 0.02978676 0.0317980051 0.019975625 0.0204988103 0.0269841962 0.0244691316 0.0249993391 0.0152525762 0.0202562902 0.0230788123 0.0295599736 0.0300517287 0 0 0]
   [0.0207109638 0.020169137 0.0174521059 0.035141062 0.0254606828 0.0198329985 0.0175270718 0.0155115807 0.0284171049 0.0343389772 0.0324972235 0.0154714128 0.0373486094 0.0346476585 0.0227766167 0.0396697745 0.0215536151 0.0403244682 0.0189256407 0.0342213921 0.0392549485 0.0171896517 0.0178977977 0.0368350893 0.0246019624 0.0267532934 0.0306463763 0.0174329709 0.0238039158 0.0268360246 0.0169647504 0.0402323045 0.021272637 0.0182735845 0.0279397834 0.0239457153 0.0374556929 0.0206654873 0 0 0]
   [0.0194932502 0.0198893789 0.0214769579 0.0174394194 0.0206444245 0.018494742 0.0192307103 0.0251845792 0.0202327762 0.031636592 0.019813966 0.0347428359 0.0386792421 0.020354677 0.0326071791 0.0211614799 0.0160803758 0.0333825462 0.0266421847 0.0172999706 0.0347201638 0.0399379805 0.0180868693 0.0324061476 0.0247201677 0.0178449322 0.018769348 0.0166080445 0.0365297 0.0159338638 0.0370132476 0.027796654 0.0333370753 0.0395986065 0.0263315439 0.035185162 0.0291326866 0.0415604971 0 0 0]
   [0.039557267 0.0350711867 0.0302410219 0.0175749529 0.0255937688 0.0211654641 0.0194463506 0.0417481475 0.0348132104 0.0254061036 0.0161706936 0.0373525731 0.0197605789 0.0201023 0.0197027475 0.022230899 0.0197698418 0.0266115926 0.0179904085 0.0337820463 0.0189370792 0.0166164041 0.0178276338 0.0198003817 0.0267890915 0.0369485915 0.0202564579 0.023884926 0.0237853099 0.0202794019 0.0214620605 0.0316044502 0.0403577946 0.0254424475 0.0269325934 0.0340213738 0.0325915143 0.0383713916 0 0 0]
   [0.0187943578 0.0358987823 0.0347431712 0.0219421852 0.0162226968 0.0224235654 0.0253677387 0.0390762687 0.033478234 0.0371681415 0.0167776775 0.0292059202 0.0235934984 0.0233368594 0.0403356962 0.0195579156 0.022565132 0.0350562967 0.0201445725 0.0182641502 0.0334370807 0.0186890531 0.0387326 0.0289516114 0.0173518434 0.0339840464 0.0164934397 0.0179543253 0.0192694049 0.0421643369 0.0389746241 0.0329829603 0.0175474584 0.018485494 0.0267191529 0.0222411659 0.0200782754 0.0219901465 0 0 0]
   [0.025467407 0.0159894265 0.0309097283 0.0281576607 0.0295665786 0.0395749 0.0216479544 0.0268964581 0.0162863061 0.028833 0.018830657 0.0223121718 0.036137946 0.0246388167 0.0327756256 0.0272978414 0.0177572183 0.0169437192 0.0323079452 0.0277915206 0.0211077388 0.0184521638 0.0318861194 0.0199025422 0.0329047926 0.0286865253 0.036798086 0.0392495878 0.0374228321 0.0264389664 0.0150047783 0.0245682579 0.0377546437 0.0157504845 0.026851546 0.0181238595 0.0202719402 0.0287023354 0 0 0]
   [0.0279630106 0.03438095 0.0156593602 0.023793187 0.0152197657 0.0283103772 0.0192617103 0.0290444642 0.036751844 0.0173868742 0.0313452967 0.0169406235 0.0388179347 0.0225316398 0.0228223186 0.0202902928 0.0193335246 0.0401728898 0.0333126113 0.0272575859 0.0230737254 0.0222081747 0.0175805595 0.023025658 0.0376363285 0.0229343921 0.0190560352 0.0285491962 0.0299642272 0.024978254 0.024871232 0.0156473517 0.0257736295 0.0317855328 0.0401444249 0.0368264019 0.0203031339 0.0350454412 0 0 0]
   [0.0234189108 0.0233356059 0.0169821344 0.0219378434 0.0196842663 0.0371475033 0.028349543 0.0166484509 0.0241583623 0.0351880975 0.0183715448 0.0145080872 0.0260854084 0.0375431925 0.0302310549 0.0350999609 0.0154193826 0.0150255 0.0206932481 0.0272239726 0.0318211354 0.0152410343 0.0222458523 0.0255517792 0.0301767159 0.0349977799 0.03506919 0.0371580049 0.0194666777 0.0333442241 0.0175094102 0.0365072675 0.0322847329 0.0302756298 0.0224493034 0.0334706903 0.0215403344 0.0338381827 0 0 0]
   [0.0374597684 0.0161677077 0.0201805141 0.0398221575 0.0293919612 0.0224315058 0.039553944 0.0162638407 0.0218282416 0.0165249761 0.0170572791 0.0195855759 0.0185060073 0.0164797194 0.0177183431 0.0295307338 0.0190107767 0.0207821522 0.0320307724 0.0421582125 0.0429754332 0.0282286964 0.0227105301 0.0311704632 0.0165158268 0.0361194536 0.0262594502 0.0246492513 0.0397175886 0.0420808122 0.0168809444 0.0201004576 0.0347088724 0.0160612017 0.0384638608 0.023522 0.0277091488 0.0196417924 0 0 0]
   [0.0188441612 0.0386332646 0.0241514985 0.0220246948 0.0159376785 0.0383360311 0.0320084058 0.0216870736 0.0357543267 0.0274430681 0.0178192127 0.0164461061 0.0149776135 0.0349907055 0.0155586312 0.0350343809 0.0353396945 0.023717327 0.0146549027 0.0154707478 0.0312321186 0.0247341748 0.0153047182 0.0203517247 0.0248250701 0.0327462032 0.0317624 0.0374268219 0.0308441296 0.0364386253 0.0241612233 0.0221870542 0.0267453492 0.0162037686 0.0274202786 0.0342663787 0.0285508521 0.035969574 0 0 0]
   [0.022789672 0.0416070409 0.025203526 0.0278077424 0.0196012743 0.015900366 0.0345994271 0.0264711734 0.0390441976 0.0265645478 0.0364829563 0.0190929621 0.0181205478 0.018052889 0.0240798928 0.0256967563 0.0167933833 0.017292954 0.0164496973 0.0259493608 0.030748779 0.0408358723 0.0319656655 0.0290932283 0.0354162157 0.0156860668 0.0184638854 0.0296025909 0.0205991063 0.0332337394 0.0160153918 0.0418223739 0.0174449757 0.0312903263 0.0291238967 0.0207018368 0.0298914053 0.0304642562 0 0 0]
   [0.0157347042 0.0168757942 0.0241500326 0.0356646255 0.0288313106 0.0167034082 0.0325166918 0.0319833793 0.0381104685 0.0279131699 0.0217530988 0.0242238734 0.0389580056 0.0373708457 0.0214042 0.0158599298 0.0398960784 0.0412008837 0.0232588891 0.0333951972 0.0298049748 0.0155095076 0.0278609321 0.0177138355 0.0167043079 0.0176127255 0.01570246 0.0291251447 0.0252700299 0.0262294598 0.0323449597 0.0153688788 0.0297372937 0.0273406766 0.0295073465 0.0192142446 0.0406384431 0.0185101759 0 0 0]
   [0.0312313344 0.0366127081 0.0265190378 0.0303149782 0.0322385579 0.0185470134 0.0211973656 0.0188536271 0.0332350694 0.0337635279 0.0347426422 0.021876229 0.0164589975 0.0259304736 0.03007566 0.0163088422 0.0358063281 0.0374689437 0.0334299058 0.0312404688 0.0181351788 0.0187908728 0.0377666131 0.0242864098 0.0303018354 0.0170693211 0.0196639039 0.0167536214 0.0205116682 0.0229019299 0.0285076927 0.0149215776 0.0219262913 0.0290582143 0.0336606428 0.0233219769 0.0378306732 0.0187398624 0 0 0]
   [0.0338541493 0.029520452 0.0183341447 0.0309847295 0.0167599022 0.0288019292 0.0337858312 0.0305668805 0.0270072203 0.0264689662 0.033351846 0.034669321 0.0239990186 0.0325156078 0.0252516288 0.0215430669 0.0168026239 0.0312527046 0.0246400461 0.0307684746 0.0385533497 0.0216841064 0.0296443161 0.0211397968 0.0259032361 0.017871717 0.018163288 0.0369662903 0.015670659 0.0391825251 0.0204469934 0.0161417406 0.0231994335 0.0281216707 0.040113166 0.0243604984 0.0163558517 0.0156027684 0 0 0]
   [0.0344383717 0.0266108625 0.0192546751 0.030542057 0.0331796817 0.0181954447 0.0373537466 0.0410928391 0.0245861746 0.0238328669 0.0292795245 0.0165653732 0.02310233 0.0264812689 0.0266602803 0.0216792412 0.0166637041 0.0203391761 0.0390425324 0.0319537483 0.0274375174 0.0396664962 0.0163889769 0.0376121029 0.0160707 0.0171447247 0.0328912 0.0188086461 0.0201589838 0.0169321094 0.0169441346 0.0173246581 0.0347404629 0.0282315165 0.0378468409 0.0327584 0.032315705 0.0158729 0 0 0]
   [0.0165130161 0.0425457843 0.0174048431 0.0299730767 0.0284795742 0.0175537784 0.0184195545 0.0222972706 0.0239583 0.0409607552 0.0205085445 0.0182554 0.0291095842 0.0232424755 0.0300351586 0.0374377556 0.0219585747 0.0281165373 0.028726073 0.0183671657 0.0306161214 0.0231795721 0.0383259356 0.0435972549 0.0187988 0.0202184804 0.0171376728 0.0267294571 0.0164524987 0.0238218904 0.0376602821 0.0206423923 0.0200636331 0.0223237053 0.0436787494 0.0350850038 0.0308793373 0.0169260073 0 0 0]
   [0.0205245577 0.0171858538 0.0175902583 0.0174852014 0.0185003951 0.023262443 0.0397512466 0.0340271145 0.0194577388 0.0361248 0.0345265828 0.029054096 0.0294366591 0.0185102094 0.0338633694 0.0156659503 0.0162627064 0.0323647894 0.0291578751 0.0163362753 0.0378420763 0.016869992 0.0266942866 0.0209670905 0.0250575244 0.0236743297 0.0237292517 0.0196165368 0.020849321 0.0308408309 0.0342585258 0.0329973772 0.0294450168 0.040069066 0.0401286408 0.0206408594 0.0326238535 0.0246073138 0 0 0]
   [0.027943803 0.0254707932 0.0306309592 0.0157783981 0.0335088708 0.0199528337 0.0393110104 0.0327591 0.0307610612 0.0161901489 0.0155367702 0.0360644087 0.0282327235 0.0289415903 0.0169881564 0.0342420824 0.0173311215 0.0251544286 0.0201209784 0.0218471251 0.0335497931 0.0255222321 0.0251689479 0.0351440683 0.0202849917 0.0372758433 0.0182063151 0.023418149 0.0192910582 0.0406109467 0.0161902234 0.020422874 0.0199759491 0.0211360138 0.035855744 0.0153793991 0.0391881764 0.0366129167 0 0 0]
   [0.021250248 0.0294678733 0.0187555924 0.0378244482 0.0151199056 0.0270273536 0.0244622733 0.0379946232 0.0201102328 0.0162161402 0.025978338 0.0158939 0.0280901715 0.0240048785 0.0375719778 0.017848555 0.0195596684 0.0249807164 0.0390582718 0.025638923 0.032465484 0.0356424451 0.0301617868 0.0218559559 0.0171876661 0.0216015652 0.0263116863 0.0307360049 0.0356892683 0.0178151652 0.0359681621 0.0409646854 0.016355373 0.023943413 0.039281249 0.0259638876 0.0177076515 0.0234944038 0 0 0]
   [0.037566863 0.0167545136 0.0237961411 0.018661838 0.0166581664 0.0319020376 0.0213174317 0.0297590084 0.0206613876 0.0155065851 0.0346561559 0.0348346 0.0214527138 0.0285905749 0.014261784 0.0267182067 0.0266357381 0.0261932276 0.0168887861 0.032692451 0.0303220712 0.021075 0.03352651 0.0320453085 0.0304385 0.0334386043 0.032010287 0.0165932607 0.031860102 0.0324823111 0.0149996458 0.0263336133 0.0368733406 0.0196285509 0.0340235196 0.0208033938 0.0229592379 0.0350786075 0 0 0]
   [0.0160642602 0.0223453958 0.0390005708 0.0317619033 0.0221390054 0.0311630629 0.0329159349 0.0221536141 0.0336911455 0.0179807059 0.0274593979 0.0408897288 0.0311409384 0.0205636304 0.0356514938 0.0327269286 0.0226851087 0.0283713024 0.0355541185 0.0164377503 0.0168044362 0.0158921406 0.0233610366 0.0284630917 0.0177901071 0.0160970408 0.0230893865 0.017193621 0.0172367804 0.0365443639 0.0240260903 0.0311002713 0.0188852549 0.0416687876 0.0272478797 0.0371285714 0.0161485244 0.030626595 0 0 0]
   [0.0236848071 0.0284777433 0.0147502478 0.0247045532 0.0380282849 0.0354335904 0.0148531776 0.0158587061 0.0283602066 0.0363723375 0.0163873788 0.0310700741 0.0215956606 0.025824327 0.0312802084 0.0327431783 0.0291395765 0.0191165097 0.0268690903 0.0357744582 0.0349158309 0.0201085098 0.0154505344 0.0151701318 0.0204923209 0.0198812746 0.018234577 0.025336666 0.0250230841 0.0226677954 0.01477818 0.0384958163 0.0397008099 0.0332078934 0.0207995363 0.0343650877 0.034318056 0.0367298536 0 0 0]
   [0.0232704338 0.0249119941 0.0188691039 0.0214962717 0.0388322659 0.0379659459 0.0262215734 0.016198311 0.0160907749 0.0188678652 0.0371477418 0.0419894308 0.0271631237 0.0237867925 0.0387830809 0.0193889029 0.0167185534 0.0185485389 0.0321571752 0.0186893046 0.0188997276 0.0186898392 0.0299008358 0.0231339857 0.0306478869 0.0225622859 0.0182048753 0.0380230434 0.0294662677 0.0422026105 0.0379097052 0.04118919 0.0192823056 0.0174697842 0.0245337263 0.0215835944 0.0275357161 0.0216675047 0 0 0]
   [0.0378642529 0.0402463824 0.0400006659 0.0267534144 0.0367644392 0.0155937718 0.0160110071 0.0239677466 0.037867263 0.0263211597 0.0370885283 0.0221978687 0.0200331416 0.0236251 0.0209863167 0.0320355445 0.0387918 0.0198921151 0.0198034439 0.015841471 0.0254232381 0.0186397359 0.0402027406 0.0246342234 0.0169895738 0.0355064236 0.0240594875 0.0237349272 0.0159140136 0.0267664958 0.0200216249 0.0246300064 0.0189000778 0.0323048 0.0368054546 0.0218616873 0.0186395869 0.0232805 0 0 0]
   [0.0357763171 0.0167599712 0.0205404162 0.0297579635 0.0194609016 0.0173905902 0.0271491576 0.033278849 0.0150509039 0.0231937524 0.0209009685 0.0327284858 0.0396310128 0.0402872823 0.0373596847 0.0219576061 0.0210718215 0.0200547688 0.0179269835 0.0315743573 0.0357690305 0.0251770895 0.0271590315 0.0198146664 0.0287171379 0.0402593128 0.0219384227 0.0382242762 0.0178776551 0.0253998674 0.0307421125 0.0161784943 0.0259114392 0.0255784 0.0152360136 0.025445167 0.030830428 0.0278897 0 0 0]
   [0.0194777213 0.0349953473 0.0234925039 0.0219907612 0.0244111326 0.0320256 0.0163982175 0.0161300153 0.0286881886 0.0153659461 0.0367312133 0.0329404473 0.0217799898 0.0319028497 0.023921812 0.0172862876 0.0251999758 0.0225793086 0.0396682732 0.0376452915 0.0227850825 0.0152081195 0.0246177632 0.0246325769 0.0160141643 0.0311144851 0.0369706638 0.0340650864 0.0166297052 0.0160438381 0.0187987071 0.0394439697 0.0208508316 0.0366607532 0.0347154215 0.019876169 0.0309394095 0.0380023494 0 0 0]
   [0.0265689902 0.0202643778 0.0208008382 0.0441958755 0.0230045319 0.0239813719 0.0214199815 0.0234172158 0.017731674 0.0171323456 0.0417773 0.0343666822 0.0195648707 0.0196750965 0.020469889 0.020640092 0.0181880444 0.0330046043 0.0387721434 0.0188659187 0.0203059092 0.0187300444 0.0259985123 0.0222875439 0.0289365053 0.030348476 0.032288909 0.030913651 0.0368786827 0.0450143069 0.0170755666 0.0226370618 0.0217493679 0.0309110228 0.0348290168 0.0171042141 0.0296137631 0.0305355825 0 0 0]
   [0.0387454 0.0278099868 0.0230215061 0.0327168219 0.0175599568 0.023728963 0.0290858243 0.0214738771 0.0191804525 0.017762227 0.0241969228 0.0270359088 0.0197411049 0.0283962935 0.0329779722 0.02471276 0.0153236873 0.0250978246 0.0393078178 0.0354893841 0.0345598347 0.0392203219 0.0359635577 0.0234093554 0.0200082418 0.0340062678 0.0301741585 0.0154211568 0.025347434 0.0190379974 0.0161910113 0.0203517769 0.0353601202 0.0342534594 0.0221818183 0.0186665785 0.0206535067 0.0318286493 0 0 0]
   [0.0225140024 0.0158418193 0.0153526217 0.0249930024 0.0319609269 0.0364463553 0.0241667666 0.0279344972 0.0191890895 0.0192482136 0.040111091 0.019748807 0.0258479584 0.0365199223 0.0169061199 0.0356765315 0.0238519926 0.0256754402 0.0298797935 0.0200537331 0.0152515182 0.0297261458 0.0197085254 0.0257878099 0.0262125246 0.0346946 0.0320886448 0.0228123479 0.0391842686 0.0167269409 0.0331740305 0.0239979159 0.0294370186 0.020875439 0.0322099291 0.0372026041 0.0242874417 0.0247035921 0 0 0]
   [0.0316652209 0.0193320792 0.0303416606 0.0248275101 0.0158570874 0.038636066 0.0193764362 0.0334128328 0.0226779711 0.0161181949 0.02458404 0.0241383407 0.0193889271 0.0347210914 0.0179972835 0.0340672396 0.0313692614 0.0192010514 0.0309547484 0.0307676718 0.0258466043 0.0227607582 0.0151282474 0.0290227961 0.0162736345 0.0308166519 0.0169625394 0.0216643289 0.0364349 0.0370601378 0.0241045132 0.0367005691 0.0370522104 0.0261377078 0.0180007 0.0201868303 0.0386129133 0.0277992971 0 0 0]]

  [[0.0220022984 0.0190328583 0.0301464926 0.0153923901 0.0152240116 0.0154727492 0.0400385074 0.0175474 0.0150096221 0.0197757389 0.0273229554 0.0263129715 0.028899882 0.0183081739 0.01970632 0.0375559479 0.0379130207 0.0375506543 0.0248927 0.0309654176 0.0224785898 0.019715121 0.0361572802 0.0251306519 0.0313333161 0.0186115876 0.0387957469 0.020093549 0.0360706188 0.0329330191 0.0177387241 0.0328917801 0.031496 0.032529 0.0166535117 0.037825942 0.0189388022 0.0315367058 0 0 0]
   [0.0397222154 0.019039 0.0397637598 0.0194818955 0.0288452357 0.0314444 0.0312608518 0.0231165513 0.0215188283 0.0174351465 0.039669577 0.0163773131 0.0326838158 0.0228090417 0.0268486347 0.0187292639 0.0295324046 0.0353950188 0.017391419 0.0157502312 0.0406392291 0.0329269841 0.0339757912 0.0286651477 0.017467197 0.0210914686 0.0225811824 0.0220042206 0.0197261628 0.0159834176 0.029029198 0.0166974198 0.0169804394 0.0284053 0.0386726484 0.0194320455 0.0373856127 0.0315218531 0 0 0]
   [0.019937396 0.039038267 0.0319296718 0.0172794871 0.0334542 0.0326710641 0.0177393425 0.033895161 0.0282281227 0.0202149786 0.0226186626 0.0337544978 0.0263524428 0.0212373752 0.0304140206 0.0246575624 0.0172946956 0.0235314388 0.0284413137 0.0384409539 0.0167230871 0.022369327 0.0244264118 0.0296535306 0.0158698726 0.0298839938 0.0314745642 0.0199467409 0.0201989803 0.0283267628 0.0300926156 0.0185760185 0.0369451158 0.0238471627 0.016726939 0.0356775187 0.0191403683 0.0389903896 0 0 0]
   [0.0269203335 0.0267926101 0.0378178954 0.0357534848 0.0185993258 0.0196264479 0.0239298604 0.0306697171 0.0345529281 0.0286560301 0.0282097794 0.0395660102 0.0379972719 0.0219872613 0.0151244411 0.0279520284 0.0348375514 0.0245377608 0.0213479251 0.0182383452 0.0386800058 0.0252514966 0.0197780356 0.0203819312 0.0238951575 0.0315842144 0.0310415328 0.017620299 0.0207682513 0.0306974798 0.0227243882 0.0230491012 0.0299356766 0.0228007622 0.0178583637 0.0205012523 0.0167781878 0.0335368961 0 0 0]
   [0.0287570171 0.0389209203 0.0328341313 0.0229395479 0.0165972337 0.0238270983 0.0164037403 0.0390983559 0.0310224555 0.0237311143 0.025916893 0.01941121 0.0364565961 0.0184258688 0.0196426623 0.0173059609 0.0327981338 0.0197252203 0.0370563492 0.03251867 0.0296791 0.0209040679 0.0223387759 0.0166160371 0.0172837973 0.0179004446 0.0316729583 0.034267202 0.0268156622 0.016642483 0.0305489395 0.0176720116 0.0396517701 0.0171001405 0.0277207159 0.0310778804 0.036427144 0.0322916806 0 0 0]
   [0.0251413565 0.0302426592 0.0283877831 0.0322838128 0.0393877178 0.030697573 0.0264131576 0.0426609218 0.0161903724 0.0237161126 0.0173181184 0.037831489 0.0345411524 0.0418360978 0.0411651097 0.0167793762 0.0208957717 0.0310667492 0.0340023115 0.0182443578 0.0190477297 0.0235352349 0.0229576882 0.0212686323 0.0218028985 0.0283329859 0.0210173335 0.0409563 0.0201648343 0.0164426956 0.0233791582 0.0221368782 0.0165618323 0.0198536757 0.0278835557 0.0176406223 0.0218153093 0.0264006611 0 0 0]
   [0.0178299248 0.0254154969 0.0365329348 0.0200980268 0.0312385447 0.036926 0.0413916633 0.0329391286 0.0333810411 0.0207918901 0.0235274564 0.0221742131 0.0196290724 0.0197728463 0.0198436528 0.0183362793 0.0167606659 0.0229936987 0.0222777 0.0228541903 0.018192606 0.021763524 0.0162562076 0.0314346068 0.0413316302 0.0170956943 0.0160342157 0.0312456135 0.0377671793 0.0286435857 0.0184112787 0.0363145955 0.030340476 0.0270614512 0.0244387668 0.0286293142 0.0412533619 0.0290714633 0 0 0]
   [0.0303977989 0.0251045357 0.0231456384 0.0263641942 0.0282877553 0.0260361265 0.0196614526 0.0253612902 0.022578232 0.021711804 0.0245364867 0.0186123513 0.0196256116 0.0208275951 0.0259957518 0.0351130553 0.016607115 0.0362725146 0.032966584 0.0270488486 0.036213994 0.0387834 0.0228451863 0.0335920975 0.0384144038 0.0171589106 0.0176528487 0.0311988536 0.0296926014 0.0338558592 0.0263109468 0.0194766037 0.0168578848 0.0273096561 0.0267830379 0.02399268 0.0163630694 0.0372432172 0 0 0]
   [0.0323885866 0.0263985526 0.0197982658 0.0309251621 0.0303163882 0.0207870267 0.0307043474 0.0240843184 0.0388518162 0.031028511 0.0287451353 0.0211859364 0.0325877331 0.0391779728 0.0371620767 0.0327239707 0.0154685657 0.0285571795 0.0311034229 0.017183492 0.0199138485 0.0160068199 0.0180182476 0.0382733271 0.0206074566 0.0336470716 0.0158113576 0.0212568138 0.0297234058 0.0203965735 0.0198261 0.0341388 0.0310133081 0.0168302543 0.019517161 0.0275983158 0.0209774952 0.0272652432 0 0 0]
   [0.0182819963 0.017815927 0.0162660442 0.0363657176 0.0318957567 0.0414027497 0.0404903106 0.0332872681 0.0211541802 0.0304894261 0.0180012174 0.0193031169 0.038738586 0.0162002798 0.0165940188 0.0337108336 0.0370550863 0.0166864023 0.0311980676 0.0201165341 0.0321199261 0.0314722098 0.0277493019 0.0362166651 0.039297916 0.0159875173 0.0392144434 0.0221149847 0.0178456455 0.0343618691 0.0214445833 0.0161154792 0.0154515095 0.0195238907 0.0201923922 0.0252975598 0.0323454179 0.0181951039 0 0 0]
   [0.0344741754 0.0213189945 0.0307094064 0.0212021526 0.0205519795 0.023057675 0.0314042121 0.0242629964 0.0182020217 0.0200759862 0.0350216813 0.0341475531 0.0196009185 0.03226877 0.0253692716 0.0168309249 0.0313116945 0.0317188874 0.0203452427 0.0216577519 0.0289852582 0.0194910243 0.039067544 0.0190960281 0.0195446 0.0333756953 0.0420297608 0.0264169574 0.0160476174 0.0188506544 0.029160032 0.0290773492 0.032966774 0.0214781519 0.0223447923 0.0179038942 0.0371760577 0.0334555209 0 0 0]
   [0.0167189017 0.035769552 0.0331698731 0.0316200145 0.0213609114 0.0289543662 0.0394719 0.040638119 0.023037862 0.0227933582 0.0279151257 0.0178991314 0.0190757383 0.0186177287 0.0205650795 0.0280140117 0.019808827 0.0249357615 0.018703891 0.0245539751 0.0285637751 0.0304121505 0.0400222056 0.0295699071 0.015319447 0.0205501989 0.0185458474 0.0361220799 0.0170412492 0.0349445 0.0153872306 0.0211490877 0.0301840771 0.0305523984 0.0152706318 0.0203253105 0.0412621275 0.0411535949 0 0 0]
   [0.0223521944 0.021795677 0.0387542732 0.0343235619 0.0391833633 0.022177361 0.0195169412 0.0318436213 0.0156434514 0.029784631 0.0336780362 0.027029328 0.0343169719 0.0314362794 0.0246605333 0.0181988459 0.0251029991 0.0213680249 0.0343593694 0.022501858 0.0257671736 0.0343868956 0.0213504359 0.0283840373 0.0292061977 0.0246541146 0.0155586321 0.0196472127 0.0374982357 0.0279867481 0.0344273448 0.0182683133 0.0361209102 0.028023459 0.0174890384 0.0165593699 0.0169789195 0.019665638 0 0 0]
   [0.0164410155 0.0328067951 0.0403986163 0.0334276929 0.039818 0.0182034504 0.0182926655 0.0271430314 0.0178069305 0.0316151 0.025266083 0.0190101452 0.0181584172 0.0160226 0.0174706802 0.0250334982 0.0348081104 0.0360291861 0.0225561634 0.0166314468 0.0263137463 0.0305988286 0.0342597924 0.0291910283 0.0197179671 0.0302178767 0.0379435457 0.0357028469 0.0378334485 0.0218859334 0.0174639206 0.0232241694 0.025683213 0.0156447422 0.0325304829 0.038234707 0.0170576498 0.019556433 0 0 0]
   [0.019376196 0.0265505854 0.0187935978 0.0210459549 0.0247710515 0.035374511 0.0201881863 0.0384182893 0.0346422791 0.0341403931 0.0361157 0.0241338443 0.021301806 0.0173672494 0.036788512 0.0318557657 0.031920366 0.0323756188 0.0255574845 0.0194998048 0.0168853719 0.0169527084 0.0187486 0.0248465016 0.0197944213 0.0403570905 0.0207531881 0.0182951111 0.0240878165 0.0221950244 0.0227814801 0.0202070065 0.0263050552 0.0249297377 0.0387427099 0.0380834267 0.0175535791 0.0382640362 0 0 0]
   [0.0385861099 0.0169646032 0.035723187 0.0177792422 0.0236127265 0.01725455 0.0313043259 0.0265912339 0.0252973605 0.0322685204 0.0227395725 0.0232488289 0.0229775012 0.034705922 0.0320654809 0.0229564868 0.0381275192 0.0253214035 0.0367895924 0.0385973789 0.0225827266 0.039170038 0.0252677444 0.0354425125 0.0282420367 0.0172029957 0.0190616194 0.0158631727 0.0269868635 0.0152808484 0.0152278179 0.0293651633 0.0269961171 0.0197931528 0.0303935558 0.0272317939 0.0214119311 0.0215682853 0 0 0]
   [0.0238646474 0.0314608738 0.032631427 0.0276835635 0.0205210969 0.0203812514 0.037324395 0.0236865804 0.0339595228 0.0312219802 0.0219297465 0.0242973715 0.0276561286 0.039811343 0.0204523616 0.0187841859 0.0206505116 0.0334474072 0.0374581702 0.0195945818 0.0176604819 0.030441666 0.0340631604 0.0289794374 0.0221611466 0.0264408048 0.0271645878 0.0180797 0.0321306475 0.0179855488 0.0253587794 0.0176626164 0.0210293215 0.0257665552 0.0311845392 0.0269464143 0.0307053179 0.019422194 0 0 0]
   [0.0430209301 0.0368812792 0.0183160771 0.0429846 0.042519249 0.0219170433 0.0166658442 0.0171566885 0.0248103794 0.0209674332 0.0255443044 0.0196134262 0.0290684979 0.0247964058 0.0224256758 0.0163022168 0.0263695549 0.0433094911 0.019135505 0.0229917523 0.0292409975 0.0315447338 0.0178883038 0.016301401 0.0398174785 0.0176652204 0.0316012651 0.0162439495 0.0316054523 0.0392167456 0.0329396948 0.0340748951 0.0182950031 0.0204499606 0.018524738 0.017687548 0.016187856 0.0359183699 0 0 0]
   [0.0162369944 0.0169121642 0.0384209156 0.0204659477 0.0208019 0.0250393171 0.0342888758 0.0365993567 0.0203604512 0.0198266711 0.0405050851 0.0253711753 0.0404336527 0.016070636 0.0202654637 0.0167206209 0.030663792 0.0334691815 0.0261721872 0.0292556863 0.0304550231 0.0353393517 0.0227131527 0.0277232 0.0247139707 0.0175809879 0.0196677484 0.0351565778 0.017793322 0.0164669603 0.0187720023 0.0278018154 0.0417871438 0.0376578607 0.0419834591 0.0181373768 0.019211987 0.0191580337 0 0 0]
   [0.0157769267 0.0375306271 0.0347367711 0.021821538 0.0238910802 0.0256188624 0.0164078493 0.033540342 0.014458959 0.0335762 0.0197387747 0.0202175733 0.0325666927 0.0194146205 0.0159130171 0.0220202059 0.0343734249 0.0310093276 0.0151278544 0.0244685337 0.0349997692 0.0374884158 0.024997076 0.0153724542 0.0367481895 0.0309444796 0.0332638361 0.0299545806 0.0234894231 0.0144704888 0.0322713591 0.0163983535 0.034766186 0.0366542451 0.0282123648 0.0243594348 0.0244146138 0.0289856102 0 0 0]
   [0.0264979322 0.0235175285 0.0318993032 0.0160422344 0.0208761636 0.0217261445 0.0337432176 0.0366642438 0.0401021689 0.0195986703 0.0271474104 0.0219093468 0.0265902523 0.0153089976 0.0251355823 0.0371153578 0.0212195702 0.0165425278 0.0247194842 0.0253941771 0.0277583096 0.0313778892 0.0313761942 0.0187338553 0.0332688 0.0238235444 0.0359067395 0.026886642 0.0188634861 0.0346310697 0.0253463723 0.030272888 0.016441742 0.0352816246 0.0166311283 0.0220632311 0.0334035307 0.0261826143 0 0 0]
   [0.0313077979 0.0164886247 0.0352967493 0.0230439957 0.0261298027 0.0338684246 0.0160497483 0.0308943074 0.0212677699 0.0269908905 0.0190010481 0.0307022594 0.0315587819 0.0368116349 0.0245607495 0.0329328515 0.0279062297 0.0195966251 0.018245684 0.030511206 0.0231805462 0.0254950244 0.0307845119 0.0154191535 0.0408629254 0.0207275636 0.0304693393 0.0348881632 0.0224365983 0.0185588431 0.0361857116 0.0299177989 0.0164613612 0.0217716061 0.0152290501 0.0275766607 0.0390531346 0.0178167801 0 0 0]
   [0.0355961695 0.0220169835 0.0389898829 0.019278165 0.0179219656 0.0363614224 0.0186156295 0.0320054144 0.017407544 0.0189608317 0.019445546 0.027395295 0.0248449165 0.039598994 0.0194881558 0.0335486345 0.0183998719 0.0197698548 0.0318655148 0.0159230661 0.0410268307 0.0164842233 0.0331461504 0.0406063795 0.0241774898 0.0353187881 0.036613781 0.0163278803 0.0189379416 0.0242345519 0.0259049833 0.0336142145 0.0335603915 0.0233524069 0.0242962334 0.0223699119 0.0221956726 0.0203983262 0 0 0]
   [0.0317376442 0.0175646022 0.0236517452 0.0292469542 0.0248377156 0.0250469539 0.0216507502 0.031324707 0.0221067257 0.0334849 0.036336109 0.0165282097 0.018117398 0.0175986923 0.0312247332 0.0189749449 0.0209409576 0.0159407221 0.0252157617 0.0314148292 0.0348174348 0.0189340282 0.0347247422 0.0423031487 0.0337974951 0.0164241362 0.0279634427 0.0403895192 0.0232545454 0.0240639988 0.034877073 0.0306476932 0.0183723867 0.0199075602 0.0354793444 0.0261035077 0.017509589 0.0274853203 0 0 0]
   [0.0259568542 0.0262483452 0.0173208788 0.0255529 0.0364480242 0.0373175703 0.029434735 0.0394912958 0.0267275348 0.0214821212 0.0198211 0.037098147 0.0171550475 0.0216605682 0.0162876882 0.0174766481 0.0164361857 0.0305541288 0.0288924314 0.0278132949 0.0195050854 0.0263189841 0.026857283 0.0168840438 0.0154806124 0.0198772419 0.0169115271 0.0339665748 0.0181483757 0.0279928483 0.0350437164 0.0401557237 0.0350507312 0.0330917686 0.0260084365 0.0341040567 0.017017046 0.0384104252 0 0 0]
   [0.0366646461 0.0209595598 0.0171974394 0.0265766308 0.0271845646 0.0158302523 0.0265454073 0.0422456414 0.0352113657 0.0206987616 0.0201001596 0.0280791428 0.0392742641 0.0335267335 0.018759124 0.0247228928 0.0314473882 0.0162573289 0.0296519045 0.0321901366 0.0167458244 0.0235530138 0.0413568057 0.0389816836 0.0160611235 0.0267795436 0.0333033577 0.0171437133 0.0226461366 0.017407164 0.0182098169 0.0164429 0.0344031677 0.0320875868 0.0266135801 0.0287970621 0.0234352294 0.0229089782 0 0 0]
   [0.0164111424 0.020993568 0.0192535389 0.0230479576 0.0310899429 0.0196138062 0.0233149286 0.0276295263 0.0371250212 0.0380033366 0.0348251648 0.0168332383 0.018252125 0.0367788561 0.0342196152 0.0167008322 0.0350744165 0.0195553824 0.0286727697 0.022885628 0.0196897984 0.0260575749 0.0186399072 0.0437914506 0.0293027963 0.0206698813 0.0229313187 0.0302416626 0.0165534969 0.0241370574 0.0355721489 0.0227691513 0.0248335097 0.0189697333 0.0226051081 0.0395407081 0.0224702172 0.0409436859 0 0 0]
   [0.0162213389 0.0225454904 0.0181687213 0.0196932238 0.0235620383 0.0216692667 0.0206645299 0.0197085328 0.0369924344 0.0243208725 0.0287920907 0.0272356756 0.0271277409 0.03790389 0.0210059565 0.0334713906 0.018106455 0.016224362 0.0184678305 0.0174609832 0.0362145863 0.0395315252 0.0408619046 0.0303214956 0.0239652898 0.0231023096 0.0329005271 0.0348189659 0.0228117164 0.0342284441 0.037637 0.0291307215 0.0195466597 0.0244531576 0.0385660678 0.026143359 0.0178381559 0.0185852777 0 0 0]
   [0.0336335972 0.0339151807 0.0192689355 0.0196958333 0.0236863494 0.0382155776 0.0217464827 0.0293372013 0.0168498792 0.0385541655 0.0213542543 0.027303841 0.0186828598 0.0323209316 0.0268422849 0.0248300079 0.0366466194 0.0212262366 0.0234804545 0.040591944 0.0171185527 0.0384535268 0.0235223603 0.0273265336 0.0165980272 0.0321292356 0.0184591617 0.0359644517 0.0208704 0.02078 0.0183051452 0.0199620426 0.0385641083 0.0331676453 0.0207837708 0.021478178 0.025257783 0.023076389 0 0 0]
   [0.0258887634 0.0362378545 0.0228248183 0.0181019548 0.0207545869 0.0176263079 0.0333064385 0.0252566691 0.0160013139 0.0198116582 0.0300996564 0.0208422244 0.0341760367 0.0318969972 0.0329237431 0.0224983115 0.0178715922 0.0235079657 0.0264516156 0.0184358396 0.0291365944 0.015376457 0.0379800908 0.0292471163 0.0365858302 0.0379972495 0.0269860607 0.0174355451 0.0314762 0.0328058824 0.0191593859 0.030509714 0.015773138 0.0328202322 0.0332036465 0.0233413558 0.0394100845 0.0162410829 0 0 0]
   [0.0211437456 0.0316677615 0.0411545299 0.0169330128 0.0194749 0.0386113934 0.0261596479 0.0163565241 0.0211547352 0.028599117 0.0221385751 0.0251281373 0.0286279507 0.0359199382 0.0180950109 0.026252823 0.0423618294 0.0184894949 0.0403229967 0.0382610708 0.0237518642 0.0160959251 0.0165056027 0.0410542637 0.0164645631 0.023045741 0.0332867242 0.0165046956 0.039955724 0.0174395368 0.0223226957 0.0302970409 0.0175871029 0.0207543168 0.0215186439 0.0290183779 0.0371231586 0.0204208232 0 0 0]
   [0.0241821893 0.0223279409 0.0289353188 0.0207986515 0.0222154204 0.0394791402 0.0269094873 0.0300360583 0.030717697 0.033110939 0.0228291284 0.0240185875 0.0373215079 0.0277017113 0.0242115948 0.021304531 0.0317256227 0.0297067165 0.023862 0.0162413 0.0238980111 0.0192767195 0.0165132619 0.0263579953 0.0224743187 0.022383688 0.041344326 0.0168169104 0.0338812396 0.031792637 0.0406151786 0.0259042 0.0166638978 0.0182883516 0.0167107582 0.0300759282 0.0186655968 0.0407013483 0 0 0]
   [0.0319150314 0.0246525742 0.0351598933 0.0281999037 0.0269261338 0.0191912744 0.0165238827 0.0192132145 0.0213021412 0.0155008081 0.0383252464 0.0253168773 0.0167446434 0.0173466131 0.0342828892 0.0412308611 0.0153118391 0.0184650421 0.0250938851 0.0255103502 0.0171141587 0.0261419173 0.0178561769 0.0236439537 0.0339305736 0.036770843 0.0185917839 0.0402090251 0.0264310483 0.0213493258 0.0355575345 0.0181102864 0.0411048569 0.0353181772 0.0327861719 0.0383767448 0.0240676869 0.0164266191 0 0 0]
   [0.0356192589 0.0141744753 0.0162942093 0.0161341038 0.0261157528 0.0233711675 0.0176170934 0.0364552364 0.0173367243 0.0234261844 0.0219870824 0.0198487435 0.0252336245 0.0297319908 0.0177495331 0.0329911113 0.0188809764 0.0312470384 0.020846881 0.0363482386 0.0154367751 0.0382832326 0.0273505822 0.0284091439 0.0263017248 0.02003159 0.0361823887 0.0372111537 0.03023009 0.0236495063 0.0235363115 0.0382888876 0.0247517936 0.022867769 0.0332804434 0.0311969817 0.0351185724 0.0264635906 0 0 0]
   [0.019256426 0.0325700529 0.0235921573 0.0316596813 0.0175601672 0.0270660371 0.0244571269 0.0255195722 0.0293864403 0.0379765518 0.0254604686 0.0188877843 0.0389255732 0.0174237974 0.0165654533 0.0280942917 0.0301911477 0.0299435575 0.0212797504 0.021048123 0.0217139702 0.0174096078 0.0195533782 0.0260980856 0.0373598374 0.0226622149 0.0241701156 0.0204471201 0.0161743388 0.0290288161 0.0276074186 0.0387023576 0.0383151807 0.0338076763 0.0354658 0.0383370034 0.0158983618 0.0203845594 0 0 0]
   [0.0406410135 0.0178144965 0.0260009915 0.0252207983 0.0212475564 0.0219034217 0.0213875957 0.0190066081 0.02904737 0.0278521832 0.0199506804 0.0359064639 0.0329041816 0.0170064159 0.0244566519 0.0225502644 0.0277215522 0.039783325 0.0184548441 0.0331519805 0.0162798334 0.0246760156 0.0174441207 0.027217526 0.0366360024 0.0268032532 0.0185579509 0.0299174916 0.0219967403 0.0255174357 0.0321136713 0.0190461203 0.0308722053 0.0403583795 0.0175177176 0.0223191828 0.0343451537 0.0363728404 0 0 0]
   [0.018338332 0.0340013728 0.042475678 0.0262249727 0.0207971726 0.0172073301 0.0363556482 0.0236519929 0.0177242458 0.0232040193 0.0194053296 0.0196847375 0.0380835682 0.0216605049 0.0246836413 0.0212761983 0.0254950915 0.0388605185 0.0324065089 0.0265908092 0.0325202085 0.0255825669 0.0249843691 0.0294753257 0.0286768042 0.0238347892 0.0205652285 0.0391638242 0.0280644316 0.0177273173 0.0231496431 0.0238769222 0.0284801405 0.0263918284 0.0190037917 0.0251810104 0.0238015782 0.0313925631 0 0 0]
   [0.0282172095 0.0183603652 0.0258853137 0.021744797 0.0422650874 0.0211064462 0.0173579175 0.0169272181 0.0223077014 0.0310324132 0.0169747286 0.0324108638 0.0221138354 0.0167046785 0.0318594053 0.0380039923 0.0174498707 0.040330898 0.0390044339 0.0253385566 0.0160782039 0.0314791128 0.0342656039 0.0247997735 0.0299038719 0.0297547784 0.0191612635 0.0165795628 0.0386252925 0.0381633863 0.022272462 0.0264914148 0.0237854589 0.0323674865 0.0225199405 0.0206555855 0.0238477923 0.0238532741 0 0 0]
   [0.0310883354 0.0270301774 0.0406274907 0.0178640373 0.0171659105 0.0314581245 0.0177552477 0.0395179279 0.0356534868 0.0408342816 0.0264986344 0.0282066967 0.0246545877 0.020860225 0.0330613665 0.0211936776 0.0272962786 0.0192517173 0.0219947081 0.0262419414 0.0332988761 0.0168175921 0.0268929377 0.0268197693 0.0303179231 0.0157488938 0.0188556183 0.029750146 0.016730627 0.0316322595 0.0266503748 0.0363694504 0.0201441683 0.0196321886 0.0251423959 0.0158817153 0.0381648615 0.0228953436 0 0 0]
   [0.0336385183 0.0204136986 0.0164914671 0.0324446298 0.0236958843 0.0253424421 0.021719031 0.0334095173 0.0311717559 0.0288922507 0.0246224105 0.0190881453 0.0204022191 0.0232325979 0.027178416 0.026574539 0.0304902773 0.0412797704 0.0170647167 0.0328295343 0.0340034477 0.0254479833 0.0316750631 0.0302456599 0.0164436735 0.0193323679 0.020873988 0.0209608488 0.0393973477 0.022305239 0.0166158397 0.019838538 0.0164077301 0.0196787864 0.0424536951 0.041262567 0.0304527488 0.0226225704 0 0 0]
   [0.0201896615 0.0400509499 0.0241716467 0.0178314745 0.0326221734 0.0293164197 0.0415775441 0.034155786 0.0341221429 0.0357283875 0.0339568444 0.0170722101 0.032945767 0.0385398939 0.0286455899 0.0317168795 0.0290441867 0.0180912632 0.0224754158 0.0229299422 0.0323682167 0.0208884142 0.0388424248 0.0217255764 0.0271832533 0.0177730583 0.0179024376 0.0172760934 0.01620006 0.0177618377 0.0177974273 0.0241776 0.0175301824 0.0194494296 0.0183263924 0.0164898988 0.0377089158 0.0354145579 0 0 0]]

  [[0.0309333485 0.0326295 0.0392814316 0.0174490698 0.0262051 0.0247989707 0.0305590462 0.0231516492 0.0257146284 0.0207982324 0.0232916344 0.026788583 0.0183432177 0.0231509972 0.019960396 0.0312697329 0.0161861 0.0366881378 0.0296577774 0.0167490318 0.0264415778 0.0177209843 0.0244158469 0.0200051 0.0196459629 0.0371234 0.0160780102 0.0386496186 0.0198291745 0.0256752372 0.0333149917 0.0259653777 0.0291891899 0.0344355293 0.0375517718 0.0263886042 0.0215815809 0.0323815234 0 0 0]
   [0.022870196 0.0280972887 0.0184581913 0.021823667 0.0320531465 0.03770262 0.0182720181 0.0309939161 0.0207055379 0.0265352968 0.0222186558 0.0334972 0.045958586 0.0181594454 0.0316819176 0.0213171877 0.0177643262 0.0263086166 0.0201552249 0.0261847246 0.0396361612 0.0175772589 0.0375517122 0.0235433932 0.0259471405 0.0176471062 0.0190050434 0.0211259443 0.0229949486 0.0214767847 0.0248994734 0.0430557244 0.0335697718 0.020669682 0.0431237146 0.0228985269 0.022679599 0.0218402855 0 0 0]
   [0.0168269817 0.0159696732 0.0208435208 0.0164474025 0.04190946 0.0175513197 0.0311964154 0.040262498 0.0317191854 0.0390815102 0.0269082673 0.0197419636 0.035877265 0.0229724385 0.0176509973 0.0244707782 0.0293057207 0.0204353966 0.0189067312 0.0396835804 0.0186506417 0.0244494695 0.0290012732 0.0275287181 0.0417774729 0.0295489356 0.0212328043 0.0279617272 0.0308259726 0.0236920137 0.024206385 0.0187210515 0.0218572766 0.0248403437 0.033823695 0.0211910699 0.0236045066 0.0293256659 0 0 0]
   [0.0252784267 0.016103562 0.0288109016 0.0219170302 0.0237758197 0.0176048167 0.0400326438 0.031858217 0.0377609245 0.0282824188 0.0320629291 0.0177894831 0.0193624888 0.0198353212 0.022833813 0.0390019156 0.0156908873 0.0323237665 0.0392928 0.0278396215 0.0375570282 0.0237211306 0.0317165256 0.0352396704 0.0175607819 0.0283749476 0.020112101 0.0159219764 0.0308744181 0.028124202 0.0193548203 0.015410684 0.0170847066 0.0318648405 0.0259222556 0.0318313055 0.0165063459 0.035364449 0 0 0]
   [0.0258876309 0.0251164436 0.0334198214 0.0383603573 0.0237717871 0.0220099799 0.0176682584 0.0192358047 0.0330686197 0.0181081146 0.0413052067 0.0173949115 0.0204176959 0.0239350758 0.0153782275 0.0233003665 0.0221052375 0.0241122544 0.0247703381 0.0283717774 0.022774009 0.0210836716 0.0251852013 0.0262243599 0.0279190149 0.0289169531 0.0286914725 0.0336836055 0.0337565877 0.0232622363 0.0158072151 0.0338500775 0.0321068801 0.0154084647 0.0405180976 0.0222109519 0.0372820087 0.0335813425 0 0 0]
   [0.0348289832 0.0265685525 0.0281102024 0.0269693062 0.0312537141 0.0165115669 0.0171251819 0.0159339085 0.0355727561 0.0254120082 0.0318127833 0.0267664362 0.0375419185 0.020747168 0.0196588803 0.0301471259 0.0157746654 0.0318809263 0.03008442 0.0197479874 0.0214368291 0.0380603671 0.0159346331 0.0157782324 0.0195286572 0.0356726646 0.0284895822 0.0247363411 0.0294068474 0.027062567 0.0225184821 0.0350946486 0.0405919142 0.0171280615 0.0398223698 0.0302795805 0.0167406071 0.0192691479 0 0 0]
   [0.017690856 0.0398023315 0.0372554921 0.0307250675 0.0331332795 0.0192083344 0.0230646934 0.0258911587 0.0274106674 0.0406728536 0.0169280265 0.0330826528 0.0193813425 0.0181993581 0.021663066 0.0333321244 0.0263170097 0.0190863889 0.0179307815 0.0197698865 0.0185560789 0.0408168174 0.02253142 0.0262359157 0.032041695 0.0175282676 0.0250303876 0.0329943225 0.0204612166 0.0353558846 0.0233501103 0.0220192689 0.0205039233 0.0260171667 0.0228639655 0.0213294569 0.0337582566 0.038060464 0 0 0]
   [0.0347597 0.0256396458 0.0332315601 0.0167365167 0.0284375753 0.037854977 0.0292837638 0.0151430592 0.0301430319 0.0380052328 0.0165294707 0.0303046964 0.0380344614 0.0202853307 0.027834814 0.0216082539 0.0156537909 0.019132603 0.0258140881 0.0346760713 0.0218796581 0.0302112363 0.0202713665 0.0337811075 0.0175169036 0.0159042124 0.0202847719 0.0371434465 0.0188596603 0.0298149958 0.029060185 0.0352584422 0.0201403126 0.0245035719 0.0277258512 0.0162475742 0.025635425 0.0366526879 0 0 0]
   [0.0162653103 0.0384015031 0.0274056066 0.0180883519 0.0255441964 0.0167603865 0.0379743055 0.0176107828 0.0156673286 0.0242699124 0.0257892702 0.0273850299 0.0333032683 0.0408244245 0.0317837857 0.0272019 0.028501194 0.0366931744 0.0194209702 0.0409637503 0.0233187247 0.0156019181 0.0171701517 0.0306359399 0.0320374519 0.0370530188 0.0375229195 0.0309178103 0.0177270826 0.0231253915 0.0183623806 0.0266232528 0.0186062101 0.0251072869 0.0215457343 0.0219016075 0.0328863524 0.0200022981 0 0 0]
   [0.026733594 0.020480372 0.036445722 0.0408161618 0.0171091482 0.0282398034 0.0177757218 0.0238880832 0.0248688851 0.0244072583 0.0303788595 0.0153142242 0.0339773297 0.0357225873 0.0210457928 0.0406436212 0.0190497972 0.0167578179 0.0233863816 0.0227332786 0.033826638 0.0167884361 0.0398886837 0.0217681918 0.0330156 0.0381965116 0.0365536772 0.015475383 0.0176346339 0.0235471893 0.0232153125 0.0255154 0.0262313355 0.0154411914 0.0189125109 0.021032149 0.033740364 0.0394423343 0 0 0]
   [0.0201005545 0.0190095417 0.0360178687 0.0195481759 0.0365715 0.0291670952 0.0157383941 0.0202892106 0.0224681739 0.0219410192 0.0384128205 0.0232509021 0.0365222469 0.0210176539 0.0212713778 0.0287615582 0.0163336191 0.018522637 0.018261062 0.0291057918 0.0251131505 0.026916638 0.0302472301 0.018007502 0.0321336314 0.015805712 0.0406983122 0.0417492092 0.0416640118 0.0320299938 0.0336321555 0.0358402319 0.0305791199 0.0270726942 0.0227496102 0.016423963 0.0192339625 0.0177916419 0 0 0]
   [0.0317741744 0.0357821062 0.0251118764 0.0219096 0.0232983287 0.0324226506 0.0222361907 0.0191036332 0.0310641173 0.0251232218 0.0256794561 0.0289171934 0.0381451026 0.0208218414 0.0387175828 0.040407043 0.0255401768 0.020931704 0.0301944837 0.0326543 0.0278358813 0.019929722 0.0180555154 0.0236094017 0.0197728965 0.0225046724 0.0232740939 0.032472726 0.0217357557 0.0159663297 0.0326666273 0.0304526184 0.0205368362 0.0235959794 0.0213052444 0.0294838287 0.0313017592 0.0156653654 0 0 0]
   [0.0215248838 0.033307258 0.0323354378 0.0266016349 0.0240592547 0.0209686197 0.0365813971 0.035374064 0.0210188963 0.0342943296 0.0207233895 0.0221327543 0.0289531965 0.0184647329 0.0414841622 0.0179920588 0.0193377305 0.0280477274 0.0289769229 0.0301152412 0.0266836602 0.0201892033 0.0308198892 0.0193710104 0.0204925984 0.0179889947 0.02094521 0.0189523883 0.0331509598 0.045392178 0.0216637887 0.0195968412 0.0201822445 0.0301310439 0.0439162664 0.0213356446 0.0216939859 0.0252004154 0 0 0]
   [0.0259657037 0.0308232512 0.0267688259 0.0182989724 0.0256699901 0.0166482255 0.0231643505 0.0364191942 0.0263366494 0.0404989757 0.0202325638 0.0158803407 0.0317459591 0.0249311309 0.0229788 0.0161023829 0.0202233791 0.0376056097 0.0210069437 0.0212603 0.0269979686 0.0354416 0.0374161303 0.0177117251 0.0225034971 0.0363181345 0.0326963291 0.0365982465 0.0286461543 0.0266074389 0.0284912363 0.0304410029 0.0169624891 0.021257339 0.0208745934 0.0208530258 0.0217193831 0.0359021649 0 0 0]
   [0.03346828 0.0293350164 0.0188840199 0.0320068076 0.0152009977 0.0313367918 0.024176985 0.0177895688 0.032422021 0.033620663 0.0170036592 0.04014102 0.0207744502 0.0200808905 0.0161247533 0.0338724256 0.0181008186 0.0305617843 0.0156883318 0.0184576381 0.0294657163 0.0218360201 0.0383095145 0.0154264951 0.0325750932 0.0219195448 0.0406745821 0.0381122679 0.0245600212 0.0219427384 0.0214681868 0.03838972 0.0228565726 0.0243867096 0.0223850161 0.0202569738 0.028684618 0.037703298 0 0 0]
   [0.0315304287 0.0403694697 0.0243374221 0.0259487201 0.0178593453 0.0227467399 0.0201878417 0.026463002 0.0286165662 0.0395777859 0.0380609147 0.0174264386 0.033671882 0.0314319842 0.0189993698 0.0275617447 0.0345550179 0.0182552319 0.0196598861 0.0209123846 0.0247802194 0.0210159831 0.0157865733 0.0366222858 0.0311806314 0.0154739041 0.0328829885 0.0179576799 0.0245514829 0.0227231886 0.0352320187 0.0276268497 0.0232720226 0.0360699482 0.0267460737 0.0188625287 0.0351414531 0.0159019548 0 0 0]
   [0.0172801986 0.0296271406 0.0296358317 0.0375256278 0.032553684 0.0164418966 0.0335505717 0.0253141373 0.0163206905 0.0342538245 0.0298376698 0.0240352619 0.0304804612 0.0169446878 0.0188709423 0.0221100636 0.0200763717 0.04262897 0.0314756818 0.0174756497 0.0285129845 0.027785033 0.026398018 0.0174159072 0.0168874897 0.0304697063 0.0277213715 0.0250523239 0.0355312489 0.0357141569 0.0417673923 0.0163320564 0.033603292 0.0195738599 0.0200468 0.0271443333 0.0200949702 0.0235096365 0 0 0]
   [0.0198372249 0.0187498592 0.0197388679 0.0174062066 0.0219920948 0.0352641791 0.0206611361 0.0196922347 0.0267023016 0.0230210982 0.040719118 0.0379917733 0.0242570862 0.033187218 0.020047171 0.0295670722 0.0398916267 0.0165024381 0.017509684 0.0179789513 0.01822461 0.0165862069 0.0227569956 0.04135276 0.0362655632 0.0336959064 0.0307449512 0.0235618632 0.0304850657 0.0165882912 0.018947022 0.0276504662 0.0379304066 0.018346088 0.0344309136 0.0431109741 0.0217298027 0.0268747322 0 0 0]
   [0.035142865 0.0368206203 0.015349173 0.0209322032 0.0350036398 0.015372036 0.0259507746 0.0279004 0.0232617985 0.0191431865 0.0173342582 0.0367075391 0.0156050865 0.0285001788 0.0370088629 0.0196249075 0.0149640441 0.0273745712 0.0177876744 0.0237793103 0.0289152153 0.0187890138 0.0393100418 0.0171763133 0.0391982794 0.0323258899 0.019785285 0.0169586483 0.0354141 0.0327434167 0.0213949829 0.0398590863 0.0360402316 0.0380470268 0.0169202983 0.0361029804 0.0149828605 0.0224731937 0 0 0]
   [0.0333937593 0.0381970517 0.0159259532 0.032886792 0.0175482668 0.0346701518 0.0306627788 0.0242387652 0.0173166804 0.0182806905 0.038590543 0.0210331436 0.0212901775 0.0361406878 0.038431067 0.0146096973 0.0262922179 0.0360499658 0.0388286747 0.0339578278 0.0296593793 0.0280525312 0.0205712058 0.0304240119 0.0162620712 0.0212871712 0.0167390127 0.0208042879 0.0244615078 0.0286528096 0.0345668718 0.0204585306 0.0167679563 0.0230631474 0.0346485451 0.0293670185 0.0197861735 0.0160828102 0 0 0]
   [0.0251379889 0.0282918494 0.0234315265 0.037757013 0.0169370454 0.0253603421 0.0297332238 0.0193983 0.0177536514 0.0356997401 0.0259221755 0.0280989632 0.0157216545 0.0262028519 0.0210707821 0.0163450949 0.0162693616 0.0330935381 0.0411237068 0.0308885481 0.0211607572 0.029725384 0.0186609775 0.028324198 0.0402542502 0.0233087745 0.0412085652 0.0353557356 0.0294647831 0.01803484 0.0403751209 0.0280988365 0.015976999 0.0204571541 0.0165731218 0.0292362683 0.0170002468 0.0325467028 0 0 0]
   [0.0235674139 0.0237227902 0.0264562014 0.0289448835 0.0221085101 0.0248003639 0.0184701886 0.0330387168 0.0188510343 0.0302695371 0.0311538093 0.0368900523 0.0309183188 0.0217466429 0.019984737 0.0266943611 0.0249491762 0.0346301459 0.0285866726 0.0168519523 0.0253828112 0.0318929665 0.0329336673 0.0364982188 0.0194759704 0.0236331578 0.0257978216 0.0281473137 0.0231845099 0.022962518 0.0363536812 0.0178772919 0.0255657621 0.0181259755 0.0170610193 0.0161394868 0.036450129 0.0398822762 0 0 0]
   [0.020299105 0.0377269089 0.0168291423 0.0343295969 0.0303855222 0.0204398353 0.0316189155 0.0201309 0.0232419465 0.0240499079 0.0308679119 0.019779902 0.0218415353 0.0202740524 0.0401915759 0.0295215771 0.0221470874 0.0300495327 0.0232646987 0.0231059249 0.0307866111 0.0424757302 0.0255510118 0.020834446 0.0173516534 0.0358331129 0.0180571675 0.0192393381 0.0206889678 0.0383261256 0.0195139404 0.0384050086 0.0183607396 0.0268073063 0.0316946916 0.0226381794 0.0184359904 0.0349044241 0 0 0]
   [0.028972473 0.0181510169 0.0351524428 0.0295086857 0.0327708721 0.0321310461 0.0342702754 0.0252343211 0.0291002356 0.0277606789 0.0240830183 0.0178432539 0.0157652386 0.0368314646 0.0227516312 0.0306273587 0.0160976648 0.0341230854 0.0368013158 0.0220672 0.036124941 0.034763936 0.0196917523 0.0309290532 0.0157765895 0.024305569 0.0160111729 0.0154559715 0.0309413932 0.0215579532 0.0235088058 0.0317127854 0.0149918 0.0223104227 0.0210474823 0.0328975879 0.0284778588 0.0294516 0 0 0]
   [0.0208453666 0.0253881663 0.037262883 0.02234184 0.0275382251 0.0371582694 0.0182633307 0.0196962971 0.0367009 0.0366333276 0.0317119807 0.0371128619 0.0256263856 0.0158872139 0.0161730833 0.019419672 0.0256200582 0.0389386304 0.0323307961 0.0253594033 0.0167905707 0.0210193302 0.0238371883 0.0217701737 0.0338882469 0.0228044093 0.0173805282 0.0372871384 0.0389991812 0.0243972614 0.0358222127 0.0380906835 0.0171085261 0.0235624295 0.0197538715 0.0236474909 0.0164393596 0.0173926763 0 0 0]
   [0.0339711346 0.0257245693 0.0391912237 0.0193598457 0.0265528318 0.0190616883 0.016149763 0.0408505313 0.0204360466 0.0285046194 0.0212737322 0.0214155074 0.0277395677 0.0280728657 0.0400363468 0.0173780508 0.0184322279 0.0337311774 0.0190236643 0.0298600681 0.0390297957 0.0211235937 0.0266516414 0.0227109324 0.0171400178 0.0239770971 0.0195818376 0.0413669981 0.0160184894 0.0400945656 0.028369559 0.030434465 0.0342326388 0.0176081751 0.0382013693 0.0189988744 0.0168560613 0.0208384357 0 0 0]
   [0.040592473 0.0242768656 0.0174961686 0.019397581 0.0354648083 0.019043088 0.0256187413 0.0295966677 0.0193787254 0.0191980489 0.0174516384 0.0289481506 0.0404781103 0.0222435296 0.0309341662 0.0308666769 0.0225442667 0.015541832 0.0401962474 0.0339974761 0.0175278019 0.0369524732 0.0174609181 0.0385058634 0.0263914745 0.0177297425 0.015242462 0.0231811367 0.0372084118 0.0208347831 0.022186609 0.0225977283 0.0227903966 0.0325193442 0.0295828115 0.0285553373 0.0235288199 0.0339386798 0 0 0]
   [0.0404692516 0.0272160228 0.0168104805 0.0159201045 0.028238114 0.0265076533 0.0307019576 0.0186546631 0.0158216655 0.0299353357 0.016694732 0.0223482158 0.0370311812 0.0358270742 0.0231726132 0.02973805 0.0186667666 0.0369164199 0.0404019132 0.0224349294 0.0285114236 0.0202577747 0.0346905701 0.0251544099 0.0358004048 0.0265292749 0.0274540521 0.0180060882 0.0253963899 0.0213613193 0.0217053723 0.0282500535 0.0333485268 0.0230826605 0.0306637697 0.016961256 0.0172909107 0.0320286565 0 0 0]
   [0.0424151532 0.0175151937 0.0373673663 0.0324495919 0.0276690144 0.0353512876 0.0199338198 0.0159624442 0.0241322294 0.0423097201 0.0161198769 0.0200852752 0.0311548095 0.0350807086 0.0170693733 0.0413670689 0.0168376807 0.01835737 0.0326976702 0.017787274 0.0200992823 0.0315812379 0.0300135687 0.0320336744 0.032683488 0.0201840531 0.026785858 0.0246505365 0.0161820017 0.0217139274 0.0223267563 0.0421272479 0.0246552713 0.0203754045 0.0289138258 0.0220609233 0.0168336928 0.0251163039 0 0 0]
   [0.0198032446 0.0256525427 0.0348712914 0.0246553961 0.0365666561 0.0255332496 0.0382343717 0.0274257082 0.0372408591 0.017779164 0.0228545088 0.0169113707 0.0212936532 0.0326790921 0.0191924162 0.0164496079 0.032241419 0.0182925221 0.0197107513 0.0316983052 0.0348832756 0.0376431793 0.0268539898 0.021298008 0.0215501972 0.0296737608 0.0306860693 0.033053115 0.0156831965 0.021538021 0.0261186697 0.030095024 0.0154258348 0.0309794173 0.0380471423 0.0231283661 0.0233342052 0.0209223572 0 0 0]
   [0.0200362951 0.0217636079 0.0388150401 0.0223331265 0.0181477908 0.0289906096 0.022825975 0.0162232891 0.0235745925 0.0395260341 0.0169140622 0.0429974981 0.037043348 0.0300929509 0.0399328507 0.0437326059 0.0183513872 0.0224755146 0.0248636063 0.0291200709 0.0216309242 0.0200926103 0.016712375 0.0347359702 0.0327568837 0.0250033215 0.0252553709 0.0196689758 0.0173645318 0.0219748653 0.0296601392 0.0336644836 0.0218404364 0.0168599971 0.0338499211 0.0185110755 0.0253789555 0.027278915 0 0 0]
   [0.0157887414 0.0150920143 0.0373016782 0.0200357139 0.0281309634 0.0280017108 0.0294341221 0.0175655056 0.0199855138 0.0311016869 0.0343783386 0.0195980147 0.0291620679 0.0347770117 0.0307280421 0.0191120096 0.0296609197 0.0325999856 0.0225749593 0.0201804321 0.0355417281 0.0211212393 0.0379995 0.0378195122 0.0165424328 0.0296938345 0.0217949748 0.0327174887 0.0209037121 0.0348042585 0.0361388065 0.0244049765 0.0320752189 0.0203434788 0.0159240328 0.0318083316 0.0202104934 0.0149464719 0 0 0]
   [0.0228956174 0.0231348276 0.0325881205 0.0198301841 0.0424191765 0.0444892496 0.0192271918 0.0239115749 0.0237348713 0.0180502702 0.0281189065 0.029074043 0.0281212106 0.0174856763 0.0205446575 0.0178554356 0.0335685 0.0435457155 0.044080548 0.0414976403 0.0218428988 0.024307495 0.0198663957 0.0221233014 0.0192317814 0.0223821346 0.033779256 0.0185067188 0.0264516231 0.0256675314 0.0174533222 0.0308708679 0.0205049105 0.0263468754 0.0201080889 0.0364997722 0.0207312778 0.0191523395 0 0 0]
   [0.0188211966 0.0246667024 0.0295552667 0.029930329 0.0221123118 0.0422595777 0.03403005 0.0161808245 0.0203268584 0.0231966581 0.0268305223 0.0268964842 0.0206489749 0.0227257684 0.0406494476 0.0303607453 0.0247466229 0.0200854987 0.0172330029 0.0244784858 0.0177145097 0.028594831 0.0406451859 0.020211922 0.0179566685 0.0238704961 0.0425067209 0.0204302222 0.0252984 0.0224527959 0.0355419405 0.0254292171 0.025136115 0.0163288079 0.0235869251 0.0292547774 0.0348132849 0.034491878 0 0 0]
   [0.0159243681 0.0255364757 0.0332439356 0.0228503365 0.0187841468 0.0154819377 0.0279804431 0.0302231237 0.0404787101 0.0379841663 0.0239978041 0.0367632806 0.0280076116 0.0360834301 0.0350171551 0.0221008845 0.02349177 0.0173309445 0.0238720737 0.022519283 0.0264820363 0.0197725091 0.0367809199 0.0173910242 0.0226156134 0.0163231213 0.0168854 0.0168454908 0.0375236832 0.0257975981 0.0289770607 0.0308504552 0.0353095 0.0285783801 0.0317661501 0.0153733585 0.0322651491 0.0227906182 0 0 0]
   [0.036744874 0.0188888889 0.0374804661 0.0169595648 0.0212001149 0.0178394467 0.0276534539 0.0378737189 0.0408299863 0.0261225831 0.0182754751 0.0365978032 0.0351891331 0.0217330232 0.0363370292 0.0210417919 0.0167843886 0.0183813497 0.0192755535 0.0195690282 0.0158246383 0.020182481 0.021640759 0.0414592735 0.0368009433 0.0209033675 0.017500218 0.0158413015 0.027342502 0.0417100899 0.024271762 0.0279858466 0.0308637042 0.02857407 0.0155199114 0.0249414854 0.0295179822 0.0343419611 0 0 0]
   [0.0261473358 0.0201554112 0.0264534988 0.023784088 0.0191487931 0.020347571 0.0415074863 0.0299474206 0.0178230219 0.0252983943 0.0203352217 0.017982522 0.0166013669 0.0353063121 0.0204028208 0.0272990912 0.0195844527 0.0178083945 0.0248734448 0.0350755677 0.0206551086 0.0179961156 0.0406399108 0.017291259 0.0293885302 0.0409226567 0.019264074 0.0429730043 0.0212538429 0.0319685563 0.0226508845 0.0236382876 0.0362249203 0.0310654398 0.0191817135 0.0272362884 0.032521762 0.0392453782 0 0 0]
   [0.0334794037 0.035162475 0.0212402306 0.0275651664 0.0163160153 0.0162630249 0.0190316457 0.0147435348 0.0187507756 0.0261234213 0.0249981266 0.031385079 0.0369821563 0.0226532631 0.03459315 0.0217845589 0.0374434181 0.0308259409 0.0198581833 0.0186457466 0.0320904739 0.036247883 0.0267609153 0.0183096789 0.031828586 0.037350677 0.0376788378 0.020958418 0.0377534851 0.017149603 0.0236184988 0.0199358761 0.0361026414 0.0192798693 0.0348511599 0.0217439141 0.0152881127 0.0252061281 0 0 0]
   [0.0160749909 0.0362193733 0.0179597232 0.0309027024 0.0169999693 0.0228240937 0.0382256471 0.0177877 0.0309269242 0.024899669 0.0163022596 0.0418101475 0.0171347596 0.0252810679 0.0166247636 0.0287473723 0.0428280532 0.0252526868 0.023785051 0.0211812817 0.0240775254 0.0430587381 0.0204820316 0.0242742058 0.0217979606 0.0167400688 0.0301378909 0.0255128276 0.0314960219 0.0361826122 0.0221232083 0.0233153179 0.0263421331 0.0340260044 0.0175138675 0.0303736385 0.0378330536 0.0229446292 0 0 0]
   [0.0286672078 0.0162737723 0.0169573445 0.0186790843 0.0387082808 0.0268943533 0.0199910253 0.0199646782 0.037643753 0.0195724256 0.0184022244 0.0306243561 0.0384567119 0.0278635286 0.0242223628 0.0197907146 0.0198515952 0.021784801 0.0205473658 0.0366461203 0.0420841128 0.0189335458 0.0270323325 0.0238394383 0.0397631861 0.0324374549 0.015697604 0.0328354724 0.0288070291 0.0158221535 0.0160475411 0.0386071131 0.0291761588 0.0252072532 0.0256605465 0.0256617498 0.0223768 0.038468767 0 0 0]
   [0.0190852191 0.017823698 0.0184199046 0.0421851613 0.0281776283 0.0311327633 0.0305566732 0.0347953215 0.0169057846 0.0238251612 0.0309104025 0.0287685432 0.0205631815 0.0405168086 0.019256575 0.0324270763 0.0352029912 0.0210147724 0.0201224629 0.0317579135 0.0441795103 0.0273807291 0.0244887341 0.0246819742 0.0193128865 0.0291317794 0.0208125859 0.0171764176 0.0285219904 0.0206208434 0.0260140449 0.0221911613 0.0272340234 0.0315129086 0.0317835063 0.0192475766 0.0245425403 0.0177187435 0 0 0]]]]
'''
#endregion

#endregion



#region ----------------------------------  7 ---------------------------------------
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
path_to_file = "./datasets/deu.txt"

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

for i in range(len(en)):
    en_encoded_temp = bpemb_en.encode_ids(en[i])
    de_encoded_temp = bpemb_de.encode_ids(de[i])

    if (len(en_encoded_temp)<=40) and (len(de_encoded_temp)<=40):
        en_encoded.append([10000] + en_encoded_temp + [10001])
        de_encoded.append([10000] + de_encoded_temp + [10001])
    
en_padded = tf.keras.preprocessing.sequence.pad_sequences(en_encoded, padding='post')
de_padded = tf.keras.preprocessing.sequence.pad_sequences(de_encoded, padding='post')
def create_padding_mask(seq):
  seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

  # add extra dimensions to add the padding
  # to the attention logits.
  return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

def create_look_ahead_mask(size):
  mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
  return mask  # (seq_len, seq_len)

def create_masks(inp, tar):
  # Encoder padding mask
  enc_padding_mask = create_padding_mask(inp)

  # Used in the 2nd attention block in the decoder.
  # This padding mask is used to mask the encoder outputs.
  dec_padding_mask = create_padding_mask(inp)

  # Used in the 1st attention block in the decoder.
  # It is used to pad and mask future tokens in the input received by 
  # the decoder.
  look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
  dec_target_padding_mask = create_padding_mask(tar)
  combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

  return enc_padding_mask, combined_mask, dec_padding_mask
en_sample_batch, de_sample_batch = np.stack((en_padded[0], en_padded[-1])), np.stack((de_padded[0], de_padded[-1]))
de_sample_batch
'''
array([[10000,  4766,  9935, 10001,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0],
       [10000,  3077,  2557,  9915,  1682,  9940,  2468,  2259,   344,
           10,  2161,   148,  9937,   105,  4001,  9940,  2377,   148,
         1715,   713,   443,   200,  2229,  8612,  9940,   369,   178,
         4416,    89,    28,  2648,  9935, 10001,     0,     0,     0,
            0,     0,     0,     0,     0,     0]], dtype=int32)'''
enc_padding_mask, combined_mask, dec_padding_mask = create_masks(en_sample_batch, de_sample_batch)
# Importantly enc_padding_mask and dec_padding_mask have the same shape. 
enc_padding_mask.shape, dec_padding_mask.shape
# (TensorShape([2, 1, 1, 41]), TensorShape([2, 1, 1, 41]))

# When the input target sentences in the batch are like above, 
# the resulting look ahead mask looks like stairs. 
# You can see that the number the steps of the "stairs"  depends 
# on the number of non-zero elements of the inputs. 
tf.print(combined_mask, summarize=-1)

#region output
'''
[[[[0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]
   [0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]
   [0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]
   [0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]
   [0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]
   [0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]
   [0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]
   [0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]
   [0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]
   [0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]
   [0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]
   [0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]
   [0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]
   [0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]
   [0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]
   [0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]
   [0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]
   [0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]
   [0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]
   [0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]
   [0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]
   [0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]
   [0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]
   [0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]
   [0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]
   [0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]
   [0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]
   [0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]
   [0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]
   [0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]
   [0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]
   [0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]
   [0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]
   [0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]
   [0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]
   [0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]
   [0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]
   [0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]
   [0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]
   [0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]
   [0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]
   [0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]]]


 [[[0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]
   [0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]
   [0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]
   [0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]
   [0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]
   [0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]
   [0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]
   [0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]
   [0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]
   [0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]
   [0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]
   [0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]
   [0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]
   [0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]
   [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]
   [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]
   [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]
   [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]
   [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]
   [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]
   [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]
   [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]
   [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]
   [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]
   [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]
   [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]
   [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]
   [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1]
   [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1]
   [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1]
   [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1]
   [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1]
   [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1]
   [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1]
   [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1]
   [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1]
   [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1]
   [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1]
   [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1]
   [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1]
   [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1]
   [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1]]]]
'''
#endregion

#endregion




#----------------------------------  xxxxxxx ---------------------------------------

#----------------------------------  xxxxxxx ---------------------------------------

#----------------------------------  xxxxxxx ---------------------------------------

#----------------------------------  xxxxxxx ---------------------------------------

#----------------------------------  xxxxxxx ---------------------------------------

#----------------------------------  xxxxxxx ---------------------------------------


#region ----------------------------------  translator_wp_8.ipynb  ---------------------------------------
def evaluate(inp_sentence):
  '''
  The <start> token is [10000], 
  and the <end> token is [10001].
  '''
  start_token = [vocab_size]
  end_token = [vocab_size + 1]

    '''
    You first encode an input sentences into a tensor with integers. 
    '''
  inp_sentence = start_token + bpemb_1.encode_ids(inp_sentence) + end_token
  encoder_input = tf.expand_dims(inp_sentence, 0)

  '''
  The translated output is first [10000], which means <start>
  '''
  decoder_input = [vocab_size]
  output = tf.expand_dims(decoder_input, 0)

  '''
  In this loop, for MAX_LENGTH times at most, you repeat 10002-class classification, 
  that is , you choose one word every loop and append it to the 'output'.
  When you finish MAX_LENGTH loops, or you choose [10001] as a classification result, 
  which means <end> token, you stop decoding. 
  '''
    
  '''
  During training Transformer-based translators, you put in the whole target sentences, 
  but you need to simualte this loop even during training. 
  That is why you need look ahead mask during training to hide the upcoming tokens
  which are not decoded yet.
  '''
  for i in range(MAX_LENGTH):
    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
        encoder_input, output)

    # predictions.shape == (batch_size, seq_len, vocab_size)
    predictions, attention_weights_encoder, attention_weights_decoder = transformer(encoder_input, 
                                                 output,
                                                 False,
                                                 enc_padding_mask,
                                                 combined_mask,
                                                 dec_padding_mask)

    # select the last word from the seq_len dimension
    predictions = predictions[: ,-1:, :]  # (batch_size, 1, vocab_size)

    predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

    # return the result if the predicted_id is equal to the end token
    if predicted_id == vocab_size+1:
      return tf.squeeze(output, axis=0), attention_weights_encoder, attention_weights_decoder

    # concatentate the predicted_id to the output which is given to the decoder
    # as its input.
    output = tf.concat([output, predicted_id], axis=-1)

  return tf.squeeze(output, axis=0), attention_weights_encoder, attention_weights_decoder


#endregion

#region ----------------------------------translator_wp_evaluation.ipynb---------------------------------------

translate("I'm the only one who cannot speak German in the office.", plot='encoder_layer4_block', img_name='sample_img_1')

# As I mentioned in the fourth article, the last layer seems to focus on the beginnign and the end of sentences. 
translate("He admired her as a great director.", plot='encoder_layer4_block', img_name='sample_img_1')

# The word alighments in English and German. 
translate("He admired her as a great director.", plot='decoder_layer4_block2', img_name='sample_img_3')


# With a bit more complicated input, the word alighments get more blurly. 
translate("I went to the shop to buy something to drink.", plot='decoder_layer4_block2', img_name='sample_img_3')




# With this translator, you can translate deadly funny jokes into German. 
translate("He will have written the funniest joke in the world, and as a consequence he will die, laughing.", plot='decoder_layer4_block1', img_name='sample_img_3')




#endregion







