import imp
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


from b_transformer import Transformer
from c_encoder import Encoder
from d_decoder import Decoder
from utils import positional_encoding,create_padding_mask, create_look_ahead_mask, CustomSchedule, create_masks
from h_multihead import scaled_dot_product_attention,MultiHeadAttention,point_wise_feed_forward_network


if __name__=='__main__':
    bpemb_de = BPEmb(lang='de', vs=10000, dim=100)
    bpemb_en = BPEmb(lang='en', vs=10000, dim=100)
    path_to_file = "./data/cmn.txt"
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


    input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(en_padded, de_padded, test_size=0.2)


    BUFFER_SIZE = len(input_tensor_train)
    BATCH_SIZE = 64
    steps_per_epoch = len(input_tensor_train)//BATCH_SIZE
    # embedding_dim = 256
    units = 1024
    vocab_inp_size = 10000 + 2
    vocab_tar_size = 10000 + 2

    dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train)).shuffle(BUFFER_SIZE)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)


    num_layers = 4
    d_model = 128
    dff = 512
    num_heads = 8
    input_vocab_size = 10000 + 2
    target_vocab_size = 10000 + 2
    dropout_rate = 0.1
    EPOCHS = 30
    learning_rate = CustomSchedule(d_model)

    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

    def loss_function(real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = loss_object(real, pred)

        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask
        
        return tf.reduce_sum(loss_)/tf.reduce_sum(mask)



    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    transformer = Transformer(num_layers, d_model, num_heads, dff,
                            input_vocab_size, target_vocab_size, 
                            pe_input=input_vocab_size, 
                            pe_target=target_vocab_size,
                            rate=dropout_rate)



    checkpoint_path = "./checkpoints_deu/train"

    ckpt = tf.train.Checkpoint(transformer=transformer,optimizer=optimizer)

    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

    # if a checkpoint exists, restore the latest checkpoint.
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print ('Latest checkpoint restored!!')

    # The @tf.function trace-compiles train_step into a TF graph for faster
    # execution. The function specializes to the precise shape of the argument
    # tensors. To avoid re-tracing due to the variable sequence lengths or variable
    # batch sizes (the last batch is smaller), use input_signature to specify
    # more generic shapes.

    train_step_signature = [
        tf.TensorSpec(shape=(None, None), dtype=tf.int32),
        tf.TensorSpec(shape=(None, None), dtype=tf.int32),
    ]

    # @tf.function(input_signature=train_step_signature)
    def train_step(inp, tar):
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]
        
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)
        
        with tf.GradientTape() as tape:
            predictions, _ = transformer(inp, tar_inp, 
                                        True, 
                                        enc_padding_mask, 
                                        combined_mask, 
                                        dec_padding_mask)
            loss = loss_function(tar_real, predictions)

        gradients = tape.gradient(loss, transformer.trainable_variables)    
        optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))
        
        train_loss(loss)
        train_accuracy(tar_real, predictions)






tf.debugging.set_log_device_placement(True)
with tf.device('/GPU:0'):

    for epoch in range(EPOCHS):
        start = time.time()

        train_loss.reset_states()
        train_accuracy.reset_states()

        # inp -> portuguese, tar -> english
        for (batch, (inp, tar)) in enumerate(dataset.take(steps_per_epoch)):
            train_step(inp, tar)
        
            if batch % 50 == 0:
                print ('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1, batch, train_loss.result(), train_accuracy.result()))
        
        if (epoch + 1) % 5 == 0:
            ckpt_save_path = ckpt_manager.save()
            print ('Saving checkpoint for epoch {} at {}'.format(epoch+1,ckpt_save_path))
        
        print ('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1, 
                                                    train_loss.result(), 
                                                    train_accuracy.result()))

        print ('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))
