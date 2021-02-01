# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 13:27:43 2020

@author: ACER
"""
import os
#import pandas as pd
import numpy as np
import cv2
#from PIL import Image
import random
#import tensorflow as tf
#import re
#import datetime
#import io
#from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
#import pickle
#from sklearn.model_selection import train_test_split
#import string
#from utils import generate_token_index
#from utils import score_prediction, generate_token_index, y_labels, generate_dataset
import json
import tensorflow as tf
from tensorflow import keras
import string
import yaml

from pre_proc import read_image_BW, read_image_color, normalize_0_mean_1_variance_BW, normalize_0_mean_1_variance_color

with open('config.yml') as f:
    config = yaml.load(f)
    
#config['images_folder'] = os.path.join('../', config['images_folder'])
#config['labels_file'] = os.path.join('../', config['labels_file'])

with open(config['labels_file']) as f:
    dataset = json.load(f)
    
fraction = 0.1
max_len = int(len(dataset['train']) * fraction)
indices = np.random.randint(0, len(dataset['train']), max_len)
dataset['train_subsampled'] = [dataset['train'][j] for j in indices]

print(len(dataset['train_subsampled']))
print(dataset['val'][1])

from DataGen import DataGenerator
train_generator = DataGenerator(config, dataset['train_subsampled'], shuffle=True, use_data_augmentation=False)
val_generator = DataGenerator(config, dataset['val'], shuffle=True, use_data_augmentation=False)


y_size = config['image']['image_size']['y_size']
x_size = config['image']['image_size']['x_size']
max_seq_length = config['network']['max_seq_lenght']
latent_dim = config['network']['latent_dim']
num_decoder_tokens = train_generator.num_decoder_tokens

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, LSTM, Dense, TimeDistributed, Conv2D, MaxPooling2D, Reshape, Dropout, BatchNormalization, Activation, Bidirectional, concatenate, add, Lambda, Permute
from tensorflow.keras.callbacks import EarlyStopping
#import keras.backend as K
from tensorflow.keras.optimizers import Adam

optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

encoder_inputs = Input(shape=(y_size, x_size, 1), name='input_encoder')

#encoder_inputs = Input(shape=(128, 384, 1), name='input_encoder')

#encoder_inputs = Input(shape=(92, 248, 1), name='input_encoder')

#(7,7)
encoder_layer = Conv2D(16, (7, 7), strides=(1,1), padding='same', use_bias=False, name='conv_1')(encoder_inputs)
encoder_layer = BatchNormalization(name='batch_norm_1')(encoder_layer)
encoder_layer = Activation('relu', name='activation_1')(encoder_layer)
encoder_layer = MaxPooling2D(pool_size=(2, 2), padding='valid', name='maxpool_1')(encoder_layer)

#(7,7)
encoder_layer = Conv2D(32, (5, 5), padding='same', use_bias=False, name='conv_2')(encoder_layer)
encoder_layer = BatchNormalization(name='batch_norm_2')(encoder_layer)
encoder_layer = Activation('relu', name='activation_2')(encoder_layer)
encoder_layer = MaxPooling2D(pool_size=(2, 2), padding='valid', name='maxpool_2')(encoder_layer)

#(5,5)
encoder_layer = Conv2D(64, (5, 5), padding='same', use_bias=False, name='conv_3')(encoder_layer)
encoder_layer = BatchNormalization(name='batch_norm_3')(encoder_layer)
encoder_layer = Activation('relu', name='activation_3')(encoder_layer)
encoder_layer = MaxPooling2D(pool_size=(2, 2), padding='valid', name='maxpool_3')(encoder_layer)

#(5,5)
encoder_layer = Conv2D(128, (3, 3), padding='same', use_bias=False, name='conv_4')(encoder_layer)
encoder_layer = BatchNormalization(name='batch_norm_4')(encoder_layer)
encoder_layer = Activation('relu', name='activation_4')(encoder_layer)
encoder_layer = MaxPooling2D(pool_size=(2, 2), padding='valid', name='maxpool_4')(encoder_layer)

#(3,3)
encoder_layer = Conv2D(256, (3, 3), padding='same', use_bias=False, name='conv_5')(encoder_layer)
encoder_layer = BatchNormalization(name='batch_norm_5')(encoder_layer)
encoder_layer = Activation('relu', name='activation_5')(encoder_layer)
encoder_layer = MaxPooling2D(pool_size=(2, 2), padding='valid', name='maxpool_5')(encoder_layer)

#(3,3)
encoder_layer = Conv2D(512, (3, 3), padding='same', use_bias=False, name='conv_6')(encoder_layer)
encoder_layer = BatchNormalization(name='batch_norm_6')(encoder_layer)
encoder_layer = Activation('relu', name='activation_6')(encoder_layer)
encoder_layer = MaxPooling2D(pool_size=(2, 2), padding='valid', name='maxpool_6')(encoder_layer)

conv_shapes = encoder_layer.shape[1:]
timesteps = int(conv_shapes[0]*conv_shapes[1])
num_features = int(conv_shapes[2])

encoder_layer = Reshape((-1, num_features), name='reshape')(encoder_layer)

#encoder
encoder = LSTM(latent_dim, return_state=True, name='lstm_encoder')
_, state_h, state_c = encoder(encoder_layer)
encoder_states = [state_h, state_c]

#decoder
decoder_inputs = Input(shape=(max_seq_length, num_decoder_tokens), name='input_decoder_teacher_forcing')
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True, name='lstm_decoder')
decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)

#decoder_dense_time_dist = TimeDistributed(Dense(num_decoder_tokens, activation='softmax'), name='time_distributed_dense')
#decoder_outputs = decoder_dense_time_dist(decoder_outputs)

decoder_dense = Dense(num_decoder_tokens, activation='softmax', name='dense')
decoder_outputs = decoder_dense(decoder_outputs)


model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.summary()


train = True

#batch_size = 128  # Batch size for training.
epochs = 50 # Number of epochs to train for.

# Early stopping  
early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='auto')

if train == True:    
    # Run training
    model.compile(optimizer=optimizer, loss='categorical_crossentropy')
    #model.fit_generator(generator = generator epochs=epochs, verbose=1)

    model.fit_generator(generator=train_generator, validation_data=val_generator, epochs=epochs, verbose=1, 
                        callbacks=[])
    
if train == True:
    model_json = model.to_json()
    with open("graph.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights("weights.h5")
    
graph_path = 'graph.json'
weights_path = 'weights.h5'

from tensorflow.keras.models import model_from_json
with open(graph_path, 'r') as json_file:
    loaded_model_json = json_file.read()
model = model_from_json(loaded_model_json)

# load weights into new model
model.load_weights(weights_path)

images_folder = 'datasets/words/'
test_generator = DataGenerator(config, dataset['test'], shuffle=False, use_data_augmentation=False)

images_test, labels_test = test_generator.get_full_dataset()

index = random.randint(0, images_test.shape[0] - 1)
plt.imshow(np.reshape(images_test[index, :, :,:], (test_generator.y_size, test_generator.x_size)), cmap=plt.get_cmap('gray'))
plt.show()
print("label: ", labels_test[index])

images_test.shape

encoder_inference = Model(model.get_input_at(0)[0], model.get_layer("lstm_encoder").output[1:])

#decoder
decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_inputs_inference = Input(shape=(1, num_decoder_tokens), name='input_decoder_inference')

decoder_lstm = model.get_layer("lstm_decoder")
decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs_inference, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]

#dense layer
decoder_dense = model.get_layer('dense')
decoder_outputs = decoder_dense(decoder_outputs)

#inference
decoder_inference = Model([decoder_inputs_inference] + decoder_states_inputs, [decoder_outputs] + decoder_states)

def decode_sequence(input_seq, batch_dim, num_decoder_tokens, max_decoder_seq_length, token_indices, reverse_token_indices):
    # Encode the input as state vectors.
    states_value = encoder_inference.predict(input_seq, batch_size = batch_dim)

    full_seq = np.zeros((batch_dim, 1, num_decoder_tokens))

    target_seq = np.zeros((batch_dim, num_decoder_tokens))
    # Populate the first character of target sequence with the start character.
    target_seq[:, token_indices['[']] = 1.
   
    for i in range(max_decoder_seq_length):
        output_tokens, h, c = decoder_inference.predict([np.expand_dims(target_seq, axis=1)] + states_value, 
                                                        batch_size = batch_dim )
    
        # Sample a token
        sampled_token_index = np.argmax(output_tokens[:, -1, :], axis=1)
        #        confidence.append(np.max(output_tokens[:, -1, :]))

        target_seq = np.zeros((batch_dim, num_decoder_tokens))    
        for j in range(batch_dim):
            target_seq[j, sampled_token_index[j]] = 1.

        states_value = [h, c]

        #concatenate with the full sequence array
        full_seq = np.concatenate((full_seq, np.expand_dims(target_seq, axis=1)), axis=1)

    #remove first time element (is empty)
    full_seq = full_seq[:, 1:, :]
    decoded_sentences = []
    
    for i in range(batch_dim):
        sentence = []
        for j in range(full_seq.shape[1]):
            sampled_token_index = np.argmax(full_seq[i, j, :])   
            sentence.append(reverse_token_indices[sampled_token_index])
            #print(sampled_token_index)
    
        decoded_sentences.append(''.join(sentence).replace("]", ""))  
            
    return decoded_sentences

def predict(images, batch_size = 512):
    n_images = images.shape[0]
    y_size = images.shape[1]
    x_size = images.shape[2]

    n_batches = (n_images + batch_size - 1) // batch_size
    
    output_list = []
    
    for i in range(n_batches):

#    for i in range(1):

        batch_in, batch_out = (batch_size)* i, (batch_size)* i + batch_size

        if batch_out >= n_images:
            batch_out = n_images

        input_seq = images[batch_in:batch_out, :, :, :]
        batch_dim = batch_out - batch_in
        decoded_sentences = decode_sequence(input_seq, batch_dim, test_generator.num_decoder_tokens, 
                                          test_generator.max_seq_length,
                                          test_generator.token_indices,
                                          test_generator.reverse_token_indices)

        output_list.append(decoded_sentences)
    
    #flatten list
    flattened_list = [item for sublist in output_list for item in sublist]
    
    return flattened_list

pred_test = predict(images_test)
pred_test[index]

test_img = 'datasets/b02/b02-013/b02-013-00-00.png'

def read_image_BW_single(filename, y_size, x_size):

    fpath = filename
    img = cv2.imread(r'test2.png',0)
    
    if img is not None: 
        (wt, ht) = x_size, y_size
        (h, w) = img.shape
        fx = w / wt
        fy = h / ht
        f = max(fx, fy)
        newSize = (max(min(wt, int(w / f)), 1), max(min(ht, int(h / f)), 1)) # scale according to f (result at least 1 and at most wt or ht)
        img = cv2.resize(img, newSize)
        target = np.ones([ht, wt]) * 255
        target[0:newSize[1], 0:newSize[0]] = img
        img = target          
    else:
        img = np.zeros((x_size, y_size)) 
            
    return img 

t_img = read_image_BW_single(test_img, y_size, x_size)
t_img = normalize_0_mean_1_variance_BW(t_img, y_size, x_size)
t_img = np.expand_dims(t_img, axis = 0)

o = predict(t_img,batch_size = 1)

from spellchecker import SpellChecker

spell = SpellChecker()

misspelled = spell.unknown(o)

for word in misspelled:
    # Get the one `most likely` answer
    print(spell.correction(word))

    # Get a list of `likely` options
    print(spell.candidates(word))




