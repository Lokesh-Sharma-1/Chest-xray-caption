import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt
import re
import collections
import random
import numpy as np
import pandas as pd
import os
import time
import json
from PIL import Image
import time
st.set_option('deprecation.showPyplotGlobalUse', False)

st.header('Medical Report Generation Using Chest X-ray images')
from PIL import Image
image = Image.open('/home/lokesh/Desktop/image.jpeg')
st.image(image, caption='credit:Photo by CDC on Unsplash',width=700)

st.markdown('This web app generates medical report for Chest X-ray image. Upload a chest X-ray image below and the report will be generated automatically.')

#parameters
embedding_dim = 256
units = 1024
features_shape = 1024
attention_features_shape = 49
max_length = 100

class Attention(tf.keras.Model):
    '''This class implements attention mechanism.'''
    def __init__(self, units):
        super(Attention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, features, hidden):
        hidden_with_time_axis = tf.expand_dims(hidden, 1)

        attention_hidden_layer = (tf.nn.tanh(self.W1(features) +
                                         self.W2(hidden_with_time_axis)))

        score = self.V(attention_hidden_layer)

        attention_weights = tf.nn.softmax(score, axis=1)

        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights

class CNN_Encoder(tf.keras.Model):
    '''This class implement Encoder for image.'''
    def __init__(self, embedding_dim):
        super(CNN_Encoder, self).__init__()
        self.fc = tf.keras.layers.Dense(embedding_dim)

    def call(self, x):
        x = self.fc(x)
        x = tf.nn.relu(x)
        return x

class RNN_Decoder(tf.keras.Model):
    '''This class implement decoder with attention mechanism using recurrent neural network.'''
    def __init__(self, embedding_dim, units, vocab_size):
        super(RNN_Decoder, self).__init__()
        self.units = units

        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')
        self.fc1 = tf.keras.layers.Dense(self.units)
        self.fc2 = tf.keras.layers.Dense(vocab_size)

        self.attention = Attention(self.units)

    def call(self, x, features, hidden):
        context_vector, attention_weights = self.attention(features, hidden)

        x = self.embedding(x)

        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        output, state = self.gru(x)

        x = self.fc1(output)

        x = tf.reshape(x, (-1, x.shape[2]))

        x = self.fc2(x)

        return x, state, attention_weights

    def reset_state(self, batch_size):
        return tf.zeros((batch_size, self.units))

def load_image(img):
    '''This function performs the preprocessing on the image.'''
    img = tf.image.resize(img, (224, 224))
    img = tf.keras.applications.densenet.preprocess_input(img)
    return img

def evaluate(image):
    '''This function predicts the report and calucaltes the attention'''
    attention_plot = np.zeros((max_length, attention_features_shape))

    hidden = decoder.reset_state(batch_size=1)

    temp_input = tf.expand_dims(image, 0)
    img_tensor_val = image_features_extract_model(temp_input)
    img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0],
                                                 -1,
                                                 img_tensor_val.shape[3]))

    features = encoder(img_tensor_val)

    dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)
    result = []

    for i in range(max_length):
        predictions, hidden, attention_weights = decoder(dec_input,
                                                         features,
                                                         hidden)

        attention_plot[i] = tf.reshape(attention_weights, (-1, )).numpy()

        predicted_id = tf.random.categorical(predictions, 1)[0][0].numpy()
        predicted_word = tf.compat.as_text(tokenizer.index_word[predicted_id])
        result.append(predicted_word)

        if predicted_word == '<end>':
            return result, attention_plot

        dec_input = tf.expand_dims([predicted_id], 0)

    attention_plot = attention_plot[:len(result), :]
    return result, attention_plot

def plot_attention(image, result, attention_plot):
    '''This function plots the attention on the image.'''
    temp_image = np.array(image)

    fig = plt.figure(figsize=(10, 10))

    len_result = len(result)
    for i in range(len_result):
        temp_att = np.resize(attention_plot[i], (8, 8))
        grid_size = max(int(np.ceil(len_result/2)), 2)
        ax = fig.add_subplot(grid_size, grid_size, i+1)
        ax.set_title(result[i])
        img = ax.imshow(temp_image)
        ax.imshow(temp_att, cmap='gray', alpha=0.6, extent=img.get_extent())
    st.pyplot()

def final(X):
    '''This function calculates the predicted report and attention plot.'''
    #predicting report
    start = time.time()
    predicted_report, attention_plot = evaluate(X)
    for val in predicted_report:
        if val == "<unk>" or val == "<end>":
            predicted_report.remove(val)
    end = time.time()
    st.write(f"Time : {end-start} seconds")
    return predicted_report, attention_plot

def image_extract_model():
	#creating image_features_extract_model
    chexnet_weights = "brucechou1983_CheXNet_Keras_0.3.0_weights.h5"
    image_model = tf.keras.applications.densenet.DenseNet121(include_top=True,classes=14,weights=chexnet_weights)
    new_input = image_model.input
    hidden_layer = image_model.layers[-3].output
    model = tf.keras.Model(new_input, hidden_layer)
    return model

import dill
if __name__ == '__main__':
    
    #loading encoder, decoder and tokenizer using dill library
    encoder = dill.load(open('encoder.pkl', 'rb'))
    decoder = dill.load(open('decoder.pkl', 'rb'))
    tokenizer = dill.load(open('tokenizer.pkl', 'rb'))
    
    image_features_extract_model = image_extract_model()

file = st.file_uploader("Select File:", type=["png","jpg","jpeg"])
if file is not None:
    org_img = Image.open(file)
    img = np.array(org_img)
    img = load_image(img)
    y_pred, attention_plot = final(img)
    st.write('Generated Report :', ' '.join(y_pred))
    plot_attention(org_img, y_pred, attention_plot)
