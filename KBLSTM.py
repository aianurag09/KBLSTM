import numpy as np
import pandas as pd
import cPickle
from collections import defaultdict
import re

from bs4 import BeautifulSoup

import sys
import os


os.environ['KERAS_BACKEND']='theano'
import keras
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

from keras.layers import Embedding
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding, Merge, Dropout, LSTM, GRU, Bidirectional, TimeDistributed
from keras.models import Model

from keras import backend as K
from keras.engine.topology import Layer, InputSpec
from keras import initializers
from itertools import izip

embedding_layer = Embedding(len(word_index) + 1,
							EMBEDDING_DIM,
							weights=[embedding_matrix],
							input_length=MAX_SENT_LENGTH,
							trainable=False)

sentence_input = Input((MAX_SENT_LENGTH,),name='sentence_input')

embedded_sequence = embedding_layer(sentence_input)

class AttentionWeightedAverage(Layer):
    def __init__(self, return_attention=False, **kwargs):
        self.init = initializers.get('uniform')
        self.supports_masking = True
        self.return_attention = return_attention
        super(AttentionWeightedAverage, self).__init__(** kwargs)
    def build(self, input_shape):
        self.input_spec = [InputSpec(ndim=3)]
        assert len(input_shape) == 3
        self.W = self.add_weight(shape=(input_shape[2], 1),
                                 name='{}_W'.format(self.name),
                                 initializer=self.init)
        self.trainable_weights = [self.W]
        super(AttentionWeightedAverage, self).build(input_shape)
    def call(self, x, mask=None):
        # computes a probability distribution over the timesteps
        # uses 'max trick' for numerical stability
        # reshape is done to avoid issue with Tensorflow
        # and 1-dimensional weights
        logits = K.dot(x, self.W)
        x_shape = K.shape(x)
        logits = K.reshape(logits, (x_shape[0], x_shape[1]))
        ai = K.exp(logits - K.max(logits, axis=-1, keepdims=True))
        # masked timesteps have zero weight
        if mask is not None:
            mask = K.cast(mask, K.floatx())
            ai = ai * mask
        att_weights = ai / (K.sum(ai, axis=1, keepdims=True) + K.epsilon())
        weighted_input = x * K.expand_dims(att_weights)
        result = K.sum(weighted_input, axis=1)
        if self.return_attention:
            return [result, att_weights]
        return result
    def get_output_shape_for(self, input_shape):
        return self.compute_output_shape(input_shape)
    def compute_output_shape(self, input_shape):
        output_len = input_shape[2]
        if self.return_attention:
            return [(input_shape[0], output_len), (input_shape[0], input_shape[1])]
        return (input_shape[0], output_len)
    def compute_mask(self, input, input_mask=None):
        if isinstance(input_mask, list):
            return [None] * len(input_mask)
        else:
            return None

HIDDEN_LAYER_DIMENSION = 256

state_vector = Bidirectional(LSTM(HIDDEN_LAYER_DIMENSION,return_sequences=True))(embedded_sequence)

attention_layer = AttentionWeightedAverage()(state_vector)

attention_layer = Dense(100,activation='relu')(attention_layer)

concept_layer = Input((64, 100,), name='concept_layer')

class KnowledgeLayer(Layer):
    def __init__(self, **kwargs):
        self.init = initializers.get('uniform')
        super(KnowledgeLayer, self).__init__(**kwargs)
    def build(self, input_shape):
        self.W = self.add_weight(shape=(100, 64),
                                 name='{}_W'.format(self.name),
                                 initializer=self.init)
        self.trainable_weights = [self.W]
        super(KnowledgeLayer, self).build(input_shape)  # Be sure to call this somewhere!
    def call(self, inputs,**kwargs):
        layer_output = inputs[0]
        concept_layer = inputs[1]
        weight_layer = K.dot(layer_output,self.W)
        a = K.repeat_elements(weight_layer,K.int_shape(concept_layer)[2],axis=1)
        shape = K.int_shape(weight_layer)
        b = K.reshape(a,(-1,shape[1],K.int_shape(concept_layer)[2]))
        alpha = K.sum(b * concept_layer, axis=2)
        att_weights = alpha / (K.sum(alpha, axis=1, keepdims=True))
        att_weights = K.expand_dims(att_weights,-1)
        knowledge_output = K.sum(att_weights * concept_layer, axis=1)
        return knowledge_output
    def compute_output_shape(self, input_shape):
        return (input_shape[0][0],100)


hidden = KnowledgeLayer()([attention_layer,concept_layer])

hidden = keras.layers.add([hidden,attention_layer])

preds = Dense(64,activation='softmax')(hidden)

Attention = Model([sentence_input,concept_layer],preds)

#sgd = SGD(lr=0.01,decay=1e-6,momentum=0.9,nesterov=True)

Attention.compile(loss='categorical_crossentropy',
           optimizer='adam')

