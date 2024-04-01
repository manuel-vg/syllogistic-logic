# import libraries
import os
import tensorflow as tf
from tensorflow.keras import Model ,Sequential
from tensorflow.keras.layers import Layer, Input, Dense, SimpleRNN, Reshape, Flatten, Conv1D, MaxPooling1D, MultiHeadAttention, LayerNormalization, Dropout, Embedding, Activation, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam, Adamax
from numpy.random import RandomState
from numpy import round
from json import dump
from evaluator import Evaluator

#################
# LAYER CLASSES #
#################
class RNNLayer(Layer):
    def __init__(self, units, activation):
        super().__init__()
        self.encoder = SimpleRNN(units=units, activation=activation, return_sequences=True) 
        self.decoder = SimpleRNN(units=units, activation=activation) 

    def call(self, inputs):
        x = self.encoder(inputs)
        return self.decoder(x)

class CNNLayer(Layer):
    def __init__(self, filters, kernel_size, activation, pool_size):
        super().__init__()
        self.convolution = Conv1D(filters=filters, kernel_size=kernel_size, activation=activation)
        self.maxpooling = MaxPooling1D(pool_size=pool_size)
        self.flatten = Flatten()

    def call(self, inputs):
        x = self.convolution(inputs)
        x = self.maxpooling(x)
        return self.flatten(x)

class TransformerLayer(Layer):
    def __init__(self, vocabulary_size, num_heads, ffn_units, ffn_activation, ffn_layers):
        super().__init__()
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=vocabulary_size)
        self.ffn = Sequential([Dense(units=ffn_units, activation=ffn_activation) for _ in range(ffn_layers)] + [Dense(vocabulary_size)]) # FFN with 3 + 1 layers
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(0.1)
        self.dropout2 = Dropout(0.1)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

class PositionalEncodingLayer(Layer):
    def __init__(self, tokens, vocabulary_size):
        super().__init__()
        self.token_emb = Activation('linear')
        self.pos_emb = Embedding(input_dim=tokens, output_dim=vocabulary_size)

    def call(self, x):
        tokens = tf.shape(x)[1] # number of tokens
        positions = tf.range(start=0, limit=tokens, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

######################
# NEURAL MODEL CLASS #
######################
class Neural_Model:
    def __init__(self, args, name, X_train, Y_train):    
        self.args = args                
        self.name = name
        self.X_train = X_train
        self.Y_train = Y_train
        self.model = None
        self.H = None    

    # SHUFFLE DATA FOR TRAINING
    def __permute(self, *arrays, random_seed=7):
        assert len(arrays) > 0
        length = len(arrays[0])
        perm = RandomState(random_seed).permutation(length)

        return [x[perm] for x in arrays]

    # BUILD THE MODEL
    def build(self, model_type, summary=False):
        N, tokens, vocabulary_size = self.X_train.shape        
        
        # BUILD LAYERS        
        inputs = Input(shape=self.X_train.shape[1:]) # input layer        

        if model_type == 'CNN':
            x = Reshape((int(tokens/3), vocabulary_size*3)) (inputs)
            x = CNNLayer(**self.args['layer_args']) (x)

        elif model_type == 'RNN':            
            x = Reshape((int(tokens/3), vocabulary_size*3)) (inputs)                        
            x = RNNLayer(**self.args['layer_args']) (x)                            

        elif model_type == 'MLP':
            x = Flatten() (inputs)
            x = Dense(**self.args['layer_args']) (x)      

        elif model_type == 'TRA':
            x = PositionalEncodingLayer(tokens, vocabulary_size) (inputs)
            x = TransformerLayer(vocabulary_size, **self.args['encoder_layer_args']) (x)
            x = GlobalAveragePooling1D() (x)
            x = Dropout(0.1) (x)
            x = Dense(**self.args['top_layer_args']) (x) # Top Layer
            x = Dropout(0.1) (x)

        outputs = Dense(self.Y_train.shape[1], activation="sigmoid") (x) # output layer (classifier)
        
        # CREATE MODEL
        self.model = Model(inputs=inputs, outputs=outputs, name=self.name)

        # SHOW MODEL INFO
        if summary:
            self.model.summary()        

        # COMPILE MODEL
        d = self.args['compile_args']
        optimizer, LR = d['optimizer'], d['learning_rate']        

        if optimizer == 'adamax':
            opt = Adamax(learning_rate=LR)
        else: 
            opt = Adam(learning_rate=LR)

        self.model.compile(loss=d['loss'], optimizer=opt, metrics=['accuracy']) 

    # FIT THE MODEL
    def fit(self, verbose=0):   
        # data shuffling
        self.X_train, self.Y_train = self.__permute(self.X_train, self.Y_train)     
        
        # data training
        self.H = self.model.fit(x=self.X_train, y=self.Y_train, verbose=verbose, **self.args['fit_args'])        
        
    # EVALUATE TEST DATA
    def evaluate(self, X_test, Y_test, by_length=False):
        ev = Evaluator(self.model, X_test, Y_test)
        if by_length:
            ev.evaluate_by_lenght()
        else:
            ev.evaluate()    