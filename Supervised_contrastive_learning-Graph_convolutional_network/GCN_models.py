# -*- coding: utf-8 -*-

from keras.models import Model
from keras.layers import Input, Concatenate
from keras.layers import Dense, Activation, Dropout, Lambda, BatchNormalization
from keras.regularizers import l2
import keras.backend as K
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from keras.utils import to_categorical
from keras_dgl.utils import *
from keras_dgl.layers import MultiGraphCNN, MultiGraphAttentionCNN
from keras.optimizers import SGD, Adam






def get_gcn_encoder(num_node, num_feature, num_filters, out_dim, num_class):
    
    weight_initialier = "he_normal"
    Feature_input = Input(shape=(num_node, num_feature), name='node_feature')
    Graph_input = Input(shape=(num_node*num_filters, num_node), name='graph')
    ## layer 1
    output = MultiGraphCNN(out_dim, num_filters,
                           kernel_initializer=weight_initialier)([Feature_input, Graph_input])
    output = BatchNormalization()(output)
    output = Activation('elu')(output)
    output = Dropout(0.2)(output)
    
    ## layer 2
    output = MultiGraphCNN(out_dim, num_filters, 
                           kernel_initializer=weight_initialier)([output, Graph_input])
    output = BatchNormalization()(output)
    output = Activation('elu')(output)
    output = Dropout(0.2, name='Final_Encoder_Dropout')(output)
    
    ## layer 3
    # output = MultiGraphCNN(out_dim, num_filters, 
    #                        kernel_initializer=weight_initialier)([output, Graph_input])
    # output = BatchNormalization()(output)
    # output = Activation('elu')(output)
    # output = Dropout(0.2)(output)
    
    ## embedding layer Pool 
    embeddings = Lambda(lambda x: K.mean(x, axis=2), name='Pool')(output)  # adding a node invariant layer to make sure output does not depends upon the node order in a graph.
    
    # initializer = tf.keras.initializers.Zeros()
    embeddings = Dense(8, kernel_initializer=weight_initialier)(embeddings)
    embeddings = BatchNormalization()(embeddings)
    embeddings = Activation('relu')(embeddings)
    embeddings = Dropout(0.2)(embeddings)
    
    model = Model(inputs=[Feature_input, Graph_input], 
                  outputs=embeddings,
                  name='GCN_encoder')
    return model



##   
def get_gcn_classifier(encoder,num_node, num_feature, num_filters, num_class, trainable=True):
     
    encoder_trim = Model(inputs=[encoder.layers[0].get_input_at(0), 
                                 encoder.layers[1].get_input_at(0)],
                                 outputs = encoder.get_layer('Final_Encoder_Dropout').output,
                                name='GCN_encoder_trim')

    for layer in encoder_trim.layers:
          layer.trainable = trainable
    
    Feature_input = Input(shape=(num_node, num_feature), name='node_feature')
    Graph_input = Input(shape=(num_node*num_filters, num_node), name='graph')
            
    embeddings_trim = encoder_trim([Feature_input, Graph_input])
    
    
    ## embedding layer Pool 
    embeddings = Lambda(lambda x: K.mean(x, axis=2), name='Pool')(embeddings_trim)
    
    ## dense layer 1 
    output = Dense(8, name='Dense')(embeddings)
    output = BatchNormalization()(output)
    output = Activation('relu')(output)
    output = Dropout(0.2)(output)
    
    ## dense layer 2
    # output = Dense(8)(output)
    # output = BatchNormalization()(output)
    # output = Activation('relu')(output)
    # output = Dropout(0.2)(output)
    
    ##  output layer
    output = Dense(num_class)(output)
    output = Activation('softmax')(output)
    
    model = Model(inputs=[Feature_input, Graph_input], 
                  outputs=output,
                  name='GCN_downstream_classifier')
    return model


##



##   
def get_gcn(num_node, num_feature, num_filters, out_dim, num_class):
    
    Feature_input = Input(shape=(num_node, num_feature))
    Graph_input = Input(shape=(num_node*num_filters, num_node))
    
    output = MultiGraphCNN(out_dim, num_filters)([Feature_input, Graph_input])
    output = BatchNormalization()(output)
    output = Activation('elu')(output)
    output = Dropout(0.2)(output)
    
    output = MultiGraphCNN(out_dim, num_filters)([output, Graph_input])
    output = BatchNormalization()(output)
    output = Activation('elu')(output)
    output = Dropout(0.2)(output)
    

    output = Lambda(lambda x: K.mean(x, axis=2), name='Pool')(output)  # adding a node invariant layer to make sure output does not depends upon the node order in a graph.

    # project into low-dimensional embeddings
    output = Dense(16)(output)
    output = BatchNormalization()(output)
    output = Activation('elu')(output)
    output = Dropout(0.2)(output)
    
    output = Dense(num_class)(output)
    output = Activation('softmax')(output)
    
    model = Model(inputs=[Feature_input, Graph_input], outputs=output)
    
    # sgd = SGD(lr=0.001,momentum=0.9, nesterov=True)
    adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999)
    model.compile(optimizer=adam,
                  loss='categorical_crossentropy',
                  metrics=['acc'])
    ##            metrics=['acc',tf.keras.metrics.AUC()]
    return model

