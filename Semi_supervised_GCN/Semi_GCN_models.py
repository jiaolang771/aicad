# -*- coding: utf-8 -*-

from keras.models import Input, Model, Sequential
from keras.layers import Dense, Activation, Dropout, Lambda, BatchNormalization
import keras.backend as K
import keras
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from keras.utils import to_categorical
from keras_dgl.utils import *
from keras_dgl.layers import MultiGraphCNN
from keras.optimizers import SGD, Adam
from keras.regularizers import l2
#from sklearn.model_selection import StratifiedKFold
from keras_dgl.layers import GraphCNN
from sklearn.metrics import roc_curve, auc


def get_semi_gcn(num_feature, num_filters, graph_conv_filters, num_class):
    
    model = Sequential()
#    model.add(Dropout(0.3)) 
    
    
    model.add(GraphCNN(64, num_filters, 
                        graph_conv_filters, 
                        input_shape=(num_feature,), 
                        kernel_regularizer=l2(5e-4)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.2))    
    
    
    model.add(GraphCNN(128, num_filters, 
                        graph_conv_filters, 
                        kernel_regularizer=l2(5e-4)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
##    
##    
    # model.add(GraphCNN(32, num_filters, 
    #                    graph_conv_filters, 
    #                    kernel_regularizer=l2(5e-3)))
    # model.add(BatchNormalization())
    # model.add(Activation('relu'))
    # model.add(Dropout(0.2))
##    
##    
    model.add(GraphCNN(2, num_filters, 
                       graph_conv_filters, 
                       kernel_regularizer=l2(5e-4)))
    model.add(BatchNormalization())
    model.add(Activation('softmax'))

    

    model.compile(loss='categorical_crossentropy', 
                  optimizer=Adam(lr = 0.001,
                                     beta_1=0.9,
                                     beta_2=0.999,
                                     epsilon=1e-04))

    
    return model



