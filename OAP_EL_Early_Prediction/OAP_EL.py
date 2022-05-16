# -*- coding: utf-8 -*-
"""
@author: ZhiyuanLi
"""

import numpy as np
import tensorflow as tf
import pandas as pd
import collections
from tensorflow.keras import Sequential, layers, optimizers
from xgboost import XGBClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from statistics import mean, stdev
from sklearn.cluster import SpectralClustering

# load data
df = pd.read_csv("...file...")

# create an adjacent matrix
def adjacent_matrix(d):
    # define a zero matrix
    mat = np.zeros((338,338))
    m,n = mat.shape
    
    # unique attributes
    a1 = list(d['x1'].unique())

    a2 = list(d['x2'].unique())
    a2 = [x for x in a2 if x == x]
    
    # attributes
    for i in a1:
        # get index 
        pos = d.index[df['x1'] == i].tolist()
        #print(pos)
        for x in pos:
            for y in pos:
                mat[x][y] = 1
    
    for i in a2:
        # get index 
        pos = d.index[df['x2'] == i].tolist()
        #print(pos)
        for x in pos:
            for y in pos:
                mat[x][y] = 1

    return mat

# using synthetic data, suppose data size: 338 x 10 just for simple run
# experiment setup only using small size data for convenience 
np.random.seed(123)

input_len = 338
sample_size = 10
X = np.random.rand(sample_size,input_len)
# simulate label using binomial distribution
y = np.random.binomial(1,0.5,sample_size)

# get adjacency mat
mat = adjacent_matrix(df)

# define base-classifiers
def base_classifier(u,v,k):
    M = {}
    for i in range(k):
        M[i] = XGBClassifier(
                objective='binary:logistic',
                use_label_encoder=False,
                verbosity = 0, 
                scale_pos_weight=1,
                max_depth = u,
                reg_lambda = v)
    return M

# define a meta-classifier
def meta_classifier(k):
    # build a neural net for probs input
    neuralNet = Sequential([layers.Dense(k, activation = tf.nn.relu),
                            layers.Dense(1, activation = tf.nn.sigmoid)])
    
    neuralNet.build(input_shape=(None,k))
    return neuralNet

# define OAP_EL 
def OAP_EL(X,y,mat,u,v,k,s):
    # repeat loocv
    tf.random.set_seed(s)
    
    # graph clustering
    sc = SpectralClustering(k, affinity='precomputed', n_init=100, random_state = 12345)
    sc.fit(mat)
                    
    # sort the features based on labels
    labels_clust = sc.labels_
    temp = np.append(X, [labels_clust], axis = 0)
    X = X[:,temp[temp.shape[0]-1, :].argsort()]
    
    # define k xgb models
    M = base_classifier(u,v,k)
                    
    # fit k xgb models
    freq = collections.Counter(labels_clust)
    f_n = 0
    probs_df = np.empty((len(y),1))
    
    for key, i in zip(freq, range(len(M))):
        v = freq[key]
        M[i].fit(X[:,f_n:f_n+v],y)
        
        # predicted probs
        probs = M[i].predict_proba(X[:,f_n:f_n+v])[:,1].reshape(-1,1)
        probs_df = np.append(probs_df, probs, axis = 1)
        
        f_n = f_n+v
    probs_df = np.delete(probs_df, 0, 1)
    #print(probs_df.shape)
    
    # define meta-classifier as neural network
    net_probs = meta_classifier(k)
    optimizer_fusion = optimizers.Adam(learning_rate=1e-3)
    
    # epoch can be adjusted 
    for epoch in range(1000):
        with tf.GradientTape() as tape_probs:
            x_train = tf.cast(probs_df, dtype = tf.float32)
            y_train = tf.cast(y, dtype = tf.float32)
            y_train = tf.reshape(y_train, (len(y_train),1))
            logits = net_probs(x_train)
            loss = tf.reduce_mean(tf.losses.binary_crossentropy(y_train, logits, from_logits=False))  
                
            # add regularization = []
            loss_reg = []
            for i,p in enumerate(net_probs.trainable_variables):
                if i % 2 == 0:
                    loss_reg.append(tf.nn.l2_loss(p))
            loss_regularization = tf.reduce_sum(tf.stack(loss_reg))

            # l2 norm loss
            loss = loss + 0.001*loss_regularization
                
        grads = tape_probs.gradient(loss, net_probs.trainable_variables)
        optimizer_fusion.apply_gradients(zip(grads, net_probs.trainable_variables))
            
        #if epoch % 200 == 0:
           # print('Epoch: {}, Loss: {}'.format(epoch, loss))
            
    return M, net_probs, labels_clust

# define main model
def train(s, X, y, mat):
    # define predicted label
    y_test_pred_all = []
    
    # define loocv
    loo = LeaveOneOut()
    
    # repeat loocv
    tf.random.set_seed(s)
    
    # define tuning paramaters for xgb
    # now is for testing, can be adjusted for those paramaters
    U = [2,4]
    W = [0.001]
    
    # (n-1) validation via loocv
    for train_index, test_index in loo.split(X,y): 
        X_train, X_test = X[train_index,:], X[test_index,:]
        y_train, y_test = y[train_index], y[test_index]
        
        # skip normalization/preprocessing 
        
        # define a dictionary for saving paramaters
        params = {}
        
        # tune 
        for k in range(2,4):
            for u in U:
                for w in W:
                    y_vali_pred_all = []
                    
                    for tune_index, vali_index in loo.split(X_train, y_train):
                        X_tune, X_vali = X_train[tune_index,:], X_train[vali_index,:]
                        y_tune, y_vali = y_train[tune_index], y_train[vali_index]
                        
                        # train OAP-EL
                        base, meta, labels_clust = OAP_EL(X_tune,y_tune,mat,u,w,k,s)
                        
                        # sort the features based on labels
                        temp = np.append(X_vali, [labels_clust], axis = 0)
                        X_vali = X_vali[:,temp[temp.shape[0]-1, :].argsort()]
                        
                        freq = collections.Counter(labels_clust)
                        f_n = 0
                        prob_vali_pred = np.empty((len(y_vali),1))
                        
                        for key, i in zip(freq, range(len(base))):
                            v = freq[key]
                            
                            # predicted probs
                            probs = base[i].predict_proba(X_vali[:,f_n:f_n+v])[:,1].reshape(-1,1)
                            #print(probs)
                            prob_vali_pred = np.append(prob_vali_pred, probs, axis = 1)
                            
                            f_n = f_n+v
                            
                        prob_vali_pred = np.delete(prob_vali_pred, 0, 1)
                        #print(k, u, w, prob_vali_pred)
                        
                        # get final predicted probs
                        y_vali_pred = meta(prob_vali_pred).numpy()
                        y_vali_pred[y_vali_pred>=0.5] = 1
                        y_vali_pred[y_vali_pred<0.5] = 0
                                
                        y_vali_pred_all.append(y_vali_pred[0][0])
                        #print(y_vali_pred_all)
                        
                        # since we use sythetic data, so roc_auc may not exist
                    try:
                        auc_vali = roc_auc_score(y_vali_pred_all, y_train)
                    except ValueError:
                        auc_vali = 0 # use 0 instead
    
                    params[auc_vali] = [k,u,w]

        #print(params)
        
        # pick the optimal params with highest auc
        optimal_params = max(params, key=params.get)
        k_hat = params[optimal_params][0]
        u_hat = params[optimal_params][1]
        w_hat = params[optimal_params][2]

        # fit OAP-EL based on best pamraters
        base, meta, labels_clust = OAP_EL(X_train,y_train,mat,u_hat,w_hat,k_hat,s)
        
        # sort the features based on labels
        temp = np.append(X_test, [labels_clust], axis = 0)
        X_test = X_test[:,temp[temp.shape[0]-1, :].argsort()]
                        
        freq = collections.Counter(labels_clust)
        f_n = 0
        prob_test_pred = np.empty((len(y_test),1))
        
        for key, i in zip(freq, range(len(base))):
            v = freq[key]
                            
            # predicted probs
            probs = base[i].predict_proba(X_test[:,f_n:f_n+v])[:,1].reshape(-1,1)
            #print(probs)
            prob_test_pred = np.append(prob_test_pred, probs, axis = 1)
            
            f_n = f_n+v
        prob_test_pred = np.delete(prob_test_pred, 0, 1)
   
        # get final predicted probs
        y_test_pred = meta(prob_test_pred).numpy()
        y_test_pred[y_test_pred>=0.5] = 1
        y_test_pred[y_test_pred<0.5] = 0
                                
        y_test_pred_all.append(y_test_pred[0][0])
        
    return y_test_pred_all

# compute prediction metrics with m replicates
def pred_metrics(m, y):
    acc = []
    recall = []
    speci = []
    auc = []
    
    # random seed
    seed = np.random.randint(10000, size=m)
    for s in seed:
        # get prediction
        y_pred = train(s, X, y, mat)

        # get confusion matrix
        cm = confusion_matrix(y_pred, y)  
        
        # compute metrics
        acc.append(round(accuracy_score(y, y_pred), 3))
        recall.append(round(cm[1,1]/(cm[0,1]+cm[1,1]), 3))
        speci.append(round(cm[0,0]/(cm[0,0]+cm[1,0]), 3))
        auc.append(round(roc_auc_score(y, y_pred), 3))
    
    print('Mean(SD) accuracy: {}({})'.format(mean(acc), stdev(acc)))
    print('Mean(SD) recall: {}({})'.format(mean(recall), stdev(recall)))
    print('Mean(SD) specificity: {}({})'.format(mean(speci), stdev(speci)))
    print('Mean(SD) AUC: {}({})'.format(mean(auc), stdev(auc)))

# evaluation
pred_metrics(2,y)  # suppose only 2 runs
