

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras.optimizers import SGD,Adam
from keras.layers import Input, Conv2D, Lambda, Dense, Flatten,MaxPooling2D, concatenate
from keras.layers import Dense, Activation, Dropout, Lambda, BatchNormalization
from keras.models import load_model, Model, clone_model
from keras.utils import to_categorical
from keras_dgl.utils import preprocess_adj_tensor
from sklearn.metrics import roc_curve, auc
from sklearn.utils import shuffle, class_weight
from sklearn.model_selection import StratifiedKFold, train_test_split
# laod and prepare data

# from bal_aug_matrix import bal_aug_func
from GCN_models import get_gcn, get_gcn_encoder, get_gcn_classifier


import random



import warnings

# import random as rn
warnings.simplefilter(action='ignore', category=FutureWarning)
K.clear_session()


def generate_random_adjacency_matrix(n, density=0.3):
    """
    Generates a random adjacency matrix for a graph.

    Parameters:
    - n: Number of nodes in the graph.
    - density: Density of the edges (probability of having an edge between two nodes).

    Returns:
    - A random adjacency matrix.
    """
    # Ensure the density is within the valid range [0, 1]
    density = max(0, min(1, density))

    # Generate a random matrix with values between 0 and 1
    random_matrix = np.random.rand(n, n)

    # Create an adjacency matrix based on the specified density
    adjacency_matrix = (random_matrix < density).astype(int)

    # Set the diagonal elements to 0 (no self-loops)
    np.fill_diagonal(adjacency_matrix, 0)

    return adjacency_matrix

def Graph_aug(feature, graph, label):
    print('Augmenting data...')
    aug_factor = 10
    percent = 0.01   
    num_mask_node = round(((graph.shape[1]**2)*percent)**0.5)
    
    feature_aug = feature  
    graph_aug = graph  
    label_aug = label 
    #np.zeros(aug_factor*label.shape[0], label.shape[1])
              
    for aug_i in range(aug_factor-1):
        ##  create mask to mask graph only 
        delete_mask = np.ones([graph.shape[0], graph.shape[1], graph.shape[2]]) 
        add_mask = np.zeros([graph.shape[0], graph.shape[1], graph.shape[2]])
        for sub_idx in range(graph.shape[0]):   
            # random.seed(42)
            row_idx = list(random.sample(range(0, graph.shape[1]), num_mask_node))
            # random.seed(42)
            col_idx = list(random.sample(range(0, graph.shape[1]), num_mask_node))
            for i in row_idx:
                for j in col_idx:
                    #only delete percent %
                    delete_mask[sub_idx, i, j]=0
                    delete_mask[sub_idx, j, i]=0
                    #  only add percent %
                    add_mask[sub_idx, i, j]=1
                    add_mask[sub_idx, j, i]=1                    
                    
        coin_flip = random.choices(['HEADS', 'TAILS'], [5, 5])[0]
        if coin_flip=='HEADS':
            tmp_graph = np.multiply(graph, delete_mask) 
        else:
            tmp_graph = np.maximum(graph, add_mask)
               
        ## append feature/graph/label
        feature_aug = np.concatenate((feature_aug, feature), axis=0)
        graph_aug= np.concatenate((graph_aug, tmp_graph), axis=0)
        label_aug= np.concatenate((label_aug, label), axis=0)
    print('Augmented sample size: %d' % graph_aug.shape[0])
    return feature_aug, graph_aug, label_aug



def SupCon_loss(labels, feature_vectors, temperature=0.1):  
    # print(feature_vectors.shape)
    feature_vectors_normalized = tf.math.l2_normalize(feature_vectors, axis=1)
    # Compute logits
    logits = tf.divide(
        tf.matmul(feature_vectors_normalized, 
                  tf.transpose(feature_vectors_normalized)),
        temperature)
  
    y_true = tf.squeeze(labels)
    logits = tf.convert_to_tensor(logits)
    y_true = tf.cast(y_true, logits.dtype)  
    
    # Expand to [batch_size, 1]
    y_true = tf.expand_dims(y_true, -1)
    ##  use y_true label to create a SupCon positive/negative masks
    y_true = tf.cast(tf.equal(y_true, tf.transpose(y_true)), logits.dtype)
    y_true /= tf.math.reduce_sum(y_true, 1, keepdims=True)  
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_true)    
    loss = tf.math.reduce_mean(loss)  
    return loss

def plot_acc_history(history):
    # list all data in history
    # print(history.history.keys())
    # summarize history for accuracy
    plt.figure()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    
def plot_loss_history(history):
    # list all data in history
    # print(history.history.keys()) 
    # summarize history for loss
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper right')
    plt.show()


##  ============================================ ##
##              Dataset preparation              ##
##  ============================================ ##


##  load label data & configure features and graphs   dHCP atlas
print('... load data...')
# Example: Generate 100 random adjacency matrices for a graph with 82 nodes
n_nodes = 82
n_matrices = 100
random_adjacency_matrices = [generate_random_adjacency_matrix(n_nodes, density=0.3) for _ in range(n_matrices)]

# Stack the matrices along a new axis to create a 3D NumPy array
random_adjacency_matrices_3d = np.stack(random_adjacency_matrices, axis=0)

# Print the shape of the resulting 3D array
print("Shape of the 3D array:", random_adjacency_matrices_3d.shape)

# Print the first adjacency matrix as an example
print("Random Adjacency Matrix 1:")
print(random_adjacency_matrices_3d[0])

A =random_adjacency_matrices_3d
X = random_adjacency_matrices_3d
Y = np.random.randint(2, size=100)


##  experiment configuration  ##
repeat_N = 1

Perf_metrics_vali = -100*np.ones((repeat_N, 4))
Perf_metrics_test = -100*np.ones((repeat_N, 4))
Perf_metrics_ext = -100*np.ones((repeat_N, 4))
roc_auc = -100
## data split into training / validation/ testing

for repeat_idx in range(repeat_N): 

    sub_idx = np.array(np.arange(Y.shape[0]))
    idx_train_vali, idx_test = train_test_split(sub_idx, test_size=0.2, 
                                 stratify=Y)
    idx_train, idx_vali = train_test_split(idx_train_vali, test_size=0.25, 
                                 stratify=Y[idx_train_vali,])
    
    
    feature_train = X[idx_train,]
    feature_vali =  X[idx_vali,]
    feature_test = X[idx_test,]
    
    A_train = A[idx_train,]
    A_vali = A[idx_vali,]
    A_test = A[idx_test,]
    
    label_train= Y[idx_train,].astype('int')
    label_vali= Y[idx_vali,].astype('int')
    label_test= Y[idx_test,].astype('int')
    
    
    ##  data augmentation for training data: masked graph 
    feature_train_aug, A_train_aug, label_train_aug = Graph_aug(feature_train, 
                                                        A_train, 
                                                        label_train)
    
      
    ## renormalized laplacian transformation of A
    graph_train_aug = preprocess_adj_tensor(A_train_aug, symmetric=True)
    graph_train = preprocess_adj_tensor(A_train, symmetric=True)
    graph_vali = preprocess_adj_tensor(A_vali, symmetric=True)
    graph_test = preprocess_adj_tensor(A_test, symmetric=True)
    
    
    
    ##  ============================================ ##
    ##   Supervised Constrastive Learning task       ##
    ##  ============================================ ##
    print(" ")
    print('... train SupCon GCN encoder model ...')
    num_node = A.shape[1]
    num_feature = X.shape[2]
    num_filters = 1 
    num_out = 8
    num_class = 2
    SupCon_learning_rate = 1e-4
    SupCon_batch = 128
    max_supcon_epochs= 100
    
    ##  create a GCN encoder
    GCN_encoder = get_gcn_encoder(num_node, num_feature, num_filters, num_out, num_class)
    
    adam = Adam(lr=SupCon_learning_rate, beta_1=0.9, beta_2=0.999)
    GCN_encoder.compile(optimizer= adam,
                        loss=SupCon_loss)
    GCN_encoder.summary()
     
    ##  configure check point for GCN encoder
    best_supcon_encoder_name = 'best_SCL_GCN_encoder.hdf5'
    encoder_checkpointer = ModelCheckpoint(filepath='./saved_models/'+best_supcon_encoder_name, 
                                verbose=2, monitor='val_loss', 
                                save_weights_only=True, 
                                mode='auto', 
                                save_best_only=True)
    
    
    GCN_encoder_history= GCN_encoder.fit([feature_train_aug, graph_train_aug], 
                                         label_train_aug, 
                                         # validation_split= 0.2,
                                         validation_data= ([feature_vali, graph_vali], 
                                                             label_vali),
                                         batch_size= SupCon_batch, 
                                         verbose = 2,
                                         shuffle = False,
                                         epochs= max_supcon_epochs,
                                         callbacks=[encoder_checkpointer])
    
    plot_loss_history(GCN_encoder_history)
    ##   trace back and load best GCN encoder
    best_GCN_encoder = get_gcn_encoder(num_node, num_feature, 
                                       num_filters, num_out, num_class)
    best_GCN_encoder.load_weights('./saved_models/'+best_supcon_encoder_name)

 
    
    ##  ============================================ ##
    ##           Downstream learning task            ##
    ##  ============================================ ##
    print(" ")
    print('... train SupCon GCN downstream model ...')
    clf_batch = 32
    train_encoder_flag = False
    learning_rate = 1e-4
    clf_max_epochs = 50
    
    ##  weights for imbalanced data in keras
    class_weights = dict(zip(np.unique(label_train), 
                    class_weight.compute_class_weight(class_weight = 'balanced',
                                                                     classes = np.unique(label_train),
                                                                     y = label_train))) 
    
    ##  optimize weights of weighted cross-entropy ##
    alpha = 1.0
    class_weights[1] = class_weights[1]*alpha
    ##  create downstream classification model ##
    GCN_classifier = get_gcn_classifier(best_GCN_encoder, num_node, num_feature, 
                                        num_filters, num_class, trainable=train_encoder_flag)
    
    adam = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999)
    GCN_classifier.compile(optimizer=adam,
                             loss='categorical_crossentropy',
                             metrics=['accuracy'])
    # GCN_classifier.summary()
    
    y_train_onehot = to_categorical(label_train)
    y_train_aug_onehot = to_categorical(label_train_aug)
    y_vali_onehot = to_categorical(label_vali)
    y_test_onehot = to_categorical(label_test)
    
    
    ##  configure GCN classifier checkpointer
    best_down_model_name = 'best_SCL_GCN_classifier.hdf5'
    checkpointer = ModelCheckpoint(filepath='./saved_models/'+best_down_model_name, 
                                verbose=2, monitor='val_loss', 
                                save_weights_only=True, 
                                mode='auto', save_best_only=True)
    
    GCN_classifier_history = GCN_classifier.fit([feature_train_aug, graph_train_aug], 
                         y_train_aug_onehot, 
                         validation_data= ([feature_vali, graph_vali], 
                                           y_vali_onehot),
                         batch_size= clf_batch, 
                         class_weight=class_weights,
                         verbose = 2,
                         shuffle = False,
                         epochs= clf_max_epochs,
                         callbacks=[checkpointer])
    GCN_classifier.summary()
    ##  plot SupCon model training and validation
    plot_acc_history(GCN_classifier_history)
    plot_loss_history(GCN_classifier_history)
    
    
    ##  ============================================ ##
    ##        Classifier model testing (internal)    ##
    ##  ============================================ ##
    print(" ")
    print('...evaluate best SupCon GCN model ...')
    best_model = get_gcn_classifier(best_GCN_encoder, num_node, num_feature, 
                                    num_filters, num_class, trainable=train_encoder_flag)
    # best_model.summary()
    best_model.load_weights('./saved_models/'+best_down_model_name)
    
    
    ##  validation performance
    pred_probas = best_model.predict([feature_vali, graph_vali])
    y_score = pred_probas[:,1]
    y_true = y_vali_onehot[:,1].astype(int)
    y_pred = np.argmax(pred_probas, axis=1)      
    # ACC 
    acc = np.mean(y_true==y_pred)
    # Sen
    sen = np.mean(y_pred[y_true==1]==1)
    # Spe
    spe = np.mean(y_pred[y_true==0]==0)
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)  
    Perf_metrics_vali[repeat_idx] = np.array([acc, sen, spe, roc_auc])
    print('validation performance - Acc: %0.3f, Sen: %0.3f, Spe: %0.3f, AUC: %0.3f' \
          % (acc, sen, spe, roc_auc))
        
        
    ##  test performance
    pred_probas = best_model.predict([feature_test, graph_test])
    y_score = pred_probas[:,1]
    y_true = y_test_onehot[:,1].astype(int)
    y_pred = np.argmax(pred_probas, axis=1)      
    # ACC 
    acc = np.mean(y_true==y_pred)
    # Sen
    sen = np.mean(y_pred[y_true==1]==1)
    # Spe
    spe = np.mean(y_pred[y_true==0]==0)
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)  
    Perf_metrics_test[repeat_idx] = np.array([acc, sen, spe, roc_auc])
    print('Test performance - Acc: %0.3f, Sen: %0.3f, Spe: %0.3f, AUC: %0.3f' \
          % (acc, sen, spe, roc_auc)) 
        
  
  