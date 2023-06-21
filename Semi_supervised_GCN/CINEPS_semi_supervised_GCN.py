from keras.layers import Dense, Activation, Dropout, BatchNormalization
from keras.models import Model, Sequential
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.utils import to_categorical
import keras.backend as K
import numpy as np

from keras_dgl.utils import *
from keras_dgl.layers import GraphCNN
from keras_dgl.utils import preprocess_adj_numpy
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import networkx as nx
from Semi_GCN_models import get_semi_gcn
from sklearn.model_selection import StratifiedKFold, KFold




K.clear_session()
##  load data
StruConn_data = np.random.rand(224,90,90)
Global_brain_score = np.random.rand(224,1)
Bayley_score = np.random.rand(224,1)

#
N_sub = 224
N_sub_outcome = N_sub - sum(Bayley_score[:,0]==-1)

# Prepare Graph Data
A = np.zeros((N_sub, N_sub))
for i in range(N_sub):
    for j in range(N_sub):
        diff_gbs = np.abs(Global_brain_score[i,0]-Global_brain_score[j,0])
        if i!=j and diff_gbs==0:
            A[i,j] = 1
        elif i!=j:
            A[i,j] = 1/diff_gbs  

# Node features
X = []
for sc_a in StruConn_data:
    sc_a = sc_a[np.triu_indices(90, 1)]
    X.append(sc_a)
X = np.array(X)

Y = to_categorical(Bayley_score<=0.5)     

##   split data into train and test
outcome_idx = np.where(Bayley_score[:,0]!=-1)[0]
train_idx, test_idx = train_test_split(outcome_idx, 
                                       test_size=0.2, shuffle=True)


labels = np.argmax(Y, axis=1) + 1
train_mask = np.zeros(N_sub, dtype=bool)
train_mask[train_idx] = True
# Normalize X
X /= X.sum(1).reshape(-1, 1)
X = np.array(X)

Y_train = np.zeros(Y.shape)
labels_train = np.zeros(labels.shape)
Y_train[train_idx] = Y[train_idx]
labels_train[train_idx] = labels[train_idx]

Y_test = np.zeros(Y.shape)
labels_test = np.zeros(labels.shape)
Y_test[test_idx] = Y[test_idx]
labels_test[test_idx] = labels[test_idx]

# Build Graph Convolution filters
SYM_NORM = True
A_norm = preprocess_adj_numpy(A, SYM_NORM)
num_filters = 1
graph_conv_filters = K.constant(A_norm)


# Build Model
num_feature = X.shape[1]
num_class = 2
model = get_semi_gcn(num_feature, num_filters, graph_conv_filters, num_class)
model.summary()


nb_epochs = 10
##   set class weights
Y_train_vec = np.argmax(Y_train[train_mask], axis=1)
class_weights = compute_class_weight('balanced', 
                                     classes=np.unique(Y_train_vec), y=Y_train_vec)
class_weights = dict(enumerate(class_weights))

for epoch in range(nb_epochs):
    model.fit(X, Y_train, 
              sample_weight=train_mask,
              class_weight=class_weights,
              batch_size=A.shape[0], 
              epochs=1, 
              shuffle=False, 
              verbose=0)
    
    Y_pred = model.predict(X, batch_size=A.shape[0])
    
    _, train_acc = evaluate_preds(Y_pred, [Y_train], [train_idx])
    _, test_acc = evaluate_preds(Y_pred, [Y_test], [test_idx])
    print("Epoch: {:03d}".format(epoch), "train_acc= {:.3f}".format(train_acc[0]), 
          "test_acc= {:.3f}".format(test_acc[0]))

# Sample Output
# Epoch: 000 train_acc= 0.436 test_acc= 0.489
# Epoch: 001 train_acc= 0.436 test_acc= 0.489
# Epoch: 002 train_acc= 0.436 test_acc= 0.489
# Epoch: 003 train_acc= 0.464 test_acc= 0.467
# Epoch: 004 train_acc= 0.531 test_acc= 0.400
# Epoch: 005 train_acc= 0.547 test_acc= 0.511
# Epoch: 006 train_acc= 0.564 test_acc= 0.511
# Epoch: 007 train_acc= 0.564 test_acc= 0.511
# Epoch: 008 train_acc= 0.564 test_acc= 0.511
# Epoch: 009 train_acc= 0.564 test_acc= 0.511
