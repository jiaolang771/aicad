# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 23:23:13 2022

@author: liz27
"""

# import packages
import pandas as pd
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
from sklearn import metrics


# read results
df = pd.read_csv('...result...file')
y_test = np.array(df['y'])

# prediction
y_pred = np.array(df['yhat'])
knn_pred = np.array(df['knn'])
lr_pred = np.array(df['lr'])
dt_pred = np.array(df['dt'])
rf_pred = np.array(df['rf'])
nn_pred = np.array(df['nn'])
svm_pred = np.array(df['svm'])
voting_pred = np.array(df['voting'])
bagging_pred = np.array(df['bagging'])
stacking_pred = np.array(df['stacking'])
abel_pred = np.array(df['abel'])
mNN_pred = np.array(df['mnn'])

# probabilities
probs = np.array(df['p'])
probs_knn = np.array(df['pknn'])
probs_lr = np.array(df['plr'])
probs_dt = np.array(df['pdt'])
probs_rf = np.array(df['prf'])
probs_nn = np.array(df['pnn'])
probs_svm = np.array(df['psvm'])
probs_voting = np.array(df['pvoting'])
probs_bagging = np.array(df['pbagging'])
probs_stacking = np.array(df['pstacking'])
probs_abel = np.array(df['pabel'])
probs_mNN = np.array(df['pmnn'])


# function to compute metrics
def metrics(y1,y2):
    cm = confusion_matrix(y2, y1)
    total = sum(sum(cm))
    acc = round(accuracy_score(y2, y1), 4)
    recall = round(cm[1,1]/(cm[0,1]+cm[1,1]), 4)
    speci = round(cm[0,0]/(cm[0,0]+cm[1,0]), 4)
    
    return round(acc*100,1), round(recall*100,1), round(speci*100,1)

metrics(y_test,y_pred)
metrics(y_test,knn_pred)
metrics(y_test,lr_pred)
metrics(y_test,dt_pred)
metrics(y_test,rf_pred)
metrics(y_test,nn_pred)
metrics(y_test,svm_pred)
metrics(y_test,voting_pred)
metrics(y_test,bagging_pred)
metrics(y_test,stacking_pred)
metrics(y_test,abel_pred)
metrics(y_test,mNN_pred)


# compute auc
auc_oap = roc_auc_score(y_test, probs)
auc_knn = round(roc_auc_score(y_test, probs_knn), 4)
auc_lr = round(roc_auc_score(y_test, probs_lr), 4)
auc_dt = round(roc_auc_score(y_test, probs_dt), 4)
auc_rf = round(roc_auc_score(y_test, probs_rf), 4)
auc_nn = round(roc_auc_score(y_test, probs_nn), 4)
auc_svm = round(roc_auc_score(y_test, probs_svm), 4)
auc_voting = round(roc_auc_score(y_test, probs_voting), 4)
auc_bagging = round(roc_auc_score(y_test, probs_bagging), 4)
auc_stacking = round(roc_auc_score(y_test, probs_stacking), 4)
auc_abel = round(roc_auc_score(y_test, probs_abel), 4)
auc_mNN = round(roc_auc_score(y_test, probs_mNN), 4)


# compute recall and spec for ROC
fpr, tpr, _ = roc_curve(y_test, probs)
fpr_knn, tpr_knn, _ = roc_curve(y_test, probs_knn)
fpr_lr, tpr_lr, _ = roc_curve(y_test, probs_lr)
fpr_dt, tpr_dt, _ = roc_curve(y_test, probs_dt)
fpr_rf, tpr_rf, _ = roc_curve(y_test, probs_rf)
fpr_nn, tpr_nn, _ = roc_curve(y_test, probs_nn)
fpr_svm, tpr_svm, _ = roc_curve(y_test, probs_svm)
fpr_voting, tpr_voting, _ = roc_curve(y_test, probs_voting)
fpr_bagging, tpr_bagging, _ = roc_curve(y_test, probs_bagging)
fpr_stacking, tpr_stacking, _ = roc_curve(y_test, probs_stacking)
fpr_abel, tpr_abel, _ = roc_curve(y_test, probs_abel)
fpr_mNN, tpr_mNN, _ = roc_curve(y_test, probs_mNN)


# plot the roc curve for the model
plt.figure(figsize=(15,15))
plt.plot(fpr, tpr, marker='.', label='OAP-EL (AUC: %0.2f)'% auc_oap
         , color = 'darkblue', lw=3.5)
plt.plot(fpr_knn, tpr_knn, marker='.', label='kNN (AUC: %0.2f)'% auc_knn
         , color = 'cornflowerblue', lw=1.5, alpha=.4)
plt.plot(fpr_lr, tpr_lr, marker='.', label='LR (AUC: %0.2f)'% auc_lr
, color = 'darkorange', lw=2, alpha=.4)
plt.plot(fpr_dt, tpr_dt, marker='.', label='DT (AUC: %0.2f)'% auc_dt
         , color = 'brown', lw=2, alpha=.4)
plt.plot(fpr_rf, tpr_rf, marker='.', label='RF (AUC: %0.2f)'% auc_rf
         , color = 'darkred', lw=2, alpha=.4)
plt.plot(fpr_nn, tpr_nn, marker='.', label='NN (AUC: %0.2f)'% auc_nn
         , color = 'pink', lw=2, alpha=.4)
plt.plot(fpr_svm, tpr_svm, marker='.', label='SVM (AUC: %0.2f)'% auc_svm
         , color = 'green', lw=2, alpha=.4)
plt.plot(fpr_voting, tpr_voting, marker='.', label='Voting (AUC: %0.2f)'% auc_voting
         , color = 'red', lw=2, alpha=.4)
plt.plot(fpr_bagging, tpr_bagging, marker='.', label='Bagging (AUC: %0.2f)'% auc_bagging
         , color = 'grey', lw=2, alpha=.4)
plt.plot(fpr_stacking, tpr_stacking, marker='.', label='Stacking (AUC: %0.2f)'% auc_stacking
         , color = 'olive', lw=2, alpha=.4)
plt.plot(fpr_abel, tpr_abel, marker='.', label='AB-EL (AUC: %0.2f)'% auc_abel
         , color = 'blue', lw=2.5, alpha=.6)
plt.plot(fpr_mNN, tpr_mNN, marker='.', label='OAP-mNN (AUC: %0.2f)'% auc_mNN
         , color = 'violet', lw=2.5, alpha=.6)


plt.plot([0, 1], [0, 1], color='grey', lw=1.8, linestyle='--', label = 'Random (AUC: %0.2f)' % 0.500)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize = 30)
plt.ylabel('True Positive Rate', fontsize = 30)
plt.xticks(fontsize=30) 
plt.yticks(fontsize=30)
plt.legend(loc="lower right",fontsize = 22)
plt.show()




# precision recall curve
pre, re, _ = precision_recall_curve(y_test, probs)
pre_knn, re_knn, _ = precision_recall_curve(y_test, probs_knn)
pre_lr, re_lr, _ = precision_recall_curve(y_test, probs_lr)
pre_dt, re_dt, _ = precision_recall_curve(y_test, probs_dt)
pre_rf, re_rf, _ = precision_recall_curve(y_test, probs_rf)
pre_nn, re_nn, _ = precision_recall_curve(y_test, probs_nn)
pre_svm, re_svm, _ = precision_recall_curve(y_test, probs_svm)
pre_voting, re_voting, _ = precision_recall_curve(y_test, probs_voting)
pre_bagging, re_bagging, _ = precision_recall_curve(y_test, probs_bagging)
pre_stacking, re_stacking, _ = precision_recall_curve(y_test, probs_stacking)
pre_abel, re_abel, _ = precision_recall_curve(y_test, probs_abel)
pre_mNN, re_mNN, _ = precision_recall_curve(y_test, probs_mNN)


# get prauc score
pr = auc(re, pre)
prknn = auc(re_knn, pre_knn)
prlr = auc(re_lr, pre_lr)
prdt = auc(re_dt, pre_dt)
prrf = auc(re_rf, pre_rf)
prnn = auc(re_nn, pre_nn)
prsvm = auc(re_svm, pre_svm)
prvoting = auc(re_voting, pre_voting)
prbagging = auc(re_bagging, pre_bagging)
prstacking = auc(re_stacking, pre_stacking)
prable = auc(re_abel, pre_abel)
prmnn =  auc(re_mNN, pre_mNN)

# plot the precision-recall curves
plt.figure(figsize=(15,15))
no_skill = len(y_test[y_test==1]) / len(y_test)
plt.plot(re, pre, marker='.', label='OAP-EL (PRAUC: %0.2f)'% pr
         , color = 'darkblue', lw=3.5)
plt.plot(re_knn, pre_knn, marker='.', label='kNN (PRAUC: %0.2f)'% prknn
         , color = 'cornflowerblue', lw=2, alpha=.4)
plt.plot(re_lr, pre_lr, marker='.', label='LR (PRAUC: %0.2f)'% prlr
         , color = 'darkorange', lw=2, alpha=.4)
plt.plot(re_dt, pre_dt, marker='.', label='DT (PRAUC: %0.2f)'% prdt
         , color = 'brown', lw=2, alpha=.4)
plt.plot(re_rf, pre_rf, marker='.', label='RF (PRAUC: %0.2f)'% prrf
         , color = 'darkred', lw=2, alpha=.4)
plt.plot(re_nn, pre_nn, marker='.', label='NN (PRAUC: %0.2f)'% prnn
         , color = 'pink', lw=2, alpha=.4)
plt.plot(re_svm, pre_svm, marker='.', label='SVM (PRAUC: %0.2f)'% prsvm
         , color = 'green', lw=2, alpha=.4)
plt.plot(re_voting, pre_voting, marker='.', label='Voting (PRAUC: %0.2f)'% prvoting
         , color = 'red', lw=2, alpha=.4)
plt.plot(re_bagging, pre_bagging, marker='.', label='Bagging (PRAUC: %0.2f)'% prbagging
         , color = 'grey', lw=2, alpha=.4)
plt.plot(re_stacking, pre_stacking, marker='.', label='Stacking (PRAUC: %0.2f)'% prstacking
         , color = 'olive', lw=2, alpha=.4)
plt.plot(re_abel, pre_abel, marker='.', label='AB-EL (PRAUC: %0.2f)'% prable
         , color = 'blue', lw=2.5, alpha=.5)
plt.plot(re_mNN, pre_mNN, marker='.', label='OAP-mNN (PRAUC: %0.2f)'% prmnn
         , color = 'violet', lw=2.5, alpha=.5)
plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='Random (PRAUC: %0.2f)'% no_skill) 

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall', fontsize = 30)
plt.ylabel('Precision', fontsize = 30)
plt.xticks(fontsize=30) 
plt.yticks(fontsize=30)
plt.legend(loc="upper right",fontsize = 22)
plt.show()
