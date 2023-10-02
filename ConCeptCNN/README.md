
**ADHD Classification with Neural Networks**


**Overview**

This repo contains scripts used to  to classify ADHD based on fMRI and DTI data

**Requirements:**

Libraries: numpy, scipy, keras, sklearn, os, math, Graphviz


**Datasets**
DTI: data/dti/DTI_FA_matrix.npy.

ADHD:
data/adhd/aal/aal.mat

data/adhd/adhd_test.mat

data/adhd/adhd_train.mat


**Key Functions:**

load_dti_data(): Fetches DTI data.

load_adhd_data(name): Fetches ADHD data. Parameter name can be 'test', 'train', or 'all'.

run(batch_size, epochs, save_csv): Main function for data processing, model training, evaluation, and metric storage.

