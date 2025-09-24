# ConCeptCNN: A novel multi-filter convolutional neural network

Funding: This work was supported by the National Institutes of Health R01-EB029944.

•	Chen M, Li H, Fan H, Dillman JR, Wang H, Altaye M, Zhang B, Parikh NA, He L. ConCeptCNN: A novel multi-filter convolutional neural network for the prediction of neurodevelopmental disorders using brain connectome. Med Phys 2022;49:3171–3184. PMID: 35246986; PMCID: PMC9164760.


**ADHD Classification with Neural Networks**


**Overview**

This repo contains scripts used to  to classify ADHD based on fMRI and DTI data

**Requirements:**

Libraries: numpy, scipy, keras, sklearn, os, math, Graphviz


**Dataset Structure**
DTI: data/dti/DTI_FA_matrix.npy.

ADHD:
data/adhd/aal/aal.mat

data/adhd/adhd_test.mat

data/adhd/adhd_train.mat


**Key Functions:**

load_dti_data(): Fetches DTI data.

load_adhd_data(name): Fetches ADHD data. Parameter name can be 'test', 'train', or 'all'.

run(batch_size, epochs, save_csv): Main function for data processing, model training, evaluation, and metric storage.

