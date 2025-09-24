# Dynamic weighted hypergraph convolutional network for brain functional connectome

Funding: This work was supported by the National Institutes of Health R01-EB029944.

•	Wang J, Li H, Qu G, Cecil KM, Dillman JR, Parikh NA, He L. Dynamic weighted hypergraph convolutional network for brain functional connectome analysis. Med Image Anal 2023;87:102828. PMID: 37130507; PMCID: PMC10247416.


The input data is the series of matrix in .mat file generated from MatLab. It is in a cell format in M x 4, where M is the number of subjects in each catagory (i.e. higher IQ group or lower IQ group). 
The 5 columns of matrix are:
                   1. N x N matrix, the original hypergraph similarity matrix, N is the number of nodes (ROIs).
                   2. N x N matrix, the incidence matrix.
                   3. diagonal matrix, the edge degree with exponential (-1/2).
                   4. N x P matrix, the original graph signals.
            
