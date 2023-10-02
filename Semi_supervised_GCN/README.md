# Graph Convolutional Networks (GCN) for Brain Structural Connectivity

## Description

This project demonstrates how to use a Semi-supervised Graph Convolutional Networks (GCN) model to classify brain structural connectivity data. The code loads synthetic data, processes it to suit a graph-based representation, and then trains a GCN model to classify the data based on some scores.

## Dependencies

- Python
- Keras
- Keras DGL (Deep Graph Library for Keras)
- Numpy
- NetworkX
- Scikit-learn

## How to Run

1. Ensure you have the required dependencies installed.
2. Execute the provided Python code.
3. The script will:
   - Generate synthetic data samples.
   - Process the data to fit a graph-based structure.
   - Build and train a GCN model on the data.
   - Evaluate the model and print training and test accuracies per epoch.

## Features

- Constructs graph-based representations using brain scores.
- Uses the Keras Deep Graph Library to apply graph convolutional layers.
- Provides a training loop with accuracy evaluations for both training and test datasets.
- Uses synthetic data for the demonstration. Can be replaced with real-world brain structural connectivity data.

## Note

- The provided code uses synthetic data for demonstration. Replace the random data generation with your own data loading mechanism if you wish to use real-world data.
- You might notice that the sample output indicates that the model's performance isn't optimal, which is expected given that the data is synthetic/random. Real data might produce different results.
- The `get_semi_gcn` function is expected to be defined in an external module named `Semi_GCN_models`. Ensure this module is available in the project directory.
  
This repository is for sharing codes of:

Li H, Li Z, Du K, Zhu Y, Parikh NA, He L. A Semi-Supervised Graph Convolutional Network for Early Prediction of Motor Abnormalities in Very Preterm Infants. Diagnostics (Basel). 2023 Apr 21;13(8):1508. doi: 10.3390/diagnostics13081508. PMID: 37189608; PMCID: PMC10137879.

Note: GCN packages
Keras Deep Learning on Graphs (Keras-DGL)
https://vermamachinelearning.github.io/keras-deep-graph-learning/

## License

This project is open-source. If you choose to use or adapt this code for your needs, kindly give appropriate credit to the contributors and maintain the spirit of open-source.

