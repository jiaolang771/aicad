# Stacked Sparse Autoencoder (SSAE) for Feature Learning and Classification

## Description

This project illustrates how to use a Stacked Sparse Autoencoder (SSAE) to learn representations from source data and then utilize the encoder part of the SSAE in a Deep Neural Network (DNN) model for classification on target data.

## Dependencies

- Python
- Numpy
- Keras

## How to Run

1. Ensure you have the required dependencies installed.
2. Execute the Python code provided.
3. The script will:
   - Generate synthetic data samples.
   - Build and train an SSAE model on source data.
   - Build and train a DNN model on target data using the encoder part of the SSAE.
   - Evaluate the DNN model on test data and print accuracy.
4. Additionally, you can explore the learned weights of both SSAE and DNN models.

## Features

- Demonstrates the process of unsupervised feature learning using SSAE on source data.
- Shows how to transfer learned representations to a supervised DNN model for classification on target data.
- Uses Keras for building, training, and evaluating models.
- Generates synthetic data samples for demonstration purposes.

## Note

- The provided code uses synthetic data for demonstration. To apply this to real-world data, you can replace the random data generation with your own data loading mechanism.
- Hyperparameters such as the number of epochs, batch size, and others can be adjusted based on specific needs.

## License

This project is open-source. If you choose to use or adapt this code for your needs, kindly give appropriate credit to the original author.

