# Multiscale Deep Neural Network (DNN) Model

## Description

This project implements a multiscale Deep Neural Network (DNN) model to handle inputs of various dimensions. The data used is synthetic, with different lengths, to demonstrate the architecture's versatility. This DNN model receives multiple input arrays of varying scales and predicts categorical labels based on them.

## Dependencies

- Python
- Numpy
- Keras
- DL_models (custom module containing `get_multiscaleDNN_model`)

## How to Run

1. Ensure you have the required dependencies installed.
2. Load the `DL_models` module that contains the `get_multiscaleDNN_model` function.
3. Execute the Python code provided.
4. The script will create synthetic data samples for training and testing.
5. A multiscale DNN model will be constructed and trained on the training data.
6. The validation accuracy of the model will be printed after testing on the test data.

## Features

- Utilizes a multiscale DNN architecture capable of processing multiple input arrays of varying dimensions.
- Employs Keras for building and training the DNN.
- Generates synthetic data samples for demonstration purposes.
- Provides metrics like validation accuracy for evaluation.

## Note

- The actual dimensions for the input arrays are derived from the formula: \( \frac{n \times (n-1)}{2} \), which is the formula for combinations.
- The code uses synthetic data for illustration purposes. Replace the random data generation with your data loading mechanism to work with real-world data.
- Adjust hyperparameters like the number of epochs, batch size, etc., as required.

## License

This project is open-source. Kindly give the necessary credits if you choose to use or adapt this code for your needs.
