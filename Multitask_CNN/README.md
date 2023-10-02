# Multitask Deep Neural Network (DNN) Model

## Description

This project demonstrates the use of a multitask Deep Neural Network (DNN) model. In multitask learning, a single model is trained to perform multiple tasks simultaneously, sharing information among the tasks to potentially improve generalization. The provided code builds and tests such a model on synthetic data for three tasks.

## Dependencies

- Python
- Numpy
- Keras
- DL_models (custom module containing `get_multitaskDNN_model`)

## How to Run

1. Ensure you have the required dependencies installed.
2. Load the `DL_models` module which contains the `get_multitaskDNN_model` function.
3. Execute the Python code provided.
4. The script will generate synthetic data samples for training and testing.
5. A multitask DNN model will be built and trained on the training data.
6. Model's evaluation metrics will be printed after testing on the test data.

## Features

- Implements a multitask DNN model that can handle training for multiple tasks simultaneously.
- Uses Keras for building and training the DNN.
- Generates synthetic data samples for demonstration purposes.
- Outputs evaluation metrics for each task to assess model performance.

## Note

- The multitask model receives two different input datasets from different modalities, and predicts outcomes for three tasks. This architecture can be especially helpful when you have related tasks that can benefit from shared information.
- The provided code uses synthetic data for illustration. Replace random data generation with your own data loading mechanism to apply this to real-world data.
- Hyperparameters like the number of epochs, batch size, and others can be adjusted to suit specific needs.

## License

This project is open-source. Please provide the necessary credits if you choose to use or adapt this code for your requirements
