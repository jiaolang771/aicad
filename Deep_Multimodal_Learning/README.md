# Deep Multimodal Neural Network

## Description

This repository contains a code implementation to experiment with a Deep Multimodal Neural Network using medical data samples. The primary purpose is to use various input modalities, preprocess them with a VGG19 model, and then feed them to a DeepMultimodal model. Random samples for various modalities (FC_features, SC_features, DWMA_features, Clinical_features) are created and used for both training and testing purposes.

## Dependencies

- Python
- Numpy
- Keras
- Sklearn

## How to Run

1. Ensure you have all the dependencies installed.
2. Execute the given Python code.
3. The script will generate random data, preprocess it using a VGG19 model, and then run it through the DeepMultimodal model.
4. Results, including the test accuracy, will be printed at the end.

## Features

- Utilizes the VGG19 model to preprocess input data.
- Implements transfer learning where layers of the VGG19 model are frozen.
- Uses the DeepMultimodal model (from `DL_models`) to process different modalities.
- Employs random data for experimentation.
- Separates the data into training and testing subsets.
- Evaluates the final model on the test data and prints accuracy.

## Note


- Adjust hyperparameters (like learning rate, number of epochs) to adapt your own task and dataset
- The current data is randomly generated. For practical applications, replace this with real medical data.
- Ensure the DeepMultimodal model is correctly imported from `DL_models`.

## License

This project is open-source. Kindly provide the necessary credits if you use or adapt this code for your purposes.
