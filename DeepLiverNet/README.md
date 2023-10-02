# CNN Learning Project

## Description

This repository contains code for training and evaluating Convolutional Neural Networks (CNNs) on certain medical data. It specifically deals with implementing CNN models to analyze liver-related data, using various pre-trained models like VGG19, InceptionResNetV2, and NASNetLarge as base models for transfer learning. The primary goal of the project is to predict liver conditions based on the provided features, where the prediction results are accumulated across multiple k-fold validation runs.

## Dependencies

- Python
- Tensorflow
- Keras
- Numpy
- Scipy
- Matplotlib
- Sklearn

## How to Run

1. Ensure all dependencies are installed.
2. Place your data file at the specified location or update the path in the code.
3. Execute the given Python code.

## Note


- Ensure randomness is consistent across runs by setting seed values.
- Remember to adjust the path where the data is loaded from.
- The script uses k-fold cross-validation, make sure to specify the correct number of folds.
- Different pre-trained networks can be loaded by uncommenting the desired base model.

## License

This project is open-source and available to everyone. Please provide appropriate credit if you use or adapt any part of the code.
