
# **Brain Image Analysis with Neural Networks**

Funding: This work was supported by the National Institutes of Health R01-EB029944.
A collection of scripts and functions for processing, predicting, and evaluating brain images using state-of-the-art deep learning techniques.

•	Li H, Chen M, Wang J, Illapani VSP, Parikh NA, He L. Automatic Segmentation of Diffuse White Matter Abnormality on T2-weighted Brain MRI Using Deep Learning in Very Preterm Infants. Radiology: Artificial Intelligence 2021;0:e200166.


## **Table of Contents**
- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)


## **Overview**

This repository focuses on:
- Classifying ADHD using neural networks based on fMRI and DTI data.
- Predicting brain image data and evaluating those predictions.
- Training a U-Net neural network model for image segmentation.

## **Prerequisites**

- Python 3.x
- Libraries:
  - Keras
  - Tensorflow
  - numpy
  - scipy
  - sklearn

## **Installation**

1. Clone this repository.
2. Ensure you have all the necessary libraries installed.
3. Install additional tools like Graphviz: `C:\\Program Files (x86)\\Graphviz2.38\\bin`.

## **Usage**

1. **ADHD Classification**
    ```bash
    python <script_name>.py
    ```
2. **Brain Image Prediction & Evaluation**

   Ensure to set your dataset paths for `imgs_test` and `masks_test`, then run the respective script.

3. **U-Net Model Training**

   Place your patches data in the `./Processed_patch_data/` directory and run the script. Adjust parameters like epochs and batch size as needed.


