# CNN Hyperparameter Optimization with WandB and Sweep

This project demonstrates the use of [Weights & Biases (WandB)](https://wandb.ai) and its Sweep functionality to efficiently optimize the hyperparameters of a Convolutional Neural Network (CNN) for Pneumonia detection Task.

---

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Requirements](#requirements)
- [Project Structure](#project-structure)

---

## Introduction

Hyperparameter tuning is crucial to achieving optimal performance in deep learning models. This project leverages WandB for experiment tracking and Sweep for automated hyperparameter optimization. It focuses on optimizing key parameters of a CNN, such as:

- Learning rate
- Batch size
- Number of layers
- Filter sizes
- Dropout rates

The CNN is trained on an image classification dataset, such as CIFAR-10 or MNIST.

---

## Features

- **Experiment Tracking**: Logs all training metrics and parameters to WandB.
- **Hyperparameter Optimization**: Uses WandB Sweep to automate and optimize hyperparameter tuning.
- **Configurable Architecture**: Modify the CNN model and training pipeline as needed.
- **Visualization**: Real-time performance graphs and comparative analysis in the WandB dashboard.

---

## Requirements

Ensure you have the following installed:

- Python 3.7 or higher
- [Weights & Biases](https://docs.wandb.ai/quickstart) (`wandb`)
- TensorFlow or PyTorch (depending on your implementation)
- Additional dependencies listed in `requirements.txt`

## Project structure

- **data/**: Includes the data loader for managing datasets.
- **net/**: Contains the CNN architecture and model definition.
- **sweep/**: Contains the configuration files for WandB Sweep.
- **main.py**: The main script of the project, responsible for orchestrating the training and hyperparameter optimization.
- **utils**: Utility classes and functions to support the program's functionality.


