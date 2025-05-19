# Federated Learning and Over-the-Air Computation

This repository contains the official implementation for the bachelor’s thesis:

**"Federated Learning and Over-the-Air Computation: A Comparative Study"**  
** by Fanny Nyberg & Filip Svebeck**  
School of Electrical Engineering and Computer Science, KTH Royal Institute of Technology, 2025

## About the Project

This project investigates and compares two communication strategies in Federated Learning (FL):

- **Orthogonal Federated Learning (FL)** – utilizing orthogonal communication and Federated Averaging (FedAvg).
- **Over-the-Air Federated Learning (Over-the-Air FL)** – utilizing non-orthogonal communication and Over-the-Air Computation (AirComp).

The goal of this thesis is to evaluate the convergence and accuracy of both methods under different settings.

## Project Structure

The repository includes the following files:

### Models
- `MLP.py` – A simple feedforward neural network used for MNIST.
- `CNN.py` – A convolutional neural network for CIFAR-10 classification.

### Orthogonal Federated Learning Implementations
- `O_FL_IID_MNIST_MLP.py`
- `O_FL_Non-IID_MNIST_MLP.py`
- `O_FL_IID_CIFAR-10_CNN.py`

### Over-the-Air Federated Learning Implementations
- `Air_FL_IID_MNIST_MLP.py`
- `Air_FL_Non-IID_MNIST_MLP.py`
- `Air_FL_IID_CIFAR-10_CNN.py`

Each script trains a global model using FL with the corresponding strategy and dataset.

## Requirements

Install the following Python packages before running:

```bash
pip install torch torchvision numpy matplotlib scipy
```
## License and Use
This codebase was developed as part of a bachelor's thesis at KTH and is intended for educational and research purposes.
