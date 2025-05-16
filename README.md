# Federated Learning and Over-the-Air Computation

This repository contains the official implementation for the bachelor’s thesis:

**"Federated Learning and Over-the-Air Computation: A Comparative Study"**  
**Fanny Nyberg & Filip Svebeck**  
School of Electrical Engineering and Computer Science, KTH Royal Institute of Technology, 2025

## 🎓 About the Project

This project investigates and compares two aggregation strategies in Federated Learning:

- **Federated Averaging (FedAvg)** – the standard federated averaging algorithm.
- **Over-the-Air Computation (AirComp)** – aggregation by leveraging signal superposition.

The goal of this thesis is to evaluate the convergence and accuracy of both methods under different settings.

## Project Structure

The repository includes the following files:

### Models
- `MLP.py` – A simple feedforward neural network used for MNIST.
- `CNN.py` – A convolutional neural network for CIFAR-10 classification.

### FedAvg Implementations
- `FedAvg_IID_MNIST_MLP.py`
- `FedAvg_Non-IID_MNIST_MLP.py`
- `FedAvg_IID_CIFAR-10_CNN.py`

### AirComp Implementations
- `AirComp_IID_MNIST_MLP.py`
- `AirComp_Non-IID_MNIST_MLP.py`
- `AirComp_IID_CIFAR-10_CNN.py`

Each script trains a global model using federated learning with the corresponding strategy and dataset.

## Requirements

Install the following Python packages before running:

```bash
pip install torch torchvision numpy matplotlib scipy
```
## License and USE
This codebase was developed as part of a bachelor thesis at KTH and is intended for educational and research purposes.
