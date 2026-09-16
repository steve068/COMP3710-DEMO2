# COMP3710 Pattern Recognition - Lab Demonstration 2

This repository contains the implementation of dimensionality reduction, classification, and deep learning pipelines for the COMP3710 Pattern Recognition Lab Demonstration 2.

## Overview

The project is structured into four main parts according to the lab requirements:
1. **Discrete Fourier Transform (DFT):** Implementation and performance comparison of naive DFT and GPU-accelerated tensor operations.
2. **Eigenfaces & Classification:** Principal Component Analysis (PCA) for dimensionality reduction on the Labeled Faces in the Wild (LFW) dataset, followed by classification using a Random Forest model.
3. **Convolutional Neural Networks (CNNs):** End-to-end deep learning feature extraction and classification, including a CNN for the LFW dataset and a ResNet-18 model for the DAWNBench Challenge on CIFAR10.
4. **Advanced Recognition Tasks:** Custom deep learning architectures for specific recognition problems using the OASIS MRI dataset.

## Repository Structure

| File/Directory | Description |
| :--- | :--- |
| `DFT1.1.py` | Part 1: Baseline Numpy implementation of the Discrete Fourier Transform. |
| `Pytorch DFT 1.2.py` | Part 1: GPU-accelerated DFT implementation using PyTorch tensor operations. |
| `PCA(Eigenfaces) 2.1.py` | Part 2: Feature extraction using PCA and classification via Random Forest on the LFW dataset. |
| `3.1 CNN Classifier.py` | Part 3: Custom CNN classifier (2x Conv layers, 32 filters) for the LFW dataset. |
| `3.2.py` | Part 3: DAWNBench Challenge implementation (Fast CIFAR10 classification using ResNet-18). |
| `4.1.py` | Part 4 (Task 1): Variational Autoencoder (beta-VAE) implementation for MR image manifold visualization. |
| `4.2 UNet.py` | Part 4 (Task 2): UNet architecture for MR image segmentation with live DSC (Dice Similarity Coefficient) metric calculation. |
| `main.py` | Main entry script for the project environment. |
| `vae_manifold.png` | Output visualization of the manifold created by the VAE model. |

## Requirements

* Python 3.10+
* PyTorch / TensorFlow (depending on the specific script environment)
* scikit-learn
* numpy
* matplotlib

## Execution & Hardware Notes

* **Local Execution:** Scripts corresponding to Part 1 and Part 2 can be executed locally on standard CPU/GPU environments. 
* **Cluster Execution:** For Part 3 (DAWNBench) and Part 4 (OASIS dataset), models are designed to run inference and training on a High-Performance Computing (HPC) cluster (e.g., UQ Rangpur compute cluster with NVIDIA V100/A100 GPUs) to meet the speed and memory requirements.
* **SLURM Job Submission:** To execute the DAWNBench challenge script on the cluster, use the following SLURM command:
  ```bash
  sbatch run_dawnbench.slurm