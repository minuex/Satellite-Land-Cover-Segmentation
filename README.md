# Satellite_Land_Cover_Segmentation
Semantic segmentation of satellite imagery to classify land cover types using deep learning models.

## Overview

This project focuses on land cover segmentation using deep learning models. It compares the performance of DeepLabv3+, UNet++, and SegFormer in classifying land cover types based on 10m satellite imagery. 
The goal is to evaluate their effectiveness and improve generalization through preprocessing, postprocessing, and ensemble techniques.

## Dataset

The project uses two datasets:

- AIHub Dataset: A preprocessed dataset with high segmentation accuracy.

- Field Dataset: A custom dataset collected for real-world testing, exhibiting domain differences from AIHub.

## Model Architectures

- DeepLabv3+: Encoder-decoder-based model using atrous convolution.

- UNet++: An improved version of UNet with dense skip connections.

- SegFormer: Transformer-based segmentation model for better context understanding.

## Postprocessing Techniques

To enhance model performance, several postprocessing methods were applied:

1. Edge-based postprocessing

2. Uncertainty-based filtering

## License

This project is licensed under the MIT License - see the LICENSE file for details.


