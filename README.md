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

- UNet++ & SegFormer Ensemble: Combining UNet++ and SegFormer for improved segmentation performance, especially for road segmentation.

## Postprocessing Techniques

To enhance model performance, several postprocessing methods were applied:

1. Edge-based postprocessing (using Canny edge detection to refine segmentation boundaries)

2. Uncertainty-based filtering (refining labels in areas with low softmax confidence)

3. Directional road filtering (using PCA to maintain consistent road structure in segmentation)

4. Contour simplification (approximating road contours for more natural segmentation results)

## Results & Findings

- AIHub-trained models struggled on Field Dataset due to domain differences.

- Postprocessing techniques improved segmentation accuracy by reducing misclassified regions.

- SegFormer showed the best overall performance, followed by DeepLabv3+ and UNet++.

- The ensemble of UNet++ and SegFormer significantly improved road segmentation accuracy, outperforming individual models by effectively capturing both fine details and global context.

## Model Performance Comparison
| Model                | Mean IoU | Pixel Accuracy | Building IoU | Road IoU | Field IoU | Forest IoU | Background IoU |
|----------------------|---------|---------------|--------------|----------|----------|-----------|---------------|
| Segformer B3        | 0.5305  | 0.8159        | 0.7081       | 0.5539   | 0.7557   | 0.7620    | 0.7974        |
| Selective Ensemble  | 0.5514  | 0.8307        | 0.7357       | 0.6575   | 0.7732   | 0.7457    | 0.8110        |
| **Improvement**     | 🔺 +2.09% | 🔺 +1.81% | 🔺 +2.76% | 🔺 +10.36% | 🔺 +1.75% | 🔻 -2.13% | 🔺 +1.71% |

## Contributors

[김민서] - [as09003@naver.com / minuex]

## License

This project is licensed under the MIT License - see the LICENSE file for details.


