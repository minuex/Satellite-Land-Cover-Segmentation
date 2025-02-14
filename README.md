# Satellite_Land_Cover_Segmentation
Semantic segmentation of satellite imagery to classify land cover types using deep learning models.

![Image](https://github.com/user-attachments/assets/8291f340-3786-4e67-88e4-a1dcfa63bae4)



## Overview

This project focuses on land cover segmentation using deep learning models. It compares the performance of DeepLabv3+, UNet++, and SegFormer in classifying land cover types based on 10m satellite imagery. 
The goal is to evaluate their effectiveness and improve generalization through preprocessing, postprocessing, and ensemble techniques.
Additionally, this project aims to build a satellite imagery processing pipeline for potential use in Unreal Engine, enabling integration of high-resolution land cover data into 3D environments.

![Image](https://github.com/user-attachments/assets/f5d1ec33-3e7d-41d3-8bf2-1292984ae1dd)



## Dataset

|                        | Satellite Imagery Segmentation Dataset_AIHub                                                        |
|------------------------|-----------------------------------------------------------------------------------------------------|
| Link                   | https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=&topMenu=&aihubDataSe=data&dataSetSn=71361  |
| Reference              | European Space Agency_Sentinel-2                                                                    |
| Band                   | Blue, Red, Green, NIR                                                                               |
| Spatial Resolution (m) | 10 m                                                                                                |
| Data Period (year)     | 2019 - 2020                                                                                         |
| Count                  | 810 Pairs                                                                                           |

The project uses two datasets:

- AIHub Dataset: A preprocessed dataset with high segmentation accuracy.

- Field Dataset: A custom dataset collected for real-world testing, exhibiting domain differences from AIHub.



## Model Architectures

![Image](https://github.com/user-attachments/assets/b8c0d7d6-0c69-4133-9b81-8f1dc0b01b60)

| Model      | Backbone  | Structure                   | Params   | Flops     |
|------------|-----------|-----------------------------|----------|-----------|
| DeepLabV3+ | ResNet101 | CNN + ASPP                  | 55 ~ 60m | 120 ~ 150 |
| UNet++     | ResNet50  | CNN + Dense Skip Connection | 45 ~ 50m | 680 ~ 800 |
| SegFormer  | MiT-B3    | Transformer based           | 47 ~ 50m | 180 ~ 220 |

- DeepLabv3+: Encoder-decoder-based model using atrous convolution.

- UNet++: An improved version of UNet with dense skip connections.

- SegFormer: Transformer-based segmentation model for better context understanding.

| Model       | DeepLabV3+ | UNet++  | Segformer B3 |
|------------|------------|--------|--------------|
| Loss       | 0.3129     | 0.2870 | 0.2835       |
| Inference Time (h) | 8h | 46h | 24h |
| GPU Usage  | 10        | 19     | 18           |
| Mean IoU   | 0.4915     | 0.5256 | 0.5305       |
| Pixel Accuracy | 0.8006 | 0.8069 | 0.8159       |
| Building IoU | 0.7126 | 0.6843 | 0.7081       |
| Road IoU   | 0.5108     | 0.6344 | 0.5539       |
| Field IoU  | 0.7044     | 0.6942 | 0.7557       |
| Forest IoU | 0.7072     | 0.6758 | 0.7620       |
| Background IoU | 0.7779 | 0.8010 | 0.7974       |



## Preprocessing Techniques

![Image](https://github.com/user-attachments/assets/27e30128-ecc9-4d36-9550-07dd76f9e0a3)

1. Edge Enhancement: Applied to make boundaries of objects such as buildings and roads more distinct.

2. Saturation Enhancement: Increased contrast in green regions (forests, farmlands) to better differentiate between classes.

3. CLAHE (Contrast Limited Adaptive Histogram Equalization): Ensured balanced contrast in areas with shadows or significant brightness variations.



## Postprocessing Techniques (Road Class)

![Image](https://github.com/user-attachments/assets/ca561f68-5360-4d66-93e0-9f17f15470f9)

To enhance model performance, several postprocessing methods were applied:

1. Softmax Probability-Based Road Restoration ()

2. Edge-based postprocessing (using Canny edge detection to refine segmentation boundaries)

3. Directional road filtering (using PCA to maintain consistent road structure in segmentation)

4. Skeletonization for Road Continuity (Using OpenCV Thinning Algorithm)

| Model       | Mean IoU | Pixel Accuracy | Building IoU | Road IoU | Field IoU | Forest IoU | Background IoU |
|------------|---------|---------------|--------------|----------|----------|-----------|---------------|
| No Postprocessing | 0.5850  | 0.8560        | 0.4707       | 0.1987   | 0.6266   | 0.8284    | 0.8004        |
| With Postprocessing | 0.6285  | 0.8565        | 0.4707       | 0.4162   | 0.6266   | 0.8284    | 0.8006        |
| **Change (%)**  | 🔺 +4.35% | 🔺 +0.05% | 🔺 0.00% | 🔺 +21.75% | 🔺+0.00% | 🔻 0.00% | 🔻 +0.02% |



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



## Output Format

![Image](https://github.com/user-attachments/assets/eb646d43-398d-4305-a3df-3be35d7fca56)

| Output Format  | Prediction Results | Unreal Engine Conversion |
|---------------|------------------|--------------------------|
| File Type    | GeoTIFF           | PNG                      |
| Resolution   | 512×512           | 512×512                  |
| Coordinate System | EPSG:32652    | None                     |
| Data Type    | 8-bit unsigned integer (uint8) | 8-bit unsigned integer (uint8) |
| Bands        | 1                 | 1                        |

![Image](https://github.com/user-attachments/assets/f2cf08cb-d112-431d-b0a4-683d372347bf)



## Installation 

```bash
pip install rasterio segmentation-models-pytorch torch torchvision numpy opencv-python matplotlib
```



## Contributors

[김민서] - [https://github.com/minuex](https://github.com/minuex)

## License

This project is licensed under the MIT License - see the LICENSE file for details.





