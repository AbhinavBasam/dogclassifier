
# Fine-Grained Dog Breed Classification using Ensemble Transfer Learning

![Project Status](https://img.shields.io/badge/Status-Completed-green)
![Framework](https://img.shields.io/badge/Framework-TensorFlow%20%7C%20Keras-orange)
![License](https://img.shields.io/badge/License-MIT-blue)

## 📄 Abstract
Fine-grained image classification remains a significant challenge in computer vision due to low inter-class variance and high intra-class variance. This project presents a robust deep learning framework for identifying **120 distinct dog breeds** using the Stanford Dogs Dataset. The proposed architecture employs an **Ensemble Stacking Strategy**, integrating three state-of-the-art Convolutional Neural Networks (CNNs)—**ResNet50, EfficientNetB0, and MobileNetV2**—as base learners. Their predictions are fused by a meta-learner (Multi-Layer Perceptron) to optimize classification accuracy. The system includes a post-processing knowledge base that retrieves biological traits (lifespan, temperament) and estimates prediction uncertainty.

# 🐶 Large-Scale Dog Breed Classification via Stacked Ensemble Deep Learning

**Author:** Abhinav Basam  
**Institution:** B. V. Raju Institute of Technology, Department of Computer Science and Engineering (AI & ML)  
**Dataset:** Stanford Dogs Dataset (120 Breeds, ~20,000 images)

## 📌 Project Overview
Classifying dog breeds is a notoriously difficult fine-grained image classification task due to high intra-class variation (different poses/colors) and low inter-class variation (breeds that look almost identical). This project implements a **Stacked Generalization (Ensemble) Architecture** to achieve high-accuracy classification across 120 unique breeds.

## 🧠 Model Architecture
This project utilizes a two-tier ensemble approach to maximize feature extraction and classification accuracy:
1. **Tier 1 (Base/Student Models):** Three state-of-the-art Convolutional Neural Networks (CNNs) independently extract features and generate probability distributions.
   * **ResNet50** * **EfficientNetB0** * **MobileNetV2** 2. **Tier 2 (Meta-Learner/Manager):** A Multi-Layer Perceptron (MLP) learns to optimally combine the predictions of the three base models, effectively weighting their "opinions" to produce a final verdict.

## 📊 Performance Metrics (Test Set)
The Stacked Ensemble successfully outperforms all individual base models, demonstrating the power of collaborative network predictions.

| Model | Top-1 Accuracy | Precision | F1-Score | PR-AUC |
| :--- | :--- | :--- | :--- | :--- |
| ResNet50 | 72.83% | 75.08% | 72.38% | 0.8111 |
| MobileNetV2 | 77.57% | 79.14% | 77.07% | 0.8609 |
| EfficientNetB0 | 83.28% | 84.21% | 83.01% | 0.8981 |
| **Proposed Ensemble** | **84.30%** | **85.40%** | **84.25%** | **0.9015** |

## 🚀 Live Demo Interface
The project includes a fully functional, interactive web application built with **Gradio**. 
* **Features:** Upload or capture an image to receive the top 3 predicted breeds, confidence scores, estimated lifespan, and breed personality traits.
* **Inference:** Runs in real-time by loading pre-trained weights from Google Drive without requiring model retraining.

## 🛠️ Technologies Used
* Python 3
* TensorFlow / Keras (Model building, training, and inference)
* Scikit-Learn (Evaluation metrics and PR-AUC calculations)
* Gradio (Web UI deployment)
* Matplotlib / Seaborn (Data visualization and Confusion Matrices)