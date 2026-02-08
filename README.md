
# Fine-Grained Dog Breed Classification using Ensemble Transfer Learning

![Project Status](https://img.shields.io/badge/Status-Completed-green)
![Framework](https://img.shields.io/badge/Framework-TensorFlow%20%7C%20Keras-orange)
![License](https://img.shields.io/badge/License-MIT-blue)

## 📄 Abstract
Fine-grained image classification remains a significant challenge in computer vision due to low inter-class variance and high intra-class variance. This project presents a robust deep learning framework for identifying **120 distinct dog breeds** using the Stanford Dogs Dataset. The proposed architecture employs an **Ensemble Stacking Strategy**, integrating three state-of-the-art Convolutional Neural Networks (CNNs)—**ResNet50, EfficientNetB0, and MobileNetV2**—as base learners. Their predictions are fused by a meta-learner (Multi-Layer Perceptron) to optimize classification accuracy. The system includes a post-processing knowledge base that retrieves biological traits (lifespan, temperament) and estimates prediction uncertainty.

## 1. Introduction
Dog breed identification is a classic fine-grained visual categorization (FGVC) problem. Distinguishing between breeds such as the *Norfolk Terrier* and *Norwich Terrier* requires capturing subtle discriminative features (e.g., ear shape, snout structure) while remaining invariant to pose, background, and lighting.

This repository contains the implementation of a **Multi-Model Ensemble Classifier** that leverages Transfer Learning to overcome data scarcity and computational constraints. The final application is deployed via a Gradio web interface, offering real-time inference and biological metadata retrieval.

## 2. Methodology

### 2.1 Dataset
The model was trained on the **Stanford Dogs Dataset** [1], comprising 20,580 images across 120 classes.
* **Preprocessing:** Images were resized to $224 \times 224$ pixels.
* **Augmentation:** To prevent overfitting, the training pipeline included random rotations ($\pm 20^\circ$), width/height shifts (0.2), and horizontal flips.

### 2.2 Model Architecture
We utilized a **Stacking Ensemble** approach:
1.  **Base Learners (Level-0 Models):**
    * **ResNet50:** Utilizes residual connections to solve the vanishing gradient problem in deep networks.
    * **EfficientNetB0:** Optimizes depth, width, and resolution using a compound scaling coefficient.
    * **MobileNetV2:** Uses inverted residuals and linear bottlenecks for efficient low-latency inference.
    * *Note:* All base learners were pre-trained on ImageNet and fine-tuned on the target dataset.

2.  **Meta-Learner (Level-1 Model):**
    * The Softmax output vectors from the three base learners are concatenated to form a feature vector of size $120 \times 3 = 360$.
    * A dense neural network (MLP) processes this vector to learn the optimal weighting for each expert model, producing the final probability distribution.

### 2.3 Inference Pipeline
The inference system includes an **Uncertainty Filter**. If the maximum confidence score $C_{max} < 0.5$ (50%), the prediction is flagged as "Uncertain," mitigating false positives in out-of-distribution samples.

## 3. Directory Structure