# Enhanced Dog Breed Classification using Stacked Ensemble Deep Learning

![License](https://img.shields.io/badge/license-MIT-blue) ![Python](https://img.shields.io/badge/python-3.8%2B-blue) ![TensorFlow](https://img.shields.io/badge/framework-TensorFlow-orange)

## Abstract
Fine-grained image classification presents a significant challenge in computer vision due to high inter-class similarity and background noise. This project implements a **Hybrid Inference Pipeline** that integrates an object detection module (MobileNet-SSD) for background clutter mitigation with a **Stacked Ensemble of Convolutional Neural Networks (CNNs)** for classification.

The system features a **Dual-Mode Inference Engine** capable of processing both static image uploads and real-time webcam feeds. By aggregating feature maps from ResNet50, EfficientNetB0, and MobileNetV2, the model achieves superior generalization performance (>90% accuracy) and provides a probabilistic breakdown of the top-3 predicted breeds.

## System Architecture
The solution follows a multi-stage pipeline:

1.  **Input Processing:**
    * **Static Mode:** Accepts standard image formats (JPG, PNG).
    * **Real-Time Mode:** JavaScript bridge (Cloud) or OpenCV integration (Local) captures live video frames.
2.  **Object Detection (Smart Crop):** An OpenCV DNN module (MobileNet-SSD) identifies the subject's bounding box and crops the Region of Interest (ROI), eliminating environmental noise.
3.  **Ensemble Classification:** The ROI is processed by three transfer-learning models:
    * **ResNet50:** Deep residual learning for semantic feature capture.
    * **EfficientNetB0:** Compound scaling for balanced accuracy/efficiency.
    * **MobileNetV2:** Lightweight architecture for rapid inference.
4.  **Meta-Learning:** A Logistic Regression meta-learner aggregates the probability vectors to output the final classification.

## Repository Structure

| File Name | Description |
| :--- | :--- |
| `dogcnndrive.ipynb` | **Training Pipeline:** Implements data augmentation, transfer learning, and the stacking ensemble strategy. Includes automated model checkpointing and early stopping. |
| `demo_notebook.ipynb` | **Cloud Inference:** A Google Colab-optimized notebook featuring a JavaScript-based webcam bridge and Google Drive integration for prediction history logging. |
| `app.py` | **Desktop Application:** A standalone GUI application built with Tkinter. Features real-time camera capture, Top-3 probability visualization, and privacy-aware local storage. |
| `requirements.txt` | **Dependencies:** List of required Python libraries. |

## Getting Started

### Prerequisites
* Python 3.8+
* Git
* Webcam (optional, for real-time features)

### Installation

1.  **Clone the Repository**
    ```bash
    git clone [https://github.com/AbhinavBasam/dogclassifier.git](https://github.com/AbhinavBasam/dogclassifier.git)
    cd dogclassifier
    ```

2.  **Download Trained Models**
    The model weights are hosted externally due to size constraints.
    * **Download Link:** [INSERT YOUR GOOGLE DRIVE LINK HERE]
    * **Action:** Extract the contents into a folder named `Dog_Models_Backup` in the project root.

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

### Usage

**Option 1: Local Desktop App**
Run the application to launch the Graphical User Interface:
```bash
python app.py