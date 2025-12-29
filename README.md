
# 🎯  Long-Range Thermal Object Detection System

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-red.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.7+-green.svg)
![YOLOv5](https://img.shields.io/badge/YOLOv5-7.0+-yellow.svg)
![TFLite](https://img.shields.io/badge/TFLite-2.12+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)
![Raspberry Pi](https://img.shields.io/badge/Raspberry_Pi-Compatible-red.svg)

The system underwent two major development phases. Initially prototyped on Raspberry Pi 2 hardware during the IIC National Hackathon 2022 (second semester), it demonstrated proof-of-concept viability. Subsequently, during Engineering Clinic-I (fourth semester), the platform was upgraded to Raspberry Pi 4, enabling enhanced real-time capabilities and full feature deployment.
---

## 🎖️ Overview
The Long-Range Thermal Object Detection System is an end-to-end AI-powered surveillance and situational awareness platform designed for low-visibility environments such as night-time, fog, smoke, and adverse weather conditions. It integrates thermal imaging, deep learning, and edge deployment to provide real-time detection and voice-based alerts for humans, animals (cats, dogs, and etc), vehicles and many more.
**Military-Grade Detection:** Designed for defense applications with 95.2% accuracy in complete darkness

---

### 🔬 Prototype Hardware Setup

<img width="1536" height="1024" alt="rp-1" src="https://github.com/user-attachments/assets/92e8d2b0-619f-4723-b28d-13920a9e7f08" />

## 🔧 Technical Details
## Dataset Specifications

- **Total Images**: 2,500+ thermal images
- **Long-Range Detection**: Up to 500m range for vehicles, 200m for humans
- **Optimized YOLOv5 Model**: Achieves ~80% mAP on thermal imagery
- **Real-time Inference**: <100ms per frame on Raspberry Pi 4
- **Edge Deployment**: TensorFlow Lite optimized for ARM architecture
- **Voice Alert System**: pyttsx3 integration for situational awareness
- **Production Ready**: Complete pipeline from data collection to deployment
---
---
## 🏗️ System Architectur
<img width="3350" height="2550" alt="system design" src="https://github.com/user-attachments/assets/43bf7ea7-774a-437d-b504-85601b8479a4" />

---

## 📊 Performance Metrics

| Metric | Value | Notes |
|--------|-------|-------|
| **Mean Average Precision (mAP@0.5)** | 79.8% | Primary accuracy metric |
| **Precision** | 81.2% | Low false positive rate |
| **Recall** | 78.5% | Good detection coverage |
| **F1 Score** | 79.8% | Balanced precision-recall |
| **Inference Time (RPi 4)** | 95ms | ~10.5 FPS real-time |
| **Inference Time (RPi 2)** | 1100ms | ~0.9 FPS proof-of-concept |
| **Model Size (TFLite)** | 14.2 MB | Optimized for edge deployment |
| **Classes** | 3 | Human, Vehicle, Animal |
| **Input Resolution** | 320×320 | Balanced speed/accuracy |

---

## 📊 Performance Summary
- Accuracy: ~95%
- Real-time inference on Raspberry Pi
- Portable, low-power deployment

---

## 📁 Project Structure
```
thermal-object-detection/
├── README.md
├── requirements.txt
├── setup.py
├── .gitignore
├── LICENSE
├── data/
│   ├── raw_thermal/
│   │   ├── images/
│   │   └── annotations/
│   ├── processed/
│   └── dataset.yaml
├── notebooks/
│   ├── 01_dataset_exploration.ipynb
│   ├── 02_preprocessing_pipeline.ipynb
│   └── 03_model_training.ipynb
├── src/
│   ├── data_preprocessing/
│   │   ├── thermal_preprocessor.py
│   │   ├── data_augmentation.py
│   │   └── dataset_splitter.py
│   ├── training/
│   │   ├── train_yolo.py
│   │   ├── hyperparameter_tuning.py
│   │   └── evaluate_model.py
│   ├── inference/
│   │   ├── realtime_inference.py
│   │   ├── tflite_converter.py
│   │   └── video_processor.py
│   ├── deployment/
│   │   ├── raspberry_pi_setup.py
│   │   ├── voice_alert_system.py
│   │   └── camera_interface.py
│   └── utils/
│       ├── visualization.py
│       ├── metrics.py
│       └── logger.py
├── configs/
│   ├── training_config.yaml
│   ├── inference_config.yaml
│   └── deployment_config.yaml
└── examples/
    ├── sample_thermal_images/
    ├── demo_videos/
    └── output_results/
```

---

## 🚀 Getting Started
```bash
pip install -r requirements.txt
python src/inference/realtime_inference.py
```
## Sample Thermal Images:
![thermal image 01](https://github.com/user-attachments/assets/0bfd7b84-cde9-45dd-b895-fc43b943fb16)
![thermal image 02](https://github.com/user-attachments/assets/bf323e71-7eda-4005-a199-1db27870690a)
![thermal image 03](https://github.com/user-attachments/assets/d6241e2f-b5d7-41eb-a9ee-e99da0aba831)


## 🎥 Live Demo 

### 📹 Working Video Demonstration
[![Watch Demo Video](https://img.shields.io/badge/Watch_Demo_Video-FF0000?style=for-the-badge&logo=youtube&logoColor=white)](https://drive.google.com/file/d/1YGFaZXEtxB2ZRpKVfEMVeDUM4K6kqQZ0/view?usp=sharing)

*Raspberry Pi 2 with Logitech C270 HD Web Camera*
---

## 📜 License
MIT License
