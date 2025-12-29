
# 🔥 Long-Range Thermal Object Detection System

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-red.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.7+-green.svg)
![YOLOv5](https://img.shields.io/badge/YOLOv5-7.0+-yellow.svg)
![TFLite](https://img.shields.io/badge/TFLite-2.12+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)
![Raspberry Pi](https://img.shields.io/badge/Raspberry_Pi-Compatible-red.svg)

The project was initially developed during the IIC National Hackathon 2022 in my second semester and was subsequently extended and refined as part of Engineering Clinic-I in my fourth semester.
---

## 🎯 Overview
The **Long-Range Thermal Object Detection System** is an end-to-end **AI-powered surveillance and situational awareness platform** designed for **low-visibility environments** such as night-time, fog, smoke, and adverse weather conditions.  
It integrates **thermal imaging, deep learning, and edge deployment** to provide **real-time detection and voice-based alerts** for humans, animals (cats, dogs, and etc), vehicles and many more.

---

## 🧠 Key Capabilities
- Works in **complete darkness** using thermal imaging  
- Robust to **fog, smoke, rain, and low-light conditions**  
- **Automatic classification** of Humans, Animals, and Vehicles  
- **Hands-free voice alerts** for real-time situational awareness  
- **Edge deployment** on low-cost hardware  

---

## 🏗️ System Architecture
```
[FLIR Thermal Camera] → [Raspberry Pi] → [YOLOv5 Inference] → [Voice Alerts]
```

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

---

## 📜 License
MIT License
