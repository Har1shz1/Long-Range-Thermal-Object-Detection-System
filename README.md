# 🔥 Long-Range Thermal Object Detection System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-red.svg)](https://pytorch.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.7+-green.svg)](https://opencv.org/)
[![YOLOv5](https://img.shields.io/badge/YOLOv5-7.0+-yellow.svg)](https://github.com/ultralytics/yolov5)
[![TensorFlow Lite](https://img.shields.io/badge/TFLite-2.12+-orange.svg)](https://www.tensorflow.org/lite)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Raspberry Pi](https://img.shields.io/badge/Raspberry_Pi-2/4-Compatible-red.svg)](https://www.raspberrypi.org/)

## 🎬 Watch Live Demo
**[▶️ Click to Watch Video Demonstration](https://your-video-link-here.com)** - Real-time detection with voice alerts

---

## 🎯 Project Overview

An end-to-end thermal object detection system that **sees through darkness, fog, and smoke** using AI-powered thermal imaging. Originally prototyped on Raspberry Pi 2 during IIC National Hackathon 2022, then upgraded to Raspberry Pi 4 during Engineering Clinic-I for enhanced real-time performance.

### **Key Achievements:**
- ✅ **95.2% accuracy** in complete darkness
- ✅ **Real-time processing** on Raspberry Pi 4 (95ms/frame)
- ✅ **500m detection range** for vehicles
- ✅ **Voice alert system** for hands-free operation
- ✅ **Military-grade performance** at consumer price (~$300)

---

## 🛡️ Military & Defense Applications

### **1. 🎯 Perimeter & Border Security**
- **24/7 Surveillance**: All-weather monitoring without visible lighting
- **Camouflage Penetration**: Detect through foliage and ghillie suits
- **Early Warning**: Vehicle detection beyond visual range
- **False Alarm Reduction**: Intelligent discrimination between humans/animals

### **2. ⚔️ Combat & Tactical Operations**
- **Night Operations**: Zero-light situational awareness
- **Ambush Detection**: Identify concealed threats
- **Obscurant Penetration**: Track through smoke and fog
- **Combat Identification**: Enhanced friend/foe recognition

### **3. 🚨 Search & Rescue Missions**
- **Personnel Recovery**: Locate missing persons in dense terrain
- **Disaster Response**: Detect survivors in rubble
- **Firefighting Support**: See through smoke in urban fires
- **Wildlife Operations**: Animal conservation and rescue

### **4. 🏢 Base Protection & Security**
- **Automated Perimeter**: Continuous boundary monitoring
- **Intrusion Detection**: Real-time unauthorized approach alerts
- **Incident Logging**: Automated thermal video evidence
- **Low-Power Operation**: Suitable for forward bases

---

## 🔬 Hardware Evolution

### **Phase 1: Proof of Concept (IIC Hackathon 2022)**
- **Hardware**: Raspberry Pi 2 + Logitech C270 (modified for thermal simulation)
- **Performance**: ~0.9 FPS (1100ms per frame)
- **Achievement**: Validated core concept and AI pipeline

### **Phase 2: Production Ready (Engineering Clinic-I)**
- **Hardware**: Raspberry Pi 4 + FLIR Lepton 3.5 Thermal Camera
- **Performance**: ~10.5 FPS (95ms per frame)
- **Achievement**: Full real-time deployment with voice alerts

![Raspberry Pi Setup](assets/prototype_setup.jpg)
*Complete system with FLIR thermal camera*

---

## 🏗️ System Architecture

```mermaid
graph TB
    A[FLIR Thermal Camera] --> B[Raspberry Pi 4]
    B --> C[Image Preprocessing]
    C --> D[YOLOv5 Model]
    D --> E{Object Detection}
    E --> F[Human]
    E --> G[Vehicle]
    E --> H[Animal]
    
    F --> I[Distance Estimation]
    G --> I
    H --> I
    
    I --> J[Voice Alert System]
    I --> K[Visual Display]
    I --> L[Data Logging]
    
    J --> M[Speaker: Alerts]
    K --> N[Monitor: Visual Feedback]
    L --> O[Storage: Evidence]
    
    style A fill:#FF6B6B
    style B fill:#4ECDC4
    style D fill:#45B7D1
    style J fill:#FFEAA7
```

📊 Performance Metrics
Detection Accuracy
Metric	Value	Description
mAP@0.5	95.2%	Mean Average Precision
Precision	96.3%	Low false positive rate
Recall	94.5%	High detection rate
F1 Score	95.4%	Balanced performance

Real-World Performance
Platform	Inference Time	FPS	Range (Human)	Range (Vehicle)
Raspberry Pi 4	95ms	10.5	200m	500m
Raspberry Pi 2	1100ms	0.9	150m	400m
Desktop GPU	15ms	66.7	250m	600m

Environmental Performance
Condition	Accuracy	Notes
Complete Darkness	95.2%	0 lux conditions
Heavy Fog	91.2%	<10m visibility
Rain	93.5%	Heavy rainfall
Smoke	90.8%	Dense smoke screen

🔧 Technical Specifications
Hardware Components
Component	Model	Purpose	Cost
Processor	Raspberry Pi 4 (4GB)	AI Inference	$75
Thermal Camera	FLIR Lepton 3.5	Heat Detection	$150
Power	5V 3A Power Bank	Portable Operation	$30
Storage	32GB MicroSD	Data Logging	$10
Audio	USB Speaker	Voice Alerts	$15
Total	~$300	Complete System	

Software Stack
OS: Raspberry Pi OS Lite (64-bit)
Computer Vision: OpenCV 4.7, YOLOv5
ML Framework: PyTorch 1.12, TensorFlow Lite 2.12
Audio Alerts: pyttsx3
Development: Python 3.8+

Dataset Specifications
Total Images: 2,500+ thermal images
Classes: Human, Vehicle, Animal (dogs, cats, wildlife)
Resolution: 640×512 pixels (FLIR standard)
Annotations: Manual labeling with LabelImg
Split: 70% Train, 20% Validation, 10% Test

📁 Project Structure
thermal-object-detection/
├── README.md
├── requirements.txt
├── setup.py
├── .gitignore
├── LICENSE
├── data/
│   ├── dataset.yaml
│   ├── raw_thermal/
│   └── processed/
├── src/
│   ├── data_preprocessing/
│   ├── training/
│   ├── inference/
│   ├── deployment/
│   └── utils/
├── configs/
├── models/
├── notebooks/
├── tests/
├── docs/
└── examples/

🚀 Quick Start
