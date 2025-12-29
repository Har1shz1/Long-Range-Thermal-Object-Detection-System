# 🎯  Long-Range Thermal Object Detection System

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-red.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.7+-green.svg)
![YOLOv5](https://img.shields.io/badge/YOLOv5-7.0+-yellow.svg)
![TFLite](https://img.shields.io/badge/TFLite-2.12+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)
![Raspberry Pi](https://img.shields.io/badge/Raspberry_Pi-Compatible-red.svg)

## 🎬 Watch Live Demo
[![Watch Demo Video](https://img.shields.io/badge/Watch_Demo_Video-FF0000?style=for-the-badge&logo=youtube&logoColor=white)](https://drive.google.com/file/d/1YGFaZXEtxB2ZRpKVfEMVeDUM4K6kqQZ0/view?usp=sharing)
---

## 🎯 Project Overview
An end-to-end thermal object detection system that **sees through darkness, fog, and smoke** using AI-powered thermal imaging. Originally prototyped on Raspberry Pi 2 during IIC National Hackathon 2022, then upgraded to Raspberry Pi 4 during Engineering Clinic-I for enhanced real-time performance.

## 🛡️ Applications
- Perimeter & Border Security
- Night-time Surveillance
- Search & Rescue Operations
- Defense & Tactical Monitoring
- Wildlife Monitoring

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


## 🔬 Hardware Evolution
**Phase 1:** Raspberry Pi 2 – Proof of concept  
**Phase 2:** Raspberry Pi 4 + FLIR Lepton 3.5 – Real-time deployment

## 🏗️ System Architecture
Thermal Camera → Raspberry Pi → Preprocessing → YOLOv5 → Detection → Alerts & Logging

## 📊 Performance Metrics
- Accuracy: 95.2%
- FPS (Raspberry Pi 4): ~10.5
- Detection Range: Human (200m), Vehicle (500m)

## 🔧 Tech Stack
- Python 3.8+
- PyTorch, TensorFlow Lite
- OpenCV
- YOLOv5
- Raspberry Pi OS

## 📁 Project Structure
See repository for structured folders covering data, training, inference, and deployment.

## 🚀 Usage
Clone repository, install requirements, and run real-time inference module.

## 📈 Development Timeline
- 2022: Research & Prototyping
- 2023: Real-time System Development
- 2024: Optimization & Deployment

## 📄 License
MIT License

## 📞 Contact
Email: your.email@example.com  
GitHub: yourusername
