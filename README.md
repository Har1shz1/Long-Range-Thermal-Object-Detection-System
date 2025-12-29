
# 🔥 Long-Range Thermal Object Detection System

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-red.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.7+-green.svg)
![YOLOv5](https://img.shields.io/badge/YOLOv5-7.0+-yellow.svg)
![TFLite](https://img.shields.io/badge/TFLite-2.12+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)
![Raspberry Pi](https://img.shields.io/badge/Raspberry_Pi-Compatible-red.svg)

This project was initially developed during the IIC National Hackathon and was subsequently extended and refined as part of Engineering Clinic-I in my fourth semester.
---

## 🎯 Overview
The **Long-Range Thermal Object Detection System** is an end-to-end **AI-powered surveillance and situational awareness platform** designed for **low-visibility environments** such as night-time, fog, smoke, and adverse weather conditions.  
It integrates **thermal imaging, deep learning, and edge deployment** to provide **real-time detection and voice-based alerts** for humans, animals, and vehicles.

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
Thermal-Object-Detection/
├── data/
├── src/
├── models/
├── docs/
├── requirements.txt
└── README.md
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
