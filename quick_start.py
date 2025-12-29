"""
Quick Start Script for Thermal Object Detection
"""

print("=" * 60)
print("Thermal Object Detection System - Quick Start")
print("=" * 60)

print("\n Installing dependencies...")
print("Run: pip install -r requirements.txt")

print("\n To train the model:")
print("python src/training/train_yolo.py --config configs/training_config.yaml")

print("\n To run inference on an image:")
print("python src/inference/realtime_inference.py --model models/best.pt --input sample.jpg")

print("\n To deploy on Raspberry Pi:")
print("python src/deployment/raspberry_pi_setup.py --config configs/deployment_config.yaml")

print("\n For Jupyter notebooks:")
print("jupyter notebook notebooks/")

print("\n Setup complete!")
