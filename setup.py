"""
Setup script for Thermal Object Detection System
"""

from setuptools import setup, find_packages
import os

# Read requirements
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

# Read README
with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="thermal-object-detection",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Long-Range Thermal Object Detection System with YOLOv5",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/thermal-object-detection",
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        'dev': [
            'pytest',
            'pytest-cov',
            'black',
            'flake8',
            'mypy',
            'jupyter',
        ],
        'raspberry': [
            'RPi.GPIO',
            'picamera2',
            'spidev',
            'smbus2',
        ],
        'gpu': [
            'torch==1.13.1+cu117',
            'torchvision==0.14.1+cu117',
        ],
    },
    entry_points={
        'console_scripts': [
            'thermal-train=src.training.train_yolo:main',
            'thermal-detect=src.inference.realtime_inference:main',
            'thermal-deploy=src.deployment.raspberry_pi_setup:main',
        ],
    },
    include_package_data=True,
    package_data={
        '': ['*.yaml', '*.yml', '*.json', '*.txt'],
    },
)
