Leaf Disease Detection Using Convolutional Neural Networks (CNN)

This project implements a Deep Learning–based plant leaf disease detection system using a Convolutional Neural Network (CNN).
It automatically classifies leaf images into healthy or diseased categories, helping in early detection of plant diseases in agriculture.

Project Objective

Early detection of plant diseases is critical for improving crop yield and reducing losses.
This project aims to:

Analyze leaf images

Detect diseases using a CNN model

Provide accurate classification results

Key Features

 Image-based plant disease classification

 CNN architecture built using TensorFlow & Keras

 Separate scripts for training, evaluation, and prediction

 Modular and clean code structure

 Easily extendable for new plant diseases

Technology Stack

Programming Language: Python

Deep Learning Framework: TensorFlow / Keras



leaf-disease-detection-cnn/
│
├── train.py              # Trains the CNN model
├── evaluate.py           # Evaluates model performance
├── predict.py            # Predicts disease from an image
├── model.py              # CNN architecture
├── utils.py              # Data loading & preprocessing
├── config.py             # Paths and hyperparameters
├── saved_model/           # Saved trained model
│
├── requirements.txt
├── .gitignore
└── README.md
Libraries: NumPy, OpenCV, Matplotlib

Model Type: Convolutional Neural Network (CNN)
