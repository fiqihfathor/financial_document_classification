---
title: About
---

This project is a simple image-based document classification system using deep learning models. The system is built using the FastAPI framework and the PyTorch library. The system is designed to classify documents into one of 10 categories:

- ADVE
- Email
- Form
- Letter
- Memo
- News
- Note
- Report
- Resume
- Scientific

Data used for training and testing the model is from the [Kaggle dataset](https://www.kaggle.com/datasets/suvroo/scanned-images-dataset-for-ocr-and-vlm-finetuning).

The system uses a pre-trained EfficientNet B0 model as the base model and fine-tunes it on a dataset of labeled documents. The system is trained using the Adam optimizer and the cross-entropy loss function. The system is evaluated using the accuracy metric.

Supported architectures: 

- EfficientNet
- ResNet
- Vision Transformer

The system is designed to be a simple and easy-to-use system for classifying documents. The system is not intended to be a production-ready system and should not be used for production purposes. The system is intended to be a proof-of-concept and should be used as a starting point for building a more robust document classification system.

The system is built and maintained by the following person:

- Fiqih Fathor Rachim ([fiqihfathor](https://fiqihfathor.github.io))
