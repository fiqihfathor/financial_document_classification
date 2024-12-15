#!/bin/bash

curl -L -o ./data/scanned-images-dataset-for-ocr-and-vlm-finetuning.zip \
    https://www.kaggle.com/api/v1/datasets/download/suvroo/scanned-images-dataset-for-ocr-and-vlm-finetuning

unzip ./data/scanned-images-dataset-for-ocr-and-vlm-finetuning.zip -d ./data/

echo "Data already downloaded to : ./data/"
