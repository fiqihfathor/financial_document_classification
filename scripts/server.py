from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import Any
from PIL import Image
import mlflow
import mlflow.pytorch
import torch
import torchvision.transforms as transforms
import io
import numpy as np
from src.utils.config import load_config

config = load_config()

app = FastAPI()

MODEL_PATH = config["fastapi"]["model_path"]

model = mlflow.pytorch.load_model(MODEL_PATH)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

class_names = ["ADVE","Email","Form","Letter","Memo","News","Note","Report","Resume","Scientific"]

class PredictionRequest(BaseModel):
    input_data: Any

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Endpoint for prediction
    """
    try:
        if file.content_type not in ["image/jpeg", "image/png"]:
            raise HTTPException(status_code=400, detail="Unsupported file type")

        image = Image.open(io.BytesIO(await file.read()))
        processed_image = preprocess_image(image)

        input_tensor = processed_image.to(device)

        with torch.no_grad():
            prediction = model(input_tensor)

        predicted_class_idx = torch.argmax(prediction, dim=1).item()

        predicted_class_name = class_names[predicted_class_idx]
        return {"prediction": predicted_class_name}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

def preprocess_image(image: Image.Image):
    """
    Preprocess the input image
    """
    if image.mode != 'RGB':
        image = image.convert('RGB')

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(), 
    ])
    
    image_tensor = transform(image)
    return image_tensor.unsqueeze(0)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=config["fastapi"]["host"],
        port=config["fastapi"]["port"],
    )