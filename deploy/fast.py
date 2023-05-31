from pydantic import BaseModel
from fastapi import FastAPI, File, UploadFile, Body
from fastapi.responses import FileResponse
import io
from typing import Dict, List, Any
import time
from datetime import datetime, timedelta

import sys
sys.path.append('../lightning')
from lightning_module import LitMNIST
import torch
from torchvision.transforms import ToTensor
from PIL import Image

app = FastAPI()
# from fastapi.middleware.cors import CORSMiddleware
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"]
# )

transform = ToTensor()
model = LitMNIST.load_from_checkpoint('../models/best.ckpt')
model.to('cuda')
print("model ready")

@app.post("/predict")
async def predict(file: UploadFile):
    img_file = file.file
    img_bytes = img_file.read()
    
    img = Image.open(io.BytesIO(img_bytes))
    
    print(img.size)
    img_tensor = transform(img)
    print(img_tensor.shape)
    
    img_tensor = img_tensor.unsqueeze(0)
    print(img_tensor.shape)
    logits = model(img_tensor.to(model.device))
    preds = torch.argmax(logits, dim=1)
    pred = preds.tolist()[0]
    return {"prediction" : pred}

@app.get("/ping")
def ping():
    return {"message": "pong!"}


import pathlib
import uvicorn
if __name__ == "__main__":
    uvicorn.run('__main__:app', host="0.0.0.0", port=40071)