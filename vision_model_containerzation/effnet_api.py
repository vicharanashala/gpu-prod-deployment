import os
import io
import torch
import timm
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from torchvision import transforms
from pydantic import BaseModel
from typing import List, Dict

# --- Configuration ---
MODEL_PATH = os.getenv("MODEL_PATH", "effnet_model/best_model_50epochs.pth")
NUM_CLASSES = 13
IMG_SIZE = 380
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CLASS_NAMES = [
    'Potato Early Blight',
    'Potato Healthy',
    'Potato Late Blight',
    'Tomato Bacterial Spot',
    'Tomato Early Blight',
    'Tomato Healthy',
    'Tomato Late Blight',
    'Tomato Leaf Mold',
    'Tomato Mosaic Virus',
    'Tomato Septoria Leaf Spot',
    'Tomato Spider Mites Two Spotted Spider Mite',
    'Tomato Target Spot',
    'Tomato Yellow Leaf Curl Virus'
]

# --- App Initialization ---
app = FastAPI(title="EfficientNet Leaf Disease API")

# --- Model Loading ---
print(f"Loading model on {DEVICE}...")
model = timm.create_model('efficientnet_b4', pretrained=False, num_classes=NUM_CLASSES)
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    print("Model loaded successfully.")
else:
    print(f"ERROR: Model file not found at {MODEL_PATH}")
model.to(DEVICE)
model.eval()

# --- Transforms ---
preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- Schmeas ---
class PredictionResponse(BaseModel):
    class_name: str
    confidence: float
    probabilities: Dict[str, float]

@app.get("/")
async def root():
    return {"message": "EfficientNet Leaf Disease API is running"}

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        content = await file.read()
        image = Image.open(io.BytesIO(content)).convert("RGB")
        
        input_tensor = preprocess(image).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
            
        conf, pred_idx = torch.max(probabilities, 0)
        
        prob_dict = {CLASS_NAMES[i]: float(probabilities[i]) for i in range(len(CLASS_NAMES))}
        
        return PredictionResponse(
            class_name=CLASS_NAMES[pred_idx.item()],
            confidence=float(conf),
            probabilities=prob_dict
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8029)
