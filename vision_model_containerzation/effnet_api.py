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
LEAF_MODEL_PATH = os.getenv("LEAF_MODEL_PATH", "effnet_model/best_model_50epochs.pth")
RICE_MODEL_PATH = os.getenv("RICE_MODEL_PATH", "effnet_model/best_model_b4.pth")
IMG_SIZE = 380
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LEAF_CLASS_NAMES = [
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

RICE_CLASS_NAMES = [
    'Bacterial Leaf Blight',
    'Bacterial Streak',
    'Bakanae',
    'Brown Spot',
    'False Smut',
    'Grassy Stunt Virus',
    'Healthy Leaf',
    'Hispa',
    'Insect Affected',
    'Leaf Blast',
    'Leaf Scald',
    'Leaf Smut',
    'Narrow Brown Spot',
    'Neck Blast',
    'Ragged Stunt Virus',
    'Sheath Blight',
    'Sheath Rot',
    'Stem Rot',
    'Tungro'
]

# --- App Initialization ---
app = FastAPI(title="EfficientNet Disease Classification API")

# --- Model Loading ---
print(f"Loading models on {DEVICE}...")

def load_model(path, num_classes):
    model = timm.create_model('efficientnet_b4', pretrained=False, num_classes=num_classes)
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=DEVICE))
        print(f"Model loaded from {path}")
    else:
        print(f"ERROR: Model file not found at {path}")
    model.to(DEVICE)
    model.eval()
    return model

leaf_model = load_model(LEAF_MODEL_PATH, len(LEAF_CLASS_NAMES))
rice_model = load_model(RICE_MODEL_PATH, len(RICE_CLASS_NAMES))

# --- Transforms ---
preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- Schemas ---
class PredictionResponse(BaseModel):
    class_name: str
    confidence: float
    probabilities: Dict[str, float]

@app.get("/")
async def root():
    return {"message": "EfficientNet Disease Classification API is running"}

async def run_inference(file: UploadFile, model, class_names):
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
        
        prob_dict = {class_names[i]: float(probabilities[i]) for i in range(len(class_names))}
        
        return PredictionResponse(
            class_name=class_names[pred_idx.item()],
            confidence=float(conf),
            probabilities=prob_dict
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/leaf", response_model=PredictionResponse)
async def predict_leaf(file: UploadFile = File(...)):
    return await run_inference(file, leaf_model, LEAF_CLASS_NAMES)

@app.post("/predict/rice", response_model=PredictionResponse)
async def predict_rice(file: UploadFile = File(...)):
    return await run_inference(file, rice_model, RICE_CLASS_NAMES)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8029)
