import os
import time
import uuid
import torch
import timm
import base64
import io
import uvicorn
import requests
from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form, Response, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, ConfigDict
from typing import Optional, List, Dict, Any
from PIL import Image
from torchvision import transforms
from contextlib import asynccontextmanager

# --- Configuration ---
LEAF_MODEL_PATH = os.getenv("LEAF_MODEL_PATH", "effnet_model/best_model_50epochs.pth")
RICE_MODEL_PATH = os.getenv("RICE_MODEL_PATH", "effnet_model/best_model_weightedrandomsampler_b4.pth")
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

# --- In-Memory Storage ---
FILE_STORAGE = {}

# --- Global State ---
leaf_model = None
rice_model = None
transform = None

# --- Models ---
class DocumentSource(BaseModel):
    model_config = ConfigDict(extra='ignore')
    type: Optional[str] = None
    image_url: Optional[str] = None
    document_url: Optional[str] = None

class OCRRequest(BaseModel):
    model_config = ConfigDict(extra='ignore')
    model: Optional[str] = "mistral-ocr-latest"
    document: DocumentSource
    id: Optional[str] = None 

class PageDimensions(BaseModel):
    dpi: int = 72
    height: int
    width: int

class OCRPage(BaseModel):
    index: int
    markdown: str
    images: List[Any] = []
    dimensions: PageDimensions 

class OCRResponse(BaseModel):
    pages: List[OCRPage]
    model: str

class FileObject(BaseModel):
    id: str
    object: str = "file"
    bytes: int
    created_at: int
    filename: str
    purpose: str

class FileURL(BaseModel):
    url: str

# --- Lifespan ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global leaf_model, rice_model, transform
    print(f"Loading models...")
    try:
        transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Load Leaf Model
        leaf_model = timm.create_model('efficientnet_b4', pretrained=False, num_classes=len(LEAF_CLASS_NAMES))
        if os.path.exists(LEAF_MODEL_PATH):
            leaf_model.load_state_dict(torch.load(LEAF_MODEL_PATH, map_location=DEVICE))
            print(f"Leaf model loaded from {LEAF_MODEL_PATH}")
        else:
            print(f"WARNING: Leaf model not found at {LEAF_MODEL_PATH}")
        leaf_model.to(DEVICE)
        leaf_model.eval()

        # Load Rice Model
        rice_model = timm.create_model('efficientnet_b4', pretrained=False, num_classes=len(RICE_CLASS_NAMES))
        if os.path.exists(RICE_MODEL_PATH):
            rice_model.load_state_dict(torch.load(RICE_MODEL_PATH, map_location=DEVICE))
            print(f"Rice model loaded from {RICE_MODEL_PATH}")
        else:
            print(f"WARNING: Rice model not found at {RICE_MODEL_PATH}")
        rice_model.to(DEVICE)
        rice_model.eval()

    except Exception as e:
        print(f"Error loading models: {e}")
    yield

app = FastAPI(lifespan=lifespan)

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    # Sanitize errors to avoid UnicodeDecodeError on binary inputs
    errors = exc.errors() 
    for error in errors:
        if "input" in error:
            inp = error["input"]
            if isinstance(inp, bytes):
                error["input"] = f"<bytes length={len(inp)}>"
            elif isinstance(inp, str) and len(inp) > 500:
                error["input"] = inp[:100] + "..."
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": jsonable_encoder(errors)},
    )

# --- Logging ---
@app.middleware("http")
async def log_requests(request: Request, call_next):
    print(f"INFO: {request.method} {request.url.path}")
    response = await call_next(request)
    print(f"INFO: Response Status: {response.status_code}")
    return response

# --- Helper Functions (CRITICAL FIX HERE) ---
def decode_image_from_source(source: DocumentSource) -> Image.Image:
    url = source.image_url or source.document_url
    if not url:
        raise HTTPException(status_code=400, detail="No image URL provided")

    print(f"DEBUG: Processing URL: {url}")

    # 1. Handle Base64
    if url.startswith("data:image"):
        try:
            header, encoded = url.split(",", 1)
            return Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB")
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid Base64 string")

    # 2. [FIX] Handle Internal URLs (PREVENTS DEADLOCK)
    # If the URL contains '/v1/files/', we assume it's one of ours.
    if "/v1/files/" in url and "/content" in url:
        try:
            # Parse ID from URL like: .../v1/files/{file-id}/content
            parts = url.split("/v1/files/")
            if len(parts) > 1:
                # Get the ID (everything before /content)
                file_id = parts[1].split("/content")[0]
                
                if file_id in FILE_STORAGE:
                    print(f"DEBUG: Internal file detected ({file_id}). Loading directly from RAM.")
                    file_data = FILE_STORAGE[file_id]
                    return Image.open(io.BytesIO(file_data["bytes"])).convert("RGB")
                else:
                    print(f"DEBUG: Internal file ID {file_id} not found in RAM.")
        except Exception as e:
            print(f"DEBUG: Failed to parse internal URL: {e}")

    # 3. Handle External URLs (Only if step 2 didn't return)
    print("DEBUG: Fetching via HTTP (External)...")
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        return Image.open(io.BytesIO(resp.content)).convert("RGB")
    except Exception as e:
        print(f"Error decoding image: {e}")
        raise HTTPException(status_code=400, detail=f"Image load failed: {str(e)}")

def generate_markdown_report(prediction: str, confidence: float, probabilities: torch.Tensor, class_names: List[str]) -> str:
    table_rows = []
    probs_list = probabilities[0].tolist()
    sorted_indices = sorted(range(len(probs_list)), key=lambda k: probs_list[k], reverse=True)
    
    for idx in sorted_indices:
        if probs_list[idx] > 0.01:
            table_rows.append(f"| {class_names[idx]} | {probs_list[idx]:.2%} |")

    table_content = "\n".join(table_rows)
    return f"""# Plant Disease Analysis Report
**Diagnosis:** {prediction}
**Confidence:** {confidence:.2%}

## Probability Distribution
| Class Name | Probability |
| :--- | :--- |
{table_content}
"""

# --- Endpoints ---

@app.post("/v1/files", response_model=FileObject)
async def upload_file(file: UploadFile = File(...), purpose: str = Form("ocr")):
    file_id = f"file-{uuid.uuid4()}"
    content = await file.read()
    
    FILE_STORAGE[file_id] = {
        "bytes": content,
        "filename": file.filename,
        "content_type": file.content_type
    }
    
    print(f"DEBUG: File stored in RAM: {file_id}")
    
    return FileObject(
        id=file_id,
        bytes=len(content),
        created_at=int(time.time()),
        filename=file.filename,
        purpose=purpose
    )

@app.get("/v1/files/{file_id}/url", response_model=FileURL)
async def get_file_url(file_id: str, request: Request):
    if file_id not in FILE_STORAGE:
        raise HTTPException(status_code=404, detail="File not found")
    
    # Return URL pointing to this server
    base_url = str(request.base_url).rstrip("/")
    download_url = f"{base_url}/v1/files/{file_id}/content"
    return FileURL(url=download_url)

@app.get("/v1/files/{file_id}/content")
async def serve_file_content(file_id: str):
    if file_id not in FILE_STORAGE:
        raise HTTPException(status_code=404, detail="File not found")
    
    file_data = FILE_STORAGE[file_id]
    return Response(
        content=file_data["bytes"], 
        media_type=file_data["content_type"]
    )

@app.delete("/v1/files/{file_id}")
async def delete_file(file_id: str):
    if file_id in FILE_STORAGE:
        del FILE_STORAGE[file_id]
        return {"deleted": True, "id": file_id}
    return JSONResponse(status_code=404, content={"detail": "File not found"})

@app.post("/v1/ocr", response_model=OCRResponse)
@app.post("/ocr", response_model=OCRResponse)
async def process_ocr(request: OCRRequest):
    if leaf_model is None or rice_model is None:
        raise HTTPException(status_code=503, detail="Models not loaded")

    try:
        # Decode Image (Handles deadlock automatically)
        image = decode_image_from_source(request.document)
        img_width, img_height = image.size 
        
        input_tensor = transform(image).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            # Leaf Inference
            leaf_outputs = leaf_model(input_tensor)
            leaf_probs = torch.nn.functional.softmax(leaf_outputs, dim=1)
            leaf_conf, leaf_idx = torch.max(leaf_probs, 1)
            
            # Rice Inference
            rice_outputs = rice_model(input_tensor)
            rice_probs = torch.nn.functional.softmax(rice_outputs, dim=1)
            rice_conf, rice_idx = torch.max(rice_probs, 1)

        # Compare and Select Best
        if leaf_conf.item() > rice_conf.item():
            winner_conf = leaf_conf.item()
            winner_idx = leaf_idx.item()
            winner_probs = leaf_probs
            winner_class_names = LEAF_CLASS_NAMES
        else:
            winner_conf = rice_conf.item()
            winner_idx = rice_idx.item()
            winner_probs = rice_probs
            winner_class_names = RICE_CLASS_NAMES

        markdown_content = generate_markdown_report(
            winner_class_names[winner_idx], 
            winner_conf, 
            winner_probs,
            winner_class_names
        )
        
        return OCRResponse(
            model=request.model,
            pages=[
                OCRPage(
                    index=0,
                    markdown=markdown_content,
                    images=[],
                    dimensions=PageDimensions(dpi=72, height=img_height, width=img_width)
                )
            ]
        )
    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"Error in OCR: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8029)