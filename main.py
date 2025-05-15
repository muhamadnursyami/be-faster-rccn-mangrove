from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from PIL import Image
import torch
import json
import numpy as np
import io

# =====================================
# 🚀 INIT FASTAPI APP
# =====================================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================
# 📚 LABEL MAPPING
# =====================================
COCO_CLASSES = {
    0: "Background",
    1: "Avicennia-lanata",
    2: "Bruguiera-cylindrica",
    3: "Bruguiera-gymnorrhiza",
    4: "Lumnitzera-liitorea",
    5: "Rhizophora-apiculata",
    6: "Rhizophora-mucronata",
    7: "Scyphiphora-hydrophyllacea",
    8: "Sonneratia-alba",
    9: "Xylocarpus-granatum",
}

# =====================================
# 🔧 LOAD MODEL
# =====================================
def load_model(weights_path, num_classes=10, device="cpu"):
    weights = FasterRCNN_ResNet50_FPN_Weights.COCO_V1
    model = fasterrcnn_resnet50_fpn(weights=weights)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# Load model saat startup
MODEL_PATH = "best_model.pth"  # Ganti sesuai path anda
model = load_model(MODEL_PATH)
device = "cpu"

# =====================================
# 🔍 LOAD JSON DATA
# =====================================
with open("dataTanaman.json") as f:
    tanaman_data = json.load(f)

def get_tanaman_by_label(label_name):
    for data in tanaman_data:
        if label_name.lower().replace(" ", "-") in data["nama"].lower().replace(" ", "-"):
            return data
    return None

# =====================================
# 🛬 ROUTES
# =====================================
@app.get("/")
def root():
    return {"message": "FastAPI backend for Mangrove Faster R-CNN ready!"}

@app.post("/mangrove/detect")
async def detect_image(file: UploadFile = File(...), threshold: float = 0.6):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    image_tensor = F.to_tensor(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)[0]

    results = []
    for box, label, score in zip(outputs["boxes"], outputs["labels"], outputs["scores"]):
        if score >= threshold:
            class_name = COCO_CLASSES.get(label.item(), f"Class {label.item()}")
            x1, y1, x2, y2 = map(float, box.tolist())
            data_tanaman = get_tanaman_by_label(class_name)

            results.append({
                "label": class_name,
                "score": round(float(score), 4),
                "box": [x1, y1, x2, y2],
                "data_tanaman": data_tanaman or "Data not found"
            })

    return JSONResponse(content={"message": "success", "detections": results})
