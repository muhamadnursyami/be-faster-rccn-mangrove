# Import libary
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
from fastapi.staticfiles import StaticFiles
from torchvision.transforms import functional as F
from torchvision.transforms import Resize
# =====================================
# 🚀 INIT FASTAPI APP
# =====================================
# memulai fast api
app = FastAPI()
# meng mount folder gambar mangrove kedalam url/ gambar_mangrove 
app.mount("/gambar_mangrove", StaticFiles(directory="gambar_mangrove"), name="gambar_mangrove")
# middlware untuk menangani cors
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
    # memuat model
    weights = FasterRCNN_ResNet50_FPN_Weights.COCO_V1
    model = fasterrcnn_resnet50_fpn(weights=weights)
    # mengantikan nilai roi head box predictor sesuatu dengan kelas yang diinginkan
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
# membaca file datTanaman json
with open("dataTanaman.json") as f:
    tanaman_data = json.load(f)
# mnencari  berdasarkan label nama yang dihapus tanda  spasi dan -
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

@app.get("/mangrove/get-data")
def get_data():
    return JSONResponse(content=tanaman_data)

@app.post("/mangrove/detect")
async def detect_image(file: UploadFile = File(...), threshold: float = 0.6):
    #membaca file gambar yang dikirim kan dan diconvert ke rgn
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Resize ke 640x640 agar cocok dengan training model
    image_resized = image.resize((640, 640))

    # Konversi ke tensor
    image_tensor = F.to_tensor(image_resized).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)[0]

    results = []
    # proses data jika score besar dari threshold maka
    for box, label, score in zip(outputs["boxes"], outputs["labels"], outputs["scores"]):
        if score >= threshold:
            # mengambil data label item
            class_name = COCO_CLASSES.get(label.item(), f"Class {label.item()}")
            x1, y1, x2, y2 = map(float, box.tolist())

            # Skala bounding box dari 640x640 ke ukuran gambar asli
            scale_x = image.width / 640
            scale_y = image.height / 640
            x1 *= scale_x
            x2 *= scale_x
            y1 *= scale_y
            y2 *= scale_y

            data_tanaman = get_tanaman_by_label(class_name)
            # hasil respon json  prediksi 
            results.append({
                "label": class_name,
                "score": round(float(score), 4),
                "box": [x1, y1, x2, y2],
                "data_tanaman": data_tanaman or "Data not found"
            })

    return JSONResponse(content={"message": "success", "detections": results})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=5001, reload=False)

