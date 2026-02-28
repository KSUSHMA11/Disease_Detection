from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from utils.inference import predict_image
import io

app = FastAPI(title="Plant Disease Detection API", description="API for detecting plant diseases using ViT and Swin Transformers.")

# Add CORS to allow requests from frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Welcome to Plant Disease Detection API. Use /predict to get predictions."}

@app.post("/predict")
async def predict_plant_disease(file: UploadFile = File(...), model_type: str = "vit"):
    """
    Accepts an image and returns plant name, disease, and cure recommendations.
    """
    image_bytes = await file.read()
    
    # Adjust paths if you've trained the models locally
    model_path = "vit_plant_disease.pth" if model_type == 'vit' else "swin_plant_disease.pth"
    
    try:
        prediction = predict_image(image_bytes, model_type=model_type, model_path=model_path)
        return prediction
    except Exception as e:
        return {"error": str(e)}

# To run: uvicorn backend.main:app --reload
