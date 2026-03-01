from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from utils.inference import predict_image

app = FastAPI(
    title="Plant Disease Detection API",
    description="API for detecting plant diseases using ViT and Swin Transformers.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root() -> dict[str, str]:
    return {"message": "Welcome to Plant Disease Detection API. Use /predict to get predictions."}


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
async def predict_plant_disease(file: UploadFile = File(...), model_type: str = "vit") -> dict:
    image_bytes = await file.read()
    model_path = "vit_plant_disease.pth" if model_type == "vit" else "swin_plant_disease.pth"

    try:
        return predict_image(image_bytes=image_bytes, model_type=model_type, model_path=model_path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc


# Run with: uvicorn backend.main:app --reload
