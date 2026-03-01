from __future__ import annotations

import io
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torchvision import transforms
from torchvision.models import swin_v2_t, vit_b_16

REPO_ROOT = Path(__file__).resolve().parents[1]
CLASS_NAMES_PATH = REPO_ROOT / "class_names.json"
DISEASE_INFO_PATH = REPO_ROOT / "utils" / "disease_info.json"


@lru_cache(maxsize=1)
def load_disease_info() -> dict[str, dict[str, str]]:
    if DISEASE_INFO_PATH.exists():
        with DISEASE_INFO_PATH.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {}


@lru_cache(maxsize=1)
def load_class_names() -> list[str]:
    if CLASS_NAMES_PATH.exists():
        with CLASS_NAMES_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list) and data:
            return [str(x) for x in data]
    return ["Unknown___Unknown"]


@lru_cache(maxsize=2)
def load_model(model_type: str, model_path: str, num_classes: int) -> torch.nn.Module:
    path = REPO_ROOT / model_path
    if not path.exists():
        raise FileNotFoundError(
            f"Model weights not found: {path.name}. Train or copy weights into project root."
        )

    if model_type == "swin":
        model = swin_v2_t(weights=None)
        model.head = torch.nn.Linear(model.head.in_features, num_classes)
    else:
        model = vit_b_16(weights=None)
        model.heads.head = torch.nn.Linear(model.heads.head.in_features, num_classes)

    state_dict = torch.load(path, map_location="cpu")
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def _split_label(raw_label: str) -> tuple[str, str]:
    if "___" in raw_label:
        plant, disease = raw_label.split("___", 1)
    elif "_" in raw_label:
        parts = raw_label.split("_", 1)
        plant, disease = parts[0], parts[1]
    else:
        plant, disease = "Unknown", raw_label
    return plant.replace("_", " "), disease.replace("_", " ")


def predict_image(image_bytes: bytes, model_type: str = "vit", model_path: str = "vit_plant_disease.pth") -> dict[str, Any]:
    class_names = load_class_names()
    model = load_model(model_type=model_type, model_path=model_path, num_classes=len(class_names))

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    preprocess = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    tensor = preprocess(image).unsqueeze(0)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)

    conf, pred_idx = torch.max(probs, dim=1)
    label = class_names[pred_idx.item()]
    plant_name, disease = _split_label(label)

    info = load_disease_info()
    disease_key = "healthy" if "healthy" in disease.lower() else "default"
    details = info.get(disease_key, info.get("default", {}))

    return {
        "plant_name": plant_name,
        "disease": disease,
        "confidence": round(conf.item() * 100, 2),
        "cause": details.get("cause", "No data available."),
        "cure": details.get("cure", "No data available."),
        "prevention": details.get("prevention", "No data available."),
    }
