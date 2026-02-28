# Plant Disease Detection Using Vision Transformers (ViT) and Swin Transformers

This project implements a complete deep learning pipeline to detect plant diseases from leaf images using state-of-the-art vision models: **Vision Transformer (ViT)** and **Swin Transformer**.

## Features
- **Model Training**: PyTorch scripts to train ViT and Swin Transformer on any standard image classification dataset (e.g., PlantVillage).
- **Backend API**: FastAPI backend to serve model predictions.
- **Frontend App**: HTML/CSS/JS web application for easy file uploads and reading predictions.
- **Agricultural Info**: Provides detailed Cause, Cure, and Prevention strategies for detected diseases.
- **Evaluation**: Script to generate confusion matrices and evaluate metrics (precision, recall, f1-score).

## Folder Structure
```
Plant-Disease-Detection/
│
├── backend/
│   └── main.py              # FastAPI server
├── frontend/
│   ├── index.html           # Web UI
│   ├── style.css            # UI Styling
│   └── script.js            # UI Logic
├── models/
│   ├── vit_model.py         # ViT architecture definition
│   └── swin_model.py        # Swin Transformer definition
├── utils/
│   ├── disease_info.json    # Dictionary mapping disease to cause and cure
│   └── inference.py         # Core logic for making predictions
│
├── dataset.py               # Data loading and augmentation
├── train.py                 # Training script
├── evaluate.py              # Evaluation and confusion matrix script
├── requirements.txt         # Project dependencies
└── README.md                # This file
```

## How to Run Locally

### 1. Installation
Ensure you have Python 3.8+ installed. Install the required dependencies:
```bash
pip install -r requirements.txt
```

### 2. Dataset Setup
Prepare your dataset (e.g., PlantVillage) with the following structure:
```
dataset_folder/
  train/
    Class1/
    Class2/
  val/
    Class1/
    Class2/
```

### 3. Training the Models
To train the Vision Transformer:
```bash
python train.py --data_dir /path/to/dataset --model vit --epochs 10 --batch_size 32
```
To train the Swin Transformer:
```bash
python train.py --data_dir /path/to/dataset --model swin --epochs 10 --batch_size 32
```
*Note: This will generate `vit_plant_disease.pth`, `swin_plant_disease.pth`, and `class_names.json`.*

### 4. Running the Backend API
Start the FastAPI server:
```bash
uvicorn backend.main:app --reload
```
The API will be available at `http://localhost:8000`.

### 5. Running the Frontend
Simply open the `frontend/index.html` file in your preferred web browser, or use a simple HTTP server:
```bash
cd frontend
python -m http.server 3000
```
Then visit `http://localhost:3000` in your browser.

## Deployment Options

### Render
1. Create a `render.yaml` or connect your GitHub repo to Render.
2. Select "Web Service" and choose Python.
3. Start command: `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`
4. Set up an exact copy for your frontend using Render's Static Site deployment.

### Heroku
1. Add a `Procfile` containing: `web: uvicorn backend.main:app --host=0.0.0.0 --port=${PORT:-5000}`
2. Deploy using Heroku CLI:
   ```bash
   heroku create
   git push heroku main
   ```

## Methodology & IEEE Scope
- Multi-head self-attention mechanisms in ViT offer superior global context understanding compared to traditional CNNs.
- Swin Transformers compute hierarchical context utilizing shifted window mechanisms, bridging the gap between translation invariance and global awareness.
- Pre-trained weights enhance transfer learning robustness. Performance can be compared via confusion matrices using `evaluate.py`.
