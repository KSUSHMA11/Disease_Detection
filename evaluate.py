import argparse
import torch
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from dataset import get_dataloaders
from models.vit_model import get_vit_model
from models.swin_model import get_swin_model

def evaluate_model(model_path, data_dir, model_type='vit', batch_size=32):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Get dataloaders
    dataloaders, class_names = get_dataloaders(data_dir, batch_size=batch_size)
    
    # We evaluate on tests if available, otherwise val
    phase = 'test' if 'test' in dataloaders else ('val' if 'val' in dataloaders else 'all')
    loader = dataloaders[phase]
    
    num_classes = len(class_names)
    
    if model_type == 'vit':
        model = get_vit_model(num_classes)
    else:
        model = get_swin_model(num_classes)
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    all_preds = []
    all_labels = []
    
    print(f"Evaluating {model_type} model on {phase} set...")
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    # Classification Report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))
    
    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - {model_type.upper()}')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'{model_type}_confusion_matrix.png')
    print(f"Confusion matrix saved as {model_type}_confusion_matrix.png")

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Evaluate Plant Disease Model")
    parser.add_argument('--model_path', required=True, help="Path to saved model weights")
    parser.add_argument('--data_dir', required=True, help="Path to dataset")
    parser.add_argument('--model', choices=['vit', 'swin'], default='vit')
    
    args = parser.parse_args()
    evaluate_model(args.model_path, args.data_dir, args.model)
