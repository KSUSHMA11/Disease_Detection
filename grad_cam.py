import argparse
import torch
import cv2
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from PIL import Image
from torchvision import transforms
from models.swin_model import get_swin_model
import json

def get_class_names(class_file="class_names.json"):
    try:
        with open(class_file, "r") as f:
            classes = json.load(f)
        return classes
    except FileNotFoundError:
        return []

def main():
    parser = argparse.ArgumentParser(description="Generate Grad-CAM for Swin Transformer")
    parser.add_argument('--image_path', type=str, required=True, help="Path to input image")
    parser.add_argument('--model_path', type=str, required=True, help="Path to trained Swin model weights")
    parser.add_argument('--output_path', type=str, default='gradcam_output.png', help="Output path for visualizing GradCAM")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    classes = get_class_names()
    num_classes = len(classes) if classes else 10 # Default fallback
    
    model = get_swin_model(num_classes)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = model.to(device)
    model.eval()

    # Preprocess image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # Normalize shouldn't be here if we want to visualize cleanly directly, but the model needs it
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    original_img = Image.open(args.image_path).convert('RGB')
    input_tensor = transform(original_img).unsqueeze(0).to(device)

    # For Swin, the last layernorm or attention blocks can be targeted. 
    # For torchvision's swin_v2_t, target the last norm layer before the head.
    # e.g. model.swin.norm
    target_layers = [model.swin.norm]

    # Initialize GradCAM
    cam = GradCAM(model=model, target_layers=target_layers)

    # We want to visualize the predicted class
    targets = None # None targets the highest scoring class

    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]
    
    # Needs to be 0-1 for show_cam_on_image
    rgb_img = cv2.imread(args.image_path, 1)[:, :, ::-1]
    rgb_img = cv2.resize(rgb_img, (224, 224))
    rgb_img = np.float32(rgb_img) / 255
    
    cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    cv2.imwrite(args.output_path, cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR))
    print(f"Grad-CAM saved to {args.output_path}")

if __name__ == '__main__':
    main()
