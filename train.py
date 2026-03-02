import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from dataset import get_dataloaders
from models.vit_model import get_vit_model
from models.swin_model import get_swin_model

def train_model(model, dataloaders, criterion, optimizer, scheduler, device, num_epochs=10, save_path="best_model.pth"):
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            # It's possible we only have an "all" set if there's no train/val split.
            if phase not in dataloaders:
                continue

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), save_path)
                print(f"Saved new best model with accuracy: {best_acc:.4f}")

    print(f'Best val Acc: {best_acc:4f}')
    model.load_state_dict(torch.load(save_path))
    return model

def main():
    parser = argparse.ArgumentParser(description="Train Plant Disease Detection Models")
    parser.add_argument('--data_dir', type=str, required=True, help="Path to dataset directory")
    parser.add_argument('--model', type=str, choices=['vit', 'swin'], default='vit', help="Model to train (vit or swin)")
    parser.add_argument('--epochs', type=int, default=10, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataloaders, class_names = get_dataloaders(args.data_dir, batch_size=args.batch_size)
    num_classes = len(class_names)
    print(f"Found {num_classes} classes: {class_names}")

    # Initialize model
    if args.model == 'vit':
        model = get_vit_model(num_classes)
        save_path = "vit_plant_disease.pth"
    else:
        model = get_swin_model(num_classes)
        save_path = "swin_plant_disease.pth"
        
    model = model.to(device)

    # Loss, optimizer and scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=7, gamma=0.1)

    print(f"Starting training for {args.model}...")
    model = train_model(model, dataloaders, criterion, optimizer, scheduler, device, num_epochs=args.epochs, save_path=save_path)
    
    # Save classes JSON for later inference
    import json
    with open('class_names.json', 'w') as f:
        json.dump(class_names, f)
    print("Class names saved to class_names.json")

if __name__ == '__main__':
    main()
