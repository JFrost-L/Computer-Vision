import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix
import csv
import timm
from tqdm import tqdm
import cv2
from PIL import Image
import time

# Define the models
class ResNeXtModel(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNeXtModel, self).__init__()
        self.model = timm.create_model('resnext50_32x4d', pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        self.name = 'resnext50_32x4d'

    def forward(self, x):
        return self.model(x)

class EfficientNetModel(nn.Module):
    def __init__(self, num_classes=10):
        super(EfficientNetModel, self).__init__()
        self.model = timm.create_model('efficientnet_b5', pretrained=True)
        self.model.classifier = nn.Linear(self.model.classifier.in_features, num_classes)
        self.name = 'EfficientNetB5'

    def forward(self, x):
        return self.model(x)

class RegNetModel(nn.Module):
    def __init__(self, num_classes=10):
        super(RegNetModel, self).__init__()
        self.model = timm.create_model('regnety_320', pretrained=True)
        self.model.head.fc = nn.Linear(self.model.head.fc.in_features, num_classes)
        self.name = 'regnety_320'

    def forward(self, x):
        return self.model(x)

class LabelSmoothingLoss(nn.Module):
    def __init__(self, num_classes, smoothing=0.1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = num_classes
        self.dim = -1

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            target = target.unsqueeze(1)
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target, self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))

# Image Data Loader
class ImageDataLoader:
    def __init__(self, train_data_path, test_data_path):
        print("이미지 로딩 내부")
        start_time = time.time()

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        self.mean = mean
        self.std = std

        self.train_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(), 
            transforms.RandomRotation(30),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.2),
            transforms.RandomAffine(degrees=30, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=10),
            transforms.GaussianBlur(kernel_size=3),
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3)),
            transforms.Normalize(mean, std)
        ])

        self.val_test_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        self.full_dataset = datasets.ImageFolder(root=train_data_path, transform=self.train_transforms)
        self.test_dataset = datasets.ImageFolder(root=test_data_path, transform=self.val_test_transforms)

        self.train_size = int(0.8 * len(self.full_dataset))
        self.val_size = len(self.full_dataset) - self.train_size

        self.train_dataset, self.val_dataset = random_split(self.full_dataset, [self.train_size, self.val_size])

        end_time = time.time()
        print(f"Image data loading and transformation took {end_time - start_time:.2f} seconds")

# Ensemble Model Trainer
class EnsembleModelTrainer:
    def __init__(self, models, device, train_loader, val_loader, test_loader, mean, std, epochs=30, learning_rate=0.00001, label_smoothing=0.1):
        self.device = torch.device(device)
        self.models = [model.to(self.device) for model in models]
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.criterion = LabelSmoothingLoss(num_classes=10, smoothing=label_smoothing)
        self.optimizers = [optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4) for model in models]
        self.schedulers = [optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs) for optimizer in self.optimizers]
        self.epochs = epochs
        self.mean = mean
        self.std = std
        self.num_classes = 10

    def train_and_evaluate(self):
        for epoch in range(self.epochs):
            self._train(epoch)
            self._validate(epoch)

    def _train(self, epoch):
        for model in self.models:
            model.train()

        with tqdm(total=len(self.train_loader), desc=f"Epoch {epoch + 1}/{self.epochs}", unit="batch") as pbar:
            for inputs, targets in self.train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                for optimizer in self.optimizers:
                    optimizer.zero_grad()

                outputs = [model(inputs) for model in self.models]
                loss = sum(self.criterion(output, targets) for output in outputs) / len(outputs)
                loss.backward()

                for optimizer in self.optimizers:
                    optimizer.step()

                pbar.update(1)

    def _validate(self, epoch):
        for model in self.models:
            model.eval()

        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = [model(inputs) for model in self.models]
                avg_output = torch.mean(torch.stack(outputs), dim=0)
                _, predicted = torch.max(avg_output, 1)
                correct += predicted.eq(targets.data).cpu().sum().item()
                total += targets.size(0)

        accuracy = 100. * correct / total
        print(f'Validation Epoch {epoch + 1}/{self.epochs} Accuracy: {accuracy:.2f}%')

# Example usage
if __name__ == '__main__':
    train_data_path = "C:/Users/L/Desktop/CV/Large"
    test_data_path = "C:/Users/L/Desktop/test_data"
    batch_size = 64
    learning_rate = 1e-4
    epochs = 30

    image_loader = ImageDataLoader(train_data_path, test_data_path)
    train_loader = DataLoader(image_loader.train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(image_loader.val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(image_loader.test_dataset, batch_size=batch_size, shuffle=False)

    models = [ResNeXtModel(num_classes=10), EfficientNetModel(num_classes=10), RegNetModel(num_classes=10)]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainer = EnsembleModelTrainer(models, device, train_loader, val_loader, test_loader, image_loader.mean, image_loader.std, epochs=epochs, learning_rate=learning_rate)
    trainer.train_and_evaluate()
