import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix
import csv
import timm
from tqdm import tqdm
import cv2
from PIL import Image
import glob

class ResNeXtModel(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNeXtModel, self).__init__()
        self.model = timm.create_model('resnext101_64x4d', pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        self.name = 'resnext101_64x4d'  # Add the name attribute

    def forward(self, x):
        return self.model(x)

class ImageDataLoader:
    def __init__(self, test_data_path):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        self.val_test_transforms = transforms.Compose([
            transforms.Resize((120, 120)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        self.test_dataset = datasets.ImageFolder(root=test_data_path, transform=self.val_test_transforms)

class ModelTester:
    def __init__(self, model_class, device, test_loader, model_pattern):
        self.device = torch.device(device)
        self.test_loader = test_loader
        self.model_class = model_class
        self.model_pattern = model_pattern
        self.criterion = nn.CrossEntropyLoss()
        self.model_paths = glob.glob(self.model_pattern)

    def test_models(self):
        for model_path in self.model_paths:
            model = self.model_class(num_classes=10)
            model.load_state_dict(torch.load(model_path))
            model.to(self.device)
            test_loss, test_accuracy, all_predictions = self._test(model)
            print(f"Model: {model_path} - Test Loss: {test_loss:.6f} - Test Accuracy: {test_accuracy:.2f}%")
            self._save_results_to_csv(all_predictions, model_path)

    def _test(self, model):
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        all_labels = []
        all_predictions = []

        with torch.no_grad():
            with tqdm(total=len(self.test_loader), desc=f"Testing", unit="batch", leave=False) as pbar:
                for batch_idx, (inputs, targets) in enumerate(self.test_loader):
                    inputs, targets = inputs.to(self.device), targets.to(self.device)

                    outputs = model(inputs)

                    loss = self.criterion(outputs, targets)

                    total_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    correct += predicted.eq(targets.data).cpu().sum().item()
                    total += targets.size(0)
                    all_labels.extend(targets.cpu().numpy())
                    all_predictions.extend(predicted.cpu().numpy())

                    pbar.update(1)
                    pbar.set_postfix({'loss': f'{total_loss/(batch_idx+1):.6f}', 'accuracy': f'{100.*correct/total:.2f}%'})

        test_loss = total_loss / len(self.test_loader)
        test_accuracy = 100. * correct / total
        cm = confusion_matrix(all_labels, all_predictions)
        print(f'Confusion Matrix:\n{cm}')

        return test_loss, test_accuracy, all_predictions

    def _save_results_to_csv(self, all_predictions, model_path):
        model_name = model_path.split('/')[-1].split('.')[0]
        label_dict = {i: label for i, label in enumerate(['Bicycle', 'Bridge', 'Bus', 'Car', 'Chimney', 'Crosswalk', 'Hydrant', 'Motorcycle', 'Palm', 'Traffic Light'])}
        with open(f'{model_name}_test_results.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            for i, predict_label in enumerate(all_predictions):
                writer.writerow([f'query{i+1}.png', label_dict[predict_label]])

# Example usage
if __name__ == '__main__':
    test_data_path = "C:/Users/L/Desktop/test_data"
    model_pattern = "best_model_resnext101_64x4d_epoch_*.pth"
    batch_size = 64

    image_loader = ImageDataLoader(test_data_path)
    test_loader = DataLoader(image_loader.test_dataset, batch_size=batch_size, shuffle=False)

    model_class = ResNeXtModel
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tester = ModelTester(model_class, device, test_loader, model_pattern)
    tester.test_models()
