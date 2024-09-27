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
from torch.optim.swa_utils import AveragedModel, SWALR
import matplotlib.pyplot as plt

# GPU accelerated image preprocessing using PyTorch
def preprocess_image_torch(image, target_size=(120, 120), device='cpu'):
    if isinstance(image, Image.Image):
        image = np.array(image)
    image = cv2.resize(image, target_size)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    intensity = hsv_image[:, :, 2]  # Extract V channel for intensity
    intensity = intensity.astype(np.uint8)
    intensity = cv2.equalizeHist(intensity)
    hsv_image[:, :, 2] = intensity  # Update the V channel
    image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
    image = cv2.GaussianBlur(image, (5, 5), 0)
    image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0  # Convert to tensor
    image = image.to(device)
    return image

class ResNeXtModel(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNeXtModel, self).__init__()
        self.model = timm.create_model('resnext101_64x4d', pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        self.name = 'resnext101_64x4d'  # Add the name attribute

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

class ImageDataLoader:
    def __init__(self, train_data_path, test_data_path, device='cpu'):
        print("이미지 로딩 내부")
        start_time = time.time()

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        self.mean = mean
        self.std = std
        self.device = device

        self.train_transforms = transforms.Compose([
            transforms.Resize((120, 120)),
            transforms.ToTensor(),  # First convert to tensor
            transforms.RandomRotation(30),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.2),
            transforms.RandomAffine(degrees=30, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=10),
            transforms.GaussianBlur(kernel_size=3),
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3)),
            transforms.Normalize(mean, std)
        ])

        self.val_test_transforms = transforms.Compose([
            transforms.Resize((120, 120)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        # 전체 데이터셋 로드 (초기에는 변환 없이)
        self.full_dataset = datasets.ImageFolder(root=train_data_path)
        
        # 학습 데이터셋과 검증 데이터셋으로 분리
        self.train_size = int(0.8 * len(self.full_dataset))
        self.val_size = len(self.full_dataset) - self.train_size
        self.train_dataset, self.val_dataset = random_split(self.full_dataset, [self.train_size, self.val_size])

        # 각 데이터셋에 대해 변환을 적용
        self.train_dataset.dataset.transform = self.train_transforms
        self.val_dataset.dataset.transform = self.val_test_transforms

        # 테스트 데이터셋 로드
        self.test_dataset = datasets.ImageFolder(root=test_data_path, transform=self.val_test_transforms)

        end_time = time.time()
        print(f"Image data loading and transformation took {end_time - start_time:.2f} seconds")


class ModelTrainer:
    def __init__(self, model, device, train_loader, val_loader, test_loader, mean, std, epochs=30, learning_rate=0.00001, label_smoothing=0.1):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.criterion = LabelSmoothingLoss(num_classes=10, smoothing=label_smoothing)
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs)
        self.epochs = epochs
        self.mean = mean
        self.std = std
        self.train_losses = []
        self.val_losses = []
        self.num_classes = 10
        self.best_val_loss = float('inf')  # 최적의 검증 손실을 저장
        # Initialize SWA
        self.swa_model = AveragedModel(model)
        self.swa_scheduler = SWALR(self.optimizer, swa_lr=learning_rate)

    def mixup_data(self, x, y, alpha=0.5):
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
    
        batch_size = x.size()[0]
        index = torch.randperm(batch_size).to(self.device)
    
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam

    def cutmix_data(self, x, y, alpha=1.0):
        lam = np.random.beta(alpha, alpha)
        rand_index = torch.randperm(x.size()[0]).to(self.device)
        target_a = y
        target_b = y[rand_index]
        bbx1, bby1, bbx2, bby2 = self.rand_bbox(x.size(), lam)
        x[:, :, bbx1:bbx2, bby1:bby2] = x[rand_index, :, bbx1:bbx2, bby1:bby2]
        lam = 1 - ((bbx2 - bbx1) * (bby1 - bby2) / (x.size()[-1] * x.size()[-2]))
        return x, target_a, target_b, lam

    def rand_bbox(self, size, lam):
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)
    
        cx = np.random.randint(W)
        cy = np.random.randint(H)
    
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
    
        return bbx1, bby1, bbx2, bby2

    def train_and_evaluate(self):
        for epoch in range(self.epochs):
            self._train(self.model, self.optimizer, epoch)
            val_loss, val_accuracy = self._validate(self.model, epoch)
            self.swa_model.update_parameters(self.model)
            
            if val_loss < self.best_val_loss:  # 최적의 검증 손실을 비교
                self.best_val_loss = val_loss
                self._save_model(self.model, self.model.name, epoch)  # 모델 이름에 따라 저장

        self._swa_swap()
        self._final_evaluate()
        self._plot_loss()  # 그래프를 그리는 메서드 호출

    def _train(self, model, optimizer, epoch):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        with tqdm(total=len(self.train_loader), desc=f"Epoch {epoch + 1}/{self.epochs} - {model.name}", unit="batch", leave=False) as pbar:
            for batch_idx, (inputs, targets) in enumerate(self.train_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                # Mixup
                inputs, targets_a, targets_b, lam = self.mixup_data(inputs, targets)

                optimizer.zero_grad()

                outputs = model(inputs)
                loss = lam * self.criterion(outputs, targets_a) + (1 - lam) * self.criterion(outputs, targets_b)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct += (lam * predicted.eq(targets_a.data).cpu().sum().float() + (1 - lam) * predicted.eq(targets_b.data).cpu().sum().float()).item()
                total += targets.size(0)

                pbar.update(1)
                pbar.set_postfix({'loss': f'{total_loss/(batch_idx+1):.6f}', 'accuracy': f'{100.*correct/total:.2f}%'})

        self.train_losses.append(total_loss / len(self.train_loader))
        print(f'Train Epoch {epoch + 1}/{self.epochs} - {model.name} Loss: {total_loss/len(self.train_loader):.6f} Accuracy: {100.*correct/total:.2f}%')

    def _validate(self, model, epoch):
        model.eval()
    
        total_loss = 0.0
        correct = 0
        total = 0
        all_labels = []
        all_predictions = []
    
        with torch.no_grad():
            with tqdm(total=len(self.val_loader), desc=f"Validating Epoch {epoch + 1}/{self.epochs} - {model.name}", unit="batch", leave=False) as pbar:
                for batch_idx, (inputs, targets) in enumerate(self.val_loader):
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
    
                    # Mixup
                    inputs, targets_a, targets_b, lam = self.mixup_data(inputs, targets)
    
                    outputs = model(inputs)
    
                    loss = lam * self.criterion(outputs, targets_a) + (1 - lam) * self.criterion(outputs, targets_b)
    
                    total_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    correct += (lam * predicted.eq(targets_a.data).cpu().sum().float() + (1 - lam) * predicted.eq(targets_b.data).cpu().sum().float()).item()
                    total += targets.size(0)
                    all_labels.extend(targets.cpu().numpy())
                    all_predictions.extend(predicted.cpu().numpy())
    
                    pbar.update(1)
                    pbar.set_postfix({'loss': f'{total_loss/(batch_idx+1):.6f}', 'accuracy': f'{100.*correct/total:.2f}%'})
    
        val_loss = total_loss / len(self.val_loader)
        val_accuracy = 100. * correct / total
        self.val_losses.append(val_loss)
        print(f'Validation Epoch {epoch + 1}/{self.epochs} - {model.name} Loss: {val_loss:.6f} Accuracy: {val_accuracy:.2f}%')
        cm = confusion_matrix(all_labels, all_predictions)
        print(f'Confusion Matrix:\n{cm}')
    
        return val_loss, val_accuracy

    def _swa_swap(self):
        self.swa_model.module.load_state_dict(self.model.state_dict())

    def _save_model(self, model, model_name, epoch):
        torch.save(model.state_dict(), f"best_model_{model_name}_epoch_{epoch}.pth")

    def _final_evaluate(self):
        self.model.eval()

        total_loss = 0.0
        correct = 0
        total = 0
        all_labels = []
        all_predictions = []

        with torch.no_grad():
            with tqdm(total=len(self.test_loader), desc=f"Testing", unit="batch", leave=False) as pbar:
                for batch_idx, (inputs, targets) in enumerate(self.test_loader):
                    inputs, targets = inputs.to(self.device), targets.to(self.device)

                    outputs = self.model(inputs)

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
        print(f'Test Loss: {test_loss:.6f} Accuracy: {test_accuracy:.2f}%')
        cm = confusion_matrix(all_labels, all_predictions)
        print(f'Confusion Matrix:\n{cm}')

        # Save classification results to CSV
        label_dict = {i: label for i, label in enumerate(['Bicycle', 'Bridge', 'Bus', 'Car', 'Chimney', 'Crosswalk', 'Hydrant', 'Motorcycle', 'Palm', 'Traffic Light'])}
        with open('c2_t1_a1.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            for i, predict_label in enumerate(all_predictions):
                writer.writerow([f'query{i+1}.png', label_dict[predict_label]])
                
    def _plot_loss(self):
        epochs = range(1, len(self.train_losses) + 1)
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, self.train_losses, 'b', label='Training loss')
        plt.plot(epochs, self.val_losses, 'r', label='Validation loss')
        plt.title('Training and validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

# Example usage
if __name__ == '__main__':
    train_data_path = "C:/Users/L/Desktop/CV/Large"
    test_data_path = "C:/Users/L/Desktop/test_data"
    
    batch_size = 64
    learning_rate = 1e-4
    epochs = 30

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    image_loader = ImageDataLoader(train_data_path, test_data_path, device)

    train_loader = DataLoader(image_loader.train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(image_loader.val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(image_loader.test_dataset, batch_size=batch_size, shuffle=False)

    model = ResNeXtModel(num_classes=10)

    trainer = ModelTrainer(model, device, train_loader, val_loader, test_loader, image_loader.mean, image_loader.std, epochs=epochs, learning_rate=learning_rate)

    trainer.train_and_evaluate()
