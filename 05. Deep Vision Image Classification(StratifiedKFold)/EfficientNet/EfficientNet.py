import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
import csv
import timm
from tqdm import tqdm
import time
from torch.optim.swa_utils import AveragedModel, SWALR
import matplotlib.pyplot as plt

class DenseNetModel(nn.Module):
    def __init__(self, num_classes=10):
        super(DenseNetModel, self).__init__()
        self.model = timm.create_model('densenet201', pretrained=True)
        self.model.classifier = nn.Linear(self.model.classifier.in_features, num_classes)
        self.name = 'densenet201'  # Add the name attribute

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

        # 학습 데이터 증강 및 전처리
        self.train_transforms = transforms.Compose([
            transforms.Resize((120, 120)),  # 이미지 크기 조정
            transforms.RandomRotation(30),  # 최대 30도 회전
            transforms.RandomHorizontalFlip(),  # 수평 뒤집기
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.2),  # 밝기, 대비, 채도 조절
            transforms.RandomAffine(degrees=30, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=10),  # 여러 가지 변환
            transforms.GaussianBlur(kernel_size=3),  # 가우시안 블러
            transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),  # 샤프닝 추가
            transforms.RandomApply([transforms.Equalize()], p=0.5),  # 밝기 평활화 추가
            transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),  # 샤프닝 추가
            transforms.RandomApply([transforms.ColorJitter(brightness=0.2)], p=0.5),  # 추가 밝기 조절
            transforms.RandomApply([transforms.ColorJitter(contrast=0.2)], p=0.5),  # 추가 대비 조절
            transforms.RandomApply([transforms.ColorJitter(saturation=0.2)], p=0.5),  # 추가 채도 조절
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3)),  # 이미지 일부 삭제
            transforms.ToTensor(),  # 텐서로 변환
            transforms.Normalize(mean, std)  # 정규화
        ])

        self.val_test_transforms = transforms.Compose([
            transforms.Resize((120, 120)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        # 전체 데이터셋 로드 (초기에는 변환 없이)
        self.full_dataset = datasets.ImageFolder(root=train_data_path)
        
        # 테스트 데이터셋 로드
        self.test_dataset = datasets.ImageFolder(root=test_data_path, transform=self.val_test_transforms)

        end_time = time.time()
        print(f"Image data loading and transformation took {end_time - start_time:.2f} seconds")


class ModelTrainer:
    def __init__(self, model, device, train_loader, val_loader, test_loader, mean, std, epochs=25, learning_rate=0.00001, label_smoothing=0.1):
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
        self.best_val_accuracy = 0.0  # 최적의 검증 정확도를 저장
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
        skf = StratifiedKFold(n_splits=5)
        targets = np.array([label for _, label in self.train_loader.dataset])

        for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(targets)), targets)):
            print(f"Fold {fold + 1}")

            train_subset = Subset(self.train_loader.dataset, train_idx)
            val_subset = Subset(self.train_loader.dataset, val_idx)

            train_loader = DataLoader(train_subset, batch_size=self.train_loader.batch_size, shuffle=True)
            val_loader = DataLoader(val_subset, batch_size=self.val_loader.batch_size, shuffle=False)

            for epoch in range(self.epochs):
                self._train(self.model, self.optimizer, train_loader, epoch)
                val_loss, val_accuracy = self._validate(self.model, val_loader, epoch)
                self.swa_model.update_parameters(self.model)
                self.scheduler.step()

                if val_accuracy > self.best_val_accuracy:  # 최적의 검증 정확도를 비교
                    self.best_val_accuracy = val_accuracy
                    self._save_model(self.model, self.model.name, fold, epoch)  # 모델 이름에 따라 저장

        self._swa_swap()
        self._final_evaluate()
        self._plot_loss()  # 그래프를 그리는 메서드 호출

    def _train(self, model, optimizer, train_loader, epoch):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch + 1}/{self.epochs} - {model.name}", unit="batch", leave=False) as pbar:
            for batch_idx, (inputs, targets) in enumerate(train_loader):
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

        self.train_losses.append(total_loss / len(train_loader))
        print(f'Train Epoch {epoch + 1}/{self.epochs} - {model.name} Loss: {total_loss/len(train_loader):.6f} Accuracy: {100.*correct/total:.2f}%')

    def _validate(self, model, val_loader, epoch):
        model.eval()
    
        total_loss = 0.0
        correct = 0
        total = 0
        all_labels = []
        all_predictions = []
    
        with torch.no_grad():
            with tqdm(total=len(val_loader), desc=f"Validating Epoch {epoch + 1}/{self.epochs} - {model.name}", unit="batch", leave=False) as pbar:
                for batch_idx, (inputs, targets) in enumerate(val_loader):
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
    
        val_loss = total_loss / len(val_loader)
        val_accuracy = 100. * correct / total
        self.val_losses.append(val_loss)
        print(f'Validation Epoch {epoch + 1}/{self.epochs} - {model.name} Loss: {val_loss:.6f} Accuracy: {val_accuracy:.2f}%')
        cm = confusion_matrix(all_labels, all_predictions)
        print(f'Confusion Matrix:\n{cm}')
    
        return val_loss, val_accuracy

    def _swa_swap(self):
        self.swa_model.module.load_state_dict(self.model.state_dict())

    def _save_model(self, model, model_name, fold, epoch):
        torch.save(model.state_dict(), f"best_model_{model_name}_fold_{fold}_epoch_{epoch}.pth")

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
    epochs = 25

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    image_loader = ImageDataLoader(train_data_path, test_data_path, device)

    train_loader = DataLoader(image_loader.full_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(image_loader.full_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(image_loader.test_dataset, batch_size=batch_size, shuffle=False)

    model = DenseNetModel(num_classes=10)

    trainer = ModelTrainer(model, device, train_loader, val_loader, test_loader, image_loader.mean, image_loader.std, epochs=epochs, learning_rate=learning_rate)

    trainer.train_and_evaluate()
