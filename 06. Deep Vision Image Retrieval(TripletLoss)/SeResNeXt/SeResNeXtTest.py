import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from torchvision import datasets, transforms
import timm
from tqdm import tqdm
import random
from PIL import Image
import cv2
import pandas as pd
import csv
from torch.optim.swa_utils import AveragedModel, SWALR
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import glob

# GPU accelerated image preprocessing using PyTorch
def preprocess_image_torch(image, target_size=(120, 120)):
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
    return image


class SeResNeXtModel(nn.Module):
    def __init__(self, num_classes=10):
        super(SeResNeXtModel, self).__init__()
        self.model = timm.create_model('seresnext101d_32x8d', pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        self.name = 'seresnext101d_32x8d'  # Add the name attribute

    def forward(self, x):
        return self.model(x)


class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        positive_distance = (anchor - positive).pow(2).sum(1)
        negative_distance = (anchor - negative).pow(2).sum(1)
        losses = F.relu(positive_distance - negative_distance + self.margin)
        return losses.mean()

class TripletImageDataset(Dataset):
    def __init__(self, image_folder, transform=None):
        self.image_folder = image_folder
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.label_to_indices = {}
        self._prepare_data()

    def _prepare_data(self):
        dataset = datasets.ImageFolder(self.image_folder)
        self.image_paths = [img[0] for img in dataset.imgs]
        self.labels = [dataset.classes[img[1]] for img in dataset.imgs]
        for idx, label in enumerate(self.labels):
            if label not in self.label_to_indices:
                self.label_to_indices[label] = []
            self.label_to_indices[label].append(idx)
        
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        anchor_path = self.image_paths[index]
        anchor_label = self.labels[index]
        positive_index = index
        while positive_index == index:
            positive_index = random.choice(self.label_to_indices[anchor_label])
        negative_label = anchor_label
        while negative_label == anchor_label:
            negative_label = random.choice(self.labels)
        negative_index = random.choice(self.label_to_indices[negative_label])
    
        anchor_image = Image.open(anchor_path).convert('RGB')
        positive_image = Image.open(self.image_paths[positive_index]).convert('RGB')
        negative_image = Image.open(self.image_paths[negative_index]).convert('RGB')
    
        # Preprocess using preprocess_image_torch
        anchor_image = preprocess_image_torch(anchor_image)
        positive_image = preprocess_image_torch(positive_image)
        negative_image = preprocess_image_torch(negative_image)
    
        return anchor_image, positive_image, negative_image, anchor_label

class ModelTrainer:
    def __init__(self, model, device, train_loader, val_loader, test_loader, epochs=30, learning_rate=0.00001):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.criterion = TripletLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs)
        self.epochs = epochs
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.best_retrieval_accuracy = 0.0  # Initialize best retrieval accuracy
        self.early_stopping_patience = 7
        # Initialize SWA
        self.swa_model = AveragedModel(model)
        self.swa_scheduler = SWALR(self.optimizer, swa_lr=learning_rate)

    def calculate_accuracy(self, model, data_loader, device, k=10):
        model.eval()
        correct = 0
        total = 0
        all_embeddings = []
        all_labels = []
    
        with torch.no_grad():
            for (anchor, _, _, label) in data_loader:
                anchor = anchor.to(device)
                anchor_output = model(anchor)
                all_embeddings.append(anchor_output)
                all_labels.extend(label)  # 튜플이 아닌 리스트로 추가
    
        all_embeddings = torch.cat(all_embeddings)
    
        for i in range(len(all_embeddings)):
            query_embedding = all_embeddings[i]
            query_label = all_labels[i]
    
            distances = F.pairwise_distance(query_embedding.unsqueeze(0), all_embeddings)
            closest_indices = distances.argsort()[1:k+1]
            closest_labels = [all_labels[idx] for idx in closest_indices]
    
            if query_label in closest_labels:
                correct += 1
            total += 1
    
        accuracy = correct / total
        return accuracy * 100

    def train_and_evaluate(self):
        early_stopping_counter = 0

        for epoch in range(self.epochs):
            train_loss = self._train(self.model, self.optimizer, epoch)
            val_loss, val_retrieval_accuracy = self._validate(self.model, epoch)
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            self.swa_model.update_parameters(self.model)
            
            # Check if validation loss is decreased
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                early_stopping_counter = 0
                self._save_model(self.model, self.model.name, epoch)
            else:
                early_stopping_counter += 1

            if early_stopping_counter >= self.early_stopping_patience:
                print(f"Early stopping for {self.model.name} due to no improvement in validation loss.")
                break

        self._swa_swap()
        self._final_evaluate()
        self.plot_losses()

    def _train(self, model, optimizer, epoch):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        with tqdm(total=len(self.train_loader), desc=f"Epoch {epoch + 1}/{self.epochs} - {model.name}", unit="batch", leave=False) as pbar:
            for batch_idx, (anchor, positive, negative, label) in enumerate(self.train_loader):
                anchor, positive, negative = anchor.to(self.device), positive.to(self.device), negative.to(self.device)

                optimizer.zero_grad()

                anchor_output = model(anchor)
                positive_output = model(positive)
                negative_output = model(negative)

                loss = self.criterion(anchor_output, positive_output, negative_output)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

                with torch.no_grad():
                    distances = F.pairwise_distance(anchor_output, positive_output)
                    closest_indices = distances.argsort()[:10]
                    correct += sum([1 for idx in closest_indices if label[idx] == label[0]])
                    total += len(label)

                pbar.update(1)
                accuracy = (correct / total) * 100 if total > 0 else 0
                pbar.set_postfix({'loss': f'{total_loss/(batch_idx+1):.6f}', 'accuracy': f'{accuracy:.2f}%'})

        train_loss = total_loss / len(self.train_loader)
        train_accuracy = self.calculate_accuracy(model, self.train_loader, self.device, k=10)
        print(f'Train Epoch {epoch + 1}/{self.epochs} - {model.name} - Loss: {train_loss:.6f} - Accuracy: {train_accuracy:.2f}%')
        return train_loss

    def _validate(self, model, epoch):
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            with tqdm(total=len(self.val_loader), desc=f"Validating Epoch {epoch + 1}/{self.epochs} - {model.name}", unit="batch", leave=False) as pbar:
                for batch_idx, (anchor, positive, negative, label) in enumerate(self.val_loader):
                    anchor, positive, negative = anchor.to(self.device), positive.to(self.device), negative.to(self.device)
    
                    anchor_output = model(anchor)
                    positive_output = model(positive)
                    negative_output = model(negative)
    
                    loss = self.criterion(anchor_output, positive_output, negative_output)
                    total_loss += loss.item()
    
                    distances = F.pairwise_distance(anchor_output, positive_output)
                    closest_indices = distances.argsort()[:10]
                    correct += sum([1 for idx in closest_indices if label[idx] == label[0]])
                    total += len(label)
    
                    pbar.update(1)
                    accuracy = (correct / total) * 100 if total > 0 else 0
                    pbar.set_postfix({'loss': f'{total_loss/(batch_idx+1):.6f}', 'accuracy': f'{accuracy:.2f}%'})
    
        val_loss = total_loss / len(self.val_loader)
        val_accuracy = self.calculate_accuracy(model, self.val_loader, self.device, k=10)
        val_retrieval_accuracy, _ = self._calculate_retrieval_accuracy(model)
        print(f'Validation Epoch {epoch + 1}/{self.epochs} - {model.name} - Loss: {val_loss:.6f} - Accuracy: {val_accuracy:.2f}% - Retrieval Accuracy: {val_retrieval_accuracy:.2f}%')
        return val_loss, val_retrieval_accuracy

    def _final_evaluate(self):
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            with tqdm(total=len(self.test_loader), desc=f"Testing", unit="batch", leave=False) as pbar:
                for batch_idx, (anchor, positive, negative, label) in enumerate(self.test_loader):
                    anchor, positive, negative = anchor.to(self.device), positive.to(self.device), negative.to(self.device)
    
                    anchor_output = self.model(anchor)
                    positive_output = self.model(positive)
                    negative_output = self.model(negative)
    
                    loss = self.criterion(anchor_output, positive_output, negative_output)
                    total_loss += loss.item()
    
                    distances = F.pairwise_distance(anchor_output, positive_output)
                    closest_indices = distances.argsort()[:10]
                    correct += sum([1 for idx in closest_indices if label[idx] == label[0]])
                    total += len(label)
    
                    pbar.update(1)
                    accuracy = (correct / total) * 100 if total > 0 else 0
                    pbar.set_postfix({'loss': f'{total_loss/(batch_idx+1):.6f}', 'accuracy': f'{accuracy:.2f}%'})
    
        test_loss = total_loss / len(self.test_loader)
        test_accuracy = self.calculate_accuracy(self.model, self.test_loader, self.device, k=10)
        test_retrieval_accuracy, retrieval_results = self._calculate_retrieval_accuracy(self.model)
        print(f'Test - Loss: {test_loss:.6f} - Accuracy: {test_accuracy:.2f}% - Retrieval Accuracy: {test_retrieval_accuracy:.2f}%')
        self._save_retrieval_results_to_csv(retrieval_results)


    def plot_losses(self):
       plt.figure(figsize=(10, 5))
       plt.plot(self.train_losses, label='Train Loss')
       plt.plot(self.val_losses, label='Validation Loss')
       plt.xlabel('Epoch')
       plt.ylabel('Loss')
       plt.title('Train and Validation Loss over Epochs')
       plt.legend()
       plt.grid(True)
       plt.show()

    def _save_model(self, model, model_name, epoch):
        torch.save(model.state_dict(), f"best_model_{model_name}_epoch_{epoch}.pth")

    def _swa_swap(self):
        self.swa_model.module.load_state_dict(self.model.state_dict())

    def _calculate_retrieval_accuracy(self, model):
        model.eval()
        
        train_features, train_labels = self._extract_features_labels(model, self.test_loader)
        test_features, test_labels = self._extract_features_labels(model, self.test_loader)
        
        total_correct_labels = 0
        total_labels = 0
        retrieval_results = []
        
        with torch.no_grad():
            for i, (feature, label) in enumerate(zip(test_features, test_labels)):
                neighbors = self._find_neighbors_cosine(train_features, feature, top_n=10)
                neighbor_labels = [train_labels[idx] for idx in neighbors]
                retrieval_results.append((i, neighbor_labels))
                
                correct_labels = neighbor_labels.count(label)
                total_correct_labels += correct_labels
                total_labels += len(neighbor_labels)
        
        retrieval_accuracy = (total_correct_labels / total_labels) * 100 if total_labels > 0 else 0
        return retrieval_accuracy, retrieval_results

    def _extract_features_labels(self, model, loader):
        model.eval()
        features = []
        labels = []
        
        with torch.no_grad():
            for inputs, _, _, label in loader:
                inputs = inputs.to(self.device)
                outputs = model(inputs)
                features.extend(outputs.cpu().numpy())
                labels.extend(label)
        
        return np.array(features), np.array(labels)

    def _find_neighbors_cosine(self, features, query_feature, top_n=10):
        features = torch.tensor(features).to(self.device)
        query_feature = torch.tensor(query_feature).unsqueeze(0).to(self.device)
        similarities = F.cosine_similarity(query_feature, features)
        neighbor_indices = torch.argsort(similarities, descending=True)[:top_n]
        return neighbor_indices.cpu().numpy()

    def _save_retrieval_results_to_csv(self, retrieval_results):
        with open('c2_t2_a1.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            for i, neigh_labels in retrieval_results:
                writer.writerow([f'query{i+1}.png'] + list(neigh_labels))

class ModelTester:
    def __init__(self, model_class, device, test_loader, model_pattern):
        self.device = torch.device(device)
        self.test_loader = test_loader
        self.model_class = model_class
        self.model_pattern = model_pattern
        self.criterion = TripletLoss()
        self.model_paths = glob.glob(self.model_pattern)
    
    def test_models(self):
        for model_path in self.model_paths:
            model = self.model_class()
            model.load_state_dict(torch.load(model_path))
            model.to(self.device)
            test_loss, test_accuracy, test_retrieval_accuracy, retrieval_results = self._test(model)
            print(f"Model: {model_path} - Test Loss: {test_loss:.6f} - Test Accuracy: {test_accuracy:.2f}% - Retrieval Accuracy: {test_retrieval_accuracy:.2f}%")
            self._save_retrieval_results_to_csv(retrieval_results, model_path)
    
    def _test(self, model):
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            with tqdm(total=len(self.test_loader), desc=f"Testing", unit="batch", leave=False) as pbar:
                for batch_idx, (anchor, positive, negative, label) in enumerate(self.test_loader):
                    anchor, positive, negative = anchor.to(self.device), positive.to(self.device), negative.to(self.device)

                    anchor_output = model(anchor)
                    positive_output = model(positive)
                    negative_output = model(negative)

                    loss = self.criterion(anchor_output, positive_output, negative_output)
                    total_loss += loss.item()

                    distances = F.pairwise_distance(anchor_output, positive_output)
                    closest_indices = distances.argsort()[:10]
                    correct += sum([1 for idx in closest_indices if label[idx] == label[0]])
                    total += len(label)

                    pbar.update(1)
                    accuracy = (correct / total) * 100 if total > 0 else 0
                    pbar.set_postfix({'loss': f'{total_loss/(batch_idx+1):.6f}', 'accuracy': f'{accuracy:.2f}%'})

        test_loss = total_loss / len(self.test_loader)
        test_accuracy = self.calculate_accuracy(model, self.test_loader, self.device, k=10)
        test_retrieval_accuracy, retrieval_results = self._calculate_retrieval_accuracy(model)
        return test_loss, test_accuracy, test_retrieval_accuracy, retrieval_results
    
    def calculate_accuracy(self, model, data_loader, device, k=10):
        model.eval()
        correct = 0
        total = 0
        all_embeddings = []
        all_labels = []
    
        with torch.no_grad():
            for (anchor, _, _, label) in data_loader:
                anchor = anchor.to(device)
                anchor_output = model(anchor)
                all_embeddings.append(anchor_output.cpu())
                all_labels.extend(label)  # 튜플이 아닌 리스트로 추가
    
        all_embeddings = torch.cat(all_embeddings)
    
        for i in range(len(all_embeddings)):
            query_embedding = all_embeddings[i]
            query_label = all_labels[i]
    
            distances = F.pairwise_distance(query_embedding.unsqueeze(0), all_embeddings)
            closest_indices = distances.argsort()[1:k+1]
            closest_labels = [all_labels[idx] for idx in closest_indices]
    
            if query_label in closest_labels:
                correct += 1
            total += 1
    
        accuracy = correct / total
        return accuracy * 100

    def _calculate_retrieval_accuracy(self, model):
        model.eval()
        
        train_features, train_labels = self._extract_features_labels(model, self.test_loader)
        test_features, test_labels = self._extract_features_labels(model, self.test_loader)
        
        total_correct_labels = 0
        total_labels = 0
        retrieval_results = []
        
        with torch.no_grad():
            for i, (feature, label) in enumerate(zip(test_features, test_labels)):
                neighbors = self._find_neighbors_cosine(train_features, feature, top_n=10)
                neighbor_labels = [train_labels[idx] for idx in neighbors]
                retrieval_results.append((i, neighbor_labels))
                
                correct_labels = neighbor_labels.count(label)
                total_correct_labels += correct_labels
                total_labels += len(neighbor_labels)
        
        retrieval_accuracy = (total_correct_labels / total_labels) * 100 if total_labels > 0 else 0
        return retrieval_accuracy, retrieval_results

    def _extract_features_labels(self, model, loader):
        model.eval()
        features = []
        labels = []
        
        with torch.no_grad():
            for inputs, _, _, label in loader:
                inputs = inputs.to(self.device)
                outputs = model(inputs)
                features.extend(outputs.cpu().numpy())
                labels.extend(label)
        
        return np.array(features), np.array(labels)

    def _find_neighbors_cosine(self, features, query_feature, top_n=10):
        features = torch.tensor(features).to(self.device)
        query_feature = torch.tensor(query_feature).unsqueeze(0).to(self.device)
        similarities = F.cosine_similarity(query_feature, features)
        neighbor_indices = torch.argsort(similarities, descending=True)[:top_n]
        return neighbor_indices.cpu().numpy()

    def _save_retrieval_results_to_csv(self, retrieval_results, model_path):
        model_name = model_path.split('/')[-1].split('.')[0]
        with open(f'{model_name}_retrieval_results.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            for i, neigh_labels in retrieval_results:
                writer.writerow([f'query{i+1}.png'] + list(neigh_labels))

# Example usage
if __name__ == '__main__':
    test_data_path = "C:/Users/L/Desktop/test_data"
    batch_size = 32
    model_pattern = "best_model_densenet201_epoch_*.pth"
    
    transform = transforms.Compose([
        transforms.Resize((120, 120)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_dataset = TripletImageDataset(test_data_path, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    model_class = SeResNeXtModel
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    tester = ModelTester(model_class, device, test_loader, model_pattern)
    tester.test_models()
