import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from torchvision import datasets
import random
from PIL import Image
import numpy as np
import csv
import albumentations as A
from albumentations.pytorch import ToTensorV2
import glob

class ProxyAnchorLoss(nn.Module):
    def __init__(self, num_classes, embedding_size, margin=0.1, alpha=32):
        super(ProxyAnchorLoss, self).__init__()
        self.num_classes = num_classes
        self.embedding_size = embedding_size
        self.margin = margin
        self.alpha = alpha
        self.proxies = nn.Parameter(torch.randn(num_classes, embedding_size))
        nn.init.kaiming_normal_(self.proxies, mode='fan_out')

    def forward(self, embeddings, labels):
        device = embeddings.device
        proxies = self.proxies.to(device)
        
        cosine_sim = F.linear(F.normalize(embeddings), F.normalize(proxies))
        labels_one_hot = F.one_hot(labels, num_classes=self.num_classes).float()
        
        pos_mask = labels_one_hot > 0
        neg_mask = labels_one_hot == 0

        pos_exp = torch.exp(-self.alpha * (cosine_sim - self.margin))
        neg_exp = torch.exp(self.alpha * (cosine_sim + self.margin))

        pos_term = torch.log(1 + pos_exp[pos_mask].sum())
        neg_term = torch.log(1 + neg_exp[neg_mask].sum())

        loss = pos_term + neg_term
        return loss.mean()

# 동일한 DenseNet 모델 정의
class DenseNetModel(nn.Module):
    def __init__(self, num_classes=10):
        super(DenseNetModel, self).__init__()
        self.model = timm.create_model('densenet169', pretrained=True)
        self.embedding_size = self.model.classifier.in_features
        self.model.classifier = nn.Identity()
        self.fc = nn.Linear(self.embedding_size, num_classes)
        self.name = 'densenet201'

    def forward(self, x):
        x = self.model(x)
        return x

# 동일한 TripletImageDataset 정의
class TripletImageDataset(Dataset):
    def __init__(self, image_folder, transform=None):
        self.image_folder = image_folder
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.label_to_indices = {}
        self.idx_to_class = {}
        self._prepare_data()

    def _prepare_data(self):
        dataset = datasets.ImageFolder(self.image_folder)
        self.image_paths = [img[0] for img in dataset.imgs]
        self.labels = [img[1] for img in dataset.imgs]
        self.idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}
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

        if self.transform:
            anchor_image = self.transform(image=np.array(anchor_image))['image']
            positive_image = self.transform(image=np.array(positive_image))['image']
            negative_image = self.transform(image=np.array(negative_image))['image']

        return anchor_image, positive_image, negative_image, anchor_label

# 테스트용 모델 불러오기 및 평가
class ModelTester:
    def __init__(self, model_class, device, test_loader, model_pattern):
        self.device = torch.device(device)
        self.test_loader = test_loader
        self.model_class = model_class
        self.model_pattern = model_pattern

    def test_models(self):
        model_paths = glob.glob(self.model_pattern)
        for model_path in model_paths:
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
            for batch_idx, (anchor, positive, negative, labels) in enumerate(tqdm(self.test_loader, desc="Testing", unit="batch")):
                anchor, positive, negative, labels = anchor.to(self.device), positive.to(self.device), negative.to(self.device), labels.to(self.device)

                anchor_output = model(anchor)
                positive_output = model(positive)
                negative_output = model(negative)

                loss = self._calculate_loss(anchor_output, positive_output, negative_output, labels)
                total_loss += loss.item()

                accuracy = self.calculate_accuracy(anchor_output, positive_output, negative_output, labels)
                correct += accuracy
                total += labels.size(0)

        test_loss = total_loss / len(self.test_loader)
        test_accuracy = (correct / total) * 100
        test_retrieval_accuracy, retrieval_results = self._calculate_retrieval_accuracy(model)
        return test_loss, test_accuracy, test_retrieval_accuracy, retrieval_results

    def _calculate_loss(self, anchor, positive, negative, labels):
        criterion = ProxyAnchorLoss(num_classes=10, embedding_size=anchor.size(1))
        embeddings = torch.cat((anchor, positive, negative))
        combined_labels = torch.cat((labels, labels, labels))
        return criterion(embeddings, combined_labels)

    def calculate_accuracy(self, anchor, positive, negative, labels):
        positive_distance = F.pairwise_distance(anchor, positive)
        negative_distance = F.pairwise_distance(anchor, negative)
        accuracy = (positive_distance < negative_distance).sum().item()
        return accuracy

    def _calculate_retrieval_accuracy(self, model):
        model.eval()
        test_features, test_labels = self._extract_features_labels(model, self.test_loader)
        total_correct_labels = 0
        total_labels = 0
        retrieval_results = []

        with torch.no_grad():
            for i, (feature, label) in enumerate(zip(test_features, test_labels)):
                neighbors = self._find_neighbors_cosine(test_features, feature, top_n=10)
                neighbor_labels = [test_labels[idx] for idx in neighbors]
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
                outputs = model(inputs).cpu().numpy()
                features.extend(outputs)
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
                neigh_label_names = [self.test_loader.dataset.idx_to_class[idx] for idx in neigh_labels]
                writer.writerow([f'query{i+1}.png'] + list(neigh_label_names))

# Example usage
if __name__ == '__main__':
    test_data_path = "C:/Users/L/Desktop/test_data"
    batch_size = 32
    model_pattern = "best_model_densenet201_fold_*.pth"

    transform = A.Compose([
        A.Resize(120, 120),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    test_dataset = TripletImageDataset(test_data_path, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model_class = DenseNetModel
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    tester = ModelTester(model_class, device, test_loader, model_pattern)
    tester.test_models()
