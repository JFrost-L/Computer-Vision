import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import timm
import numpy as np
import random
from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
import matplotlib.pyplot as plt
import csv
import albumentations as A
from torch.optim.swa_utils import AveragedModel, SWALR
from albumentations.pytorch import ToTensorV2

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
        
        # F.linear 대신에 F.cosine_similarity로 대체
        cosine_sim = F.cosine_similarity(F.normalize(embeddings).unsqueeze(1), F.normalize(proxies).unsqueeze(0), dim=2)
        labels_one_hot = F.one_hot(labels, num_classes=self.num_classes).float()
        
        pos_mask = labels_one_hot > 0
        neg_mask = labels_one_hot == 0

        pos_exp = torch.exp(-self.alpha * (cosine_sim - self.margin))
        neg_exp = torch.exp(self.alpha * (cosine_sim + self.margin))

        pos_term = torch.log(1 + pos_exp[pos_mask].sum())
        neg_term = torch.log(1 + neg_exp[neg_mask].sum())

        loss = pos_term + neg_term
        return loss.mean()



class ResNeXtModel(nn.Module):
    def __init__(self, num_classes=10, embedding_dim=128):
        super(ResNeXtModel, self).__init__()
        self.model = timm.create_model('resnext50d_32x4d', pretrained=True)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, embedding_dim)
        self.embedding_size = embedding_dim  # Define the embedding_size attribute
        self.name = 'resnext101_64x4d'  # Define the name attribute

    def forward(self, x):
        embeddings = self.model(x)
        return embeddings



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
    
class ModelTrainer:
    def __init__(self, model, device, train_loader, val_loader, test_loader, epochs=30, learning_rate=0.00001, fold=0):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.criterion = ProxyAnchorLoss(num_classes=10, embedding_size=model.embedding_size)
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=3, factor=0.5)
        self.epochs = epochs
        self.train_losses = []
        self.val_losses = []
        self.best_val_retrieval_accuracy = 0.0
        self.best_epoch = 0
        self.fold = fold
        self.swa_model = AveragedModel(model)
        self.swa_scheduler = SWALR(self.optimizer, swa_lr=learning_rate)

    def train_and_evaluate(self):
        for epoch in range(self.epochs):
            train_loss = self._train(self.model, self.optimizer, epoch)
            val_loss, val_retrieval_accuracy = self._validate(self.model, epoch)
            self.scheduler.step(val_loss)
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            if val_retrieval_accuracy > self.best_val_retrieval_accuracy:
                self.best_val_retrieval_accuracy = val_retrieval_accuracy
                self.best_epoch = epoch
                self._save_model(self.model, self.model.name, self.fold)

        self._final_evaluate()
        self.plot_losses()

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
                all_labels.extend(label)
    
        all_embeddings = torch.cat(all_embeddings)
    
        for i in range(len(all_embeddings)):
            query_embedding = all_embeddings[i]
            query_label = all_labels[i]
    
            similarities = F.cosine_similarity(query_embedding.unsqueeze(0), all_embeddings)
            closest_indices = similarities.argsort(descending=True)[1:k+1]
            closest_labels = [all_labels[idx] for idx in closest_indices]
    
            if query_label in closest_labels:
                correct += 1
            total += 1
    
        accuracy = correct / total
        return accuracy * 100

    def _train(self, model, optimizer, epoch):
        model.train()
        total_loss = 0.0
        with tqdm(total=len(self.train_loader), desc=f"Epoch {epoch + 1}/{self.epochs} - {model.name}", unit="batch", leave=False) as pbar:
            for batch_idx, (anchor, positive, negative, labels) in enumerate(self.train_loader):
                anchor, positive, negative, labels = anchor.to(self.device), positive.to(self.device), negative.to(self.device), labels.to(self.device)
    
                optimizer.zero_grad()
    
                anchor_output = model(anchor)
                positive_output = model(positive)
                negative_output = model(negative)
                embeddings = torch.cat((anchor_output, positive_output, negative_output))
                
                combined_labels = torch.cat((labels, labels, labels))
    
                loss = self.criterion(embeddings, combined_labels)
                loss.backward()
                optimizer.step()
    
                total_loss += loss.item()
    
                pbar.update(1)
    
        train_loss = total_loss / len(self.train_loader)
        train_accuracy = self.calculate_accuracy(model, self.train_loader, self.device, k=10)
        print(f'Train Epoch {epoch + 1}/{self.epochs} - {model.name} - Loss: {train_loss:.6f} - Accuracy: {train_accuracy:.2f}%', end=' ')
        return train_loss
    
    def _validate(self, model, epoch):
        model.eval()
        total_loss = 0.0
        with torch.no_grad():
            with tqdm(total=len(self.val_loader), desc=f"Validating Epoch {epoch + 1}/{self.epochs} - {model.name}", unit="batch", leave=False) as pbar:
                for batch_idx, (anchor, positive, negative, labels) in enumerate(self.val_loader):
                    anchor, positive, negative, labels = anchor.to(self.device), positive.to(self.device), negative.to(self.device), labels.to(self.device)
    
                    anchor_output = model(anchor)
                    positive_output = model(positive)
                    negative_output = model(negative)
                    embeddings = torch.cat((anchor_output, positive_output, negative_output))
                    
                    combined_labels = torch.cat((labels, labels, labels))
    
                    loss = self.criterion(embeddings, combined_labels)
                    total_loss += loss.item()
    
                    pbar.update(1)
    
        val_loss = total_loss / len(self.val_loader)
        val_accuracy = self.calculate_accuracy(model, self.val_loader, self.device, k=10)
        val_retrieval_accuracy, _ = self._calculate_retrieval_accuracy(model)
        print(f'Validation Epoch {epoch + 1}/{self.epochs} - {model.name} - Loss: {val_loss:.6f} - Accuracy: {val_accuracy:.2f}% - Retrieval Accuracy: {val_retrieval_accuracy:.2f}%')
        return val_loss, val_retrieval_accuracy

    def _final_evaluate(self):
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            with tqdm(total=len(self.test_loader), desc=f"Testing", unit="batch", leave=False) as pbar:
                for batch_idx, (anchor, positive, negative, labels) in enumerate(self.test_loader):
                    anchor, positive, negative, labels = anchor.to(self.device), positive.to(self.device), negative.to(self.device), labels.to(self.device)
    
                    anchor_output = self.model(anchor)
                    positive_output = self.model(positive)
                    negative_output = self.model(negative)
                    embeddings = torch.cat((anchor_output, positive_output, negative_output))
                    
                    combined_labels = torch.cat((labels, labels, labels))
    
                    loss = self.criterion(embeddings, combined_labels)
                    total_loss += loss.item()
    
                    pbar.update(1)
    
        test_loss = total_loss / len(self.test_loader)
        test_accuracy = self.calculate_accuracy(self.model, self.test_loader, self.device, k=10)
        test_retrieval_accuracy, retrieval_results = self._calculate_retrieval_accuracy(self.model)
        self._save_retrieval_results_to_csv(retrieval_results)
        return test_loss, test_accuracy, test_retrieval_accuracy


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

    def _save_model(self, model, model_name, fold):
        torch.save(model.state_dict(), f"best_model_{model_name}_fold_{fold}.pth")

    def _load_model(self, model, model_name, fold):
        model.load_state_dict(torch.load(f"best_model_{model_name}_fold_{fold}.pth"))
        model.to(self.device)

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
    
    def _find_neighbors_cosine(self, features, query_feature, top_n=10):
        features = torch.tensor(features).to(self.device)
        query_feature = torch.tensor(query_feature).unsqueeze(0).to(self.device)
        similarities = F.cosine_similarity(query_feature, features)
        neighbor_indices = torch.argsort(similarities, descending=True)[:top_n]
        return neighbor_indices.cpu().numpy()

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
    
    def _save_retrieval_results_to_csv(self, retrieval_results):
        with open(f'retrieval_results_fold_{self.fold}.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            for i, neigh_labels in retrieval_results:
                neigh_label_names = [self.test_loader.dataset.idx_to_class[idx] for idx in neigh_labels]
                writer.writerow([f'query{i+1}.png'] + list(neigh_label_names))

# Example usage
if __name__ == '__main__':
    train_data_path = "C:/Users/L/Desktop/CV/Large"
    # train_data_path = "C:/Users/L/Desktop/test_data"
    test_data_path = "C:/Users/L/Desktop/test_data"
    batch_size = 32
    learning_rate = 1e-4
    epochs = 20
    num_splits = 5

    # 데이터 증강
    trainTransform = A.Compose([
        A.Resize(120, 120),
        A.Rotate(limit=30, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.2, p=0.5),
        A.Affine(scale=(0.8, 1.2), translate_percent=(0.1, 0.1), rotate=(-30, 30), shear=(-10, 10), p=0.5),
        A.GaussianBlur(p=0.5),
        A.CoarseDropout(p=0.5, max_holes=1, max_height=30, max_width=30, min_holes=1, min_height=10, min_width=10),
        A.RandomResizedCrop(height=120, width=120, scale=(0.8, 1.0), p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    
    transform = A.Compose([
        A.Resize(120, 120),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    full_dataset = TripletImageDataset(train_data_path, transform=None)
    labels = full_dataset.labels

    skf = StratifiedKFold(n_splits=num_splits)

    all_fold_results = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        print(f"Training fold {fold + 1}/{num_splits}")
        
        train_subset = Subset(full_dataset, train_idx)
        val_subset = Subset(full_dataset, val_idx)
        
        train_subset.dataset.transform = trainTransform
        val_subset.dataset.transform = transform
        
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, pin_memory=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, pin_memory=True)
        
        test_dataset = TripletImageDataset(test_data_path, transform=transform)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
        
        model = ResNeXtModel()
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        trainer = ModelTrainer(model, device, train_loader, val_loader, test_loader, epochs=epochs, learning_rate=learning_rate, fold=fold+1)
        trainer.train_and_evaluate()
        
        fold_results = {
            'fold': fold + 1,
            'train_loss': trainer.train_losses,
            'val_loss': trainer.val_losses,
            'best_epoch': trainer.best_epoch, 
            'best_val_retrieval_accuracy': trainer.best_val_retrieval_accuracy,
        }
        all_fold_results.append(fold_results)

    for fold_result in all_fold_results:
        fold = fold_result['fold']
        print(f"Testing fold {fold} best model")
        
        model = ResNeXtModel()
        trainer = ModelTrainer(model, device, None, None, test_loader, epochs=epochs, learning_rate=learning_rate, fold=fold)
        trainer._load_model(model, model.name, fold)
        
        test_loss, test_accuracy, test_retrieval_accuracy = trainer._final_evaluate()
        
        fold_result['test_loss'] = test_loss
        fold_result['test_accuracy'] = test_accuracy
        fold_result['test_retrieval_accuracy'] = test_retrieval_accuracy
        
        print(f"Fold {fold} - Test Loss: {test_loss:.6f} - Test Accuracy: {test_accuracy:.2f}% - Test Retrieval Accuracy: {test_retrieval_accuracy:.2f}%")

    avg_train_loss = np.mean([fold['train_loss'][-1] for fold in all_fold_results])
    avg_val_loss = np.mean([fold['val_loss'][-1] for fold in all_fold_results])
    avg_best_val_retrieval_accuracy = np.mean([fold['best_val_retrieval_accuracy'] for fold in all_fold_results])
    avg_test_loss = np.mean([fold['test_loss'] for fold in all_fold_results])
    avg_test_accuracy = np.mean([fold['test_accuracy'] for fold in all_fold_results])
    avg_test_retrieval_accuracy = np.mean([fold['test_retrieval_accuracy'] for fold in all_fold_results])
    
    print(f"Average train loss over all folds: {avg_train_loss:.6f}")
    print(f"Average validation loss over all folds: {avg_val_loss:.6f}")
    print(f"Average best validation retrieval accuracy over all folds: {avg_best_val_retrieval_accuracy:.2f}%")
    print(f"Average test loss over all folds: {avg_test_loss:.6f}")
    print(f"Average test accuracy over all folds: {avg_test_accuracy:.2f}%")
    print(f"Average test retrieval accuracy over all folds: {avg_test_retrieval_accuracy:.2f}%")
