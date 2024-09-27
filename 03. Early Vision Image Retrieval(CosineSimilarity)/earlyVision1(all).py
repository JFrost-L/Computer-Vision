import numpy as np
import torch
import torch.nn.functional as F
import cv2
from skimage.feature import hog, local_binary_pattern, graycomatrix, graycoprops
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import time
import random
import csv
from sklearn.model_selection import StratifiedKFold
from scipy import signal as sg
from sklearn.preprocessing import LabelEncoder
import joblib  # For saving the model

Preprocessing = 1
FeatureExtraction = 1

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# SIFT feature extraction
def sift_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    if descriptors is None:
        return np.zeros(128)  # 디스크립터가 없으면 128차원의 0벡터 반환
    return np.mean(descriptors, axis=0)
# GLCM feature extraction
def glcm_features(image, distances=[1], angles=[0], levels=256, symmetric=True, normed=True):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    glcm = graycomatrix(gray, distances, angles, levels=levels, symmetric=symmetric, normed=normed)
    features = []
    for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']:
        feat = graycoprops(glcm, prop)
        features.append(feat.flatten())
    return np.concatenate(features)

# Law's texture feature extraction for a single channel
def laws_texture_channel(channel):
    (rows, cols) = channel.shape[:2]
    smooth_kernel = (1/25)*np.ones((5,5))
    channel_smooth = sg.convolve(channel, smooth_kernel, "same")
    channel_processed = np.abs(channel - channel_smooth)
    filter_vectors = np.array([[ 1,  4,  6,  4, 1],    # L5
                               [-1, -2,  0,  2, 1],    # E5
                               [-1,  0,  2,  0, 1],    # S5
                               [ 1, -4,  6, -4, 1]])   # R5
    filters = [np.matmul(filter_vectors[i].reshape(5,1), filter_vectors[j].reshape(1,5)) for i in range(4) for j in range(4)]
    conv_maps = np.zeros((rows, cols, 16))
    for i in range(len(filters)):
        conv_maps[:, :, i] = sg.convolve(channel_processed, filters[i], "same")
    
    texture_maps = [
        (conv_maps[:, :, 1] + conv_maps[:, :, 4]) // 2,  # L5E5 / E5L5
        (conv_maps[:, :, 2] + conv_maps[:, :, 8]) // 2,  # L5S5 / S5L5
        (conv_maps[:, :, 3] + conv_maps[:, :, 12]) // 2, # L5R5 / R5L5
        (conv_maps[:, :, 7] + conv_maps[:, :, 13]) // 2, # E5R5 / R5E5
        (conv_maps[:, :, 6] + conv_maps[:, :, 9]) // 2,  # E5S5 / S5E5
        (conv_maps[:, :, 11] + conv_maps[:, :, 14]) // 2,# S5R5 / R5S5
        conv_maps[:, :, 10],                            # S5S5
        conv_maps[:, :, 5],                             # E5E5
        conv_maps[:, :, 15],                            # R5R5
        conv_maps[:, :, 0]                              # L5L5 (use to norm TEM)
    ]
    TEM = [np.abs(texture_map).sum() / np.abs(texture_maps[9]).sum() for texture_map in texture_maps[:-1]]
    return TEM

# Law's texture feature extraction for color images
def laws_texture_color(image):
    channels = cv2.split(image)
    features = []
    for channel in channels:
        features.extend(laws_texture_channel(channel))
    return np.array(features)

# HOG feature extraction using PyTorch
def hog_features_torch(image):
    image = torch.from_numpy(image).float().permute(2, 0, 1).unsqueeze(0).to(device)
    image = F.interpolate(image, size=(120, 120), mode='bilinear', align_corners=False)
    image = image.squeeze(0).permute(1, 2, 0).cpu().numpy()
    fd, _ = hog(image, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=True, channel_axis=-1)
    return fd

# Color Histogram feature extraction
def color_histogram(image, bins=(8, 8, 8)):
    hist = cv2.calcHist([image], [0, 1, 2], None, bins, [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist

# LBP feature extraction
def lbp_features(image):
    radius = 1
    n_points = 8 * radius
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray_image, n_points, radius, method="uniform")
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)
    return hist

# Sobel Edge Detection
def sobel_edge_detection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_combined = np.hypot(sobelx, sobely)
    return sobel_combined.flatten()

# Canny Edge Detection
def canny_edge_detection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return edges.flatten()

# ORB feature extraction
def orb_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    if descriptors is None:
        return np.zeros(128)
    return np.mean(descriptors, axis=0)

# GPU accelerated image preprocessing using PyTorch
def preprocess_image_torch(image, target_size=(120, 120), use_histogram_equalization=True):
    global Preprocessing
    image = cv2.resize(image, target_size)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    intensity = hsv_image[:, :, 2]
    
    if use_histogram_equalization:
        intensity = intensity.astype(np.uint8)
        intensity = cv2.equalizeHist(intensity)
    
    hsv_image[:, :, 2] = intensity
    image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
    image = cv2.GaussianBlur(image, (5, 5), 0)
    print(f"image_{Preprocessing} 전처리 완료")
    Preprocessing += 1
    return image

# Feature extraction function using PyTorch
def extract_features_torch(image, use_histogram_equalization=True):
    global FeatureExtraction
    preprocessed_img = preprocess_image_torch(image, use_histogram_equalization=use_histogram_equalization)
    if (preprocessed_img is None) or (preprocessed_img.size == 0):
        return None
    preprocessed_img = cv2.convertScaleAbs(preprocessed_img)
    
    glcm_feature = glcm_features(preprocessed_img)
    laws_feature = laws_texture_color(preprocessed_img)
    hog_feature = hog_features_torch(preprocessed_img)
    color_hist_feature = color_histogram(preprocessed_img)
    lbp_feature = lbp_features(preprocessed_img)
    canny_feature = canny_edge_detection(preprocessed_img)
    orb_feature = orb_features(preprocessed_img)
    
    combined_features = np.hstack((glcm_feature, laws_feature,canny_feature, lbp_feature, hog_feature, color_hist_feature, orb_feature))
    
    print(f"image_{FeatureExtraction}의 feature vector 차원 : {len(combined_features)}")
    FeatureExtraction += 1
    
    return combined_features

# Function to pad features to the maximum length
def pad_features(features, max_length=None):
    if max_length is None:
        max_length = max(len(f) for f in features)
    padded_features = np.array([np.pad(f, (0, max_length - len(f)), 'constant') for f in features])
    return padded_features, max_length

# Load dataset with GPU-accelerated feature extraction
def load_dataset_torch(data_path, labels, num_samples_per_class, use_histogram_equalization=True):
    start_time = time.time()
    features = []
    labels_list = []
    
    for label in labels:
        image_dir = os.path.join(data_path, label)
        image_list = os.listdir(image_dir)
        sampled_images = random.sample(image_list, min(len(image_list), num_samples_per_class))
        
        for image_name in sampled_images:
            image_path = os.path.join(image_dir, image_name)
            img = cv2.imread(image_path)
            combined_features = extract_features_torch(img, use_histogram_equalization=use_histogram_equalization)
            if combined_features is not None:
                features.append(combined_features)
                labels_list.append(label)
    
    features, max_length = pad_features(features)
    
    end_time = time.time()
    print(f"Dataset loading and feature extraction time: {end_time - start_time} seconds")
    
    # Save max_length for later use
    np.save('max_length.npy', max_length)
    
    return np.array(features, dtype=float), np.array(labels_list), max_length

# Load test dataset
def load_test_dataset(data_path, labels, max_length, use_histogram_equalization=True):
    start_time = time.time()
    test_features = []
    test_labels = []
    
    for label in labels:
        image_dir = os.path.join(data_path, label)
        image_list = os.listdir(image_dir)
        
        for image_name in image_list:
            image_path = os.path.join(image_dir, image_name)
            img = cv2.imread(image_path)
            combined_features = extract_features_torch(img, use_histogram_equalization=use_histogram_equalization)
            if combined_features is not None:
                test_features.append(combined_features)
                test_labels.append(label)
    
    test_features, _ = pad_features(test_features, max_length)
    
    end_time = time.time()
    print(f"Test dataset loading and feature extraction time: {end_time - start_time} seconds")
    return np.array(test_features, dtype=float), np.array(test_labels)

def calculate_accuracy(neighbors, true_labels):
    retrieval_correct = 0
    total_neighbors = 0
    for true_label, neighbor_list in zip(true_labels, neighbors):
        retrieval_correct += sum(1 for neighbor in neighbor_list if true_label == neighbor)
        total_neighbors += len(neighbor_list)
    accuracy = retrieval_correct / total_neighbors * 100
    return accuracy

def main_torch():
    overall_start_time = time.time()
    print("start")
    train_data_path = "C:/Users/L/Desktop/CV/Large"
    test_data_path = "C:/Users/L/Desktop/test_data"
    labels = ['Bicycle', 'Bridge', 'Bus', 'Car', 'Chimney', 'Crosswalk', 'Hydrant', 'Motorcycle', 'Palm', 'Traffic Light']
    
    features, labels_list, max_length = load_dataset_torch(train_data_path, labels, num_samples_per_class=1448, use_histogram_equalization=True)
    
    if features is None:
        print("Feature extraction failed. Exiting...")
        return
    
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    val_accuracies = []  # List to store validation accuracies for each fold
    
    le = LabelEncoder()
    le.fit(labels)
    y_encoded = le.transform(labels_list)
    
    best_accuracy = 0
    best_scaler = None
    best_pca = None
    best_train_pca = None
    best_train_labels = None
    
    fold_num = 1
    for train_index, val_index in skf.split(features, y_encoded):
        X_train, X_val = features[train_index], features[val_index]
        y_train, y_val = y_encoded[train_index], y_encoded[val_index]
        
        # Preprocessing
        preprocessing_start_time = time.time()
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        
        pca = PCA(n_components=0.90, svd_solver='full')
        X_train_pca = pca.fit_transform(X_train)
        X_val_pca = pca.transform(X_val)
        
        preprocessing_end_time = time.time()
        print(f"Fold {fold_num} Preprocessing time: {preprocessing_end_time - preprocessing_start_time} seconds")
        
        def find_neighbors_cosine(train_features, query_feature, top_n=10):
            similarities = cosine_similarity([query_feature], train_features)
            neighbor_indices = np.argsort(similarities[0])[::-1][:top_n]
            return neighbor_indices
        
        all_val_neighbors = []
        for query_index, (query_feature, query_label) in enumerate(zip(X_val_pca, y_val)):
            val_neighbors = find_neighbors_cosine(X_train_pca, query_feature, top_n=10)
            neighbor_labels = y_train[val_neighbors]
            all_val_neighbors.append(neighbor_labels)
        
        val_retrieval_accuracy = calculate_accuracy(all_val_neighbors, y_val)
        print(f"Fold {fold_num} Validation retrieval Accuracy: {val_retrieval_accuracy}%")
        val_accuracies.append(val_retrieval_accuracy)
        
        if val_retrieval_accuracy > best_accuracy:
            best_accuracy = val_retrieval_accuracy
            best_scaler = scaler
            best_pca = pca
            best_train_pca = X_train_pca
            best_train_labels = y_train
            
            joblib.dump(best_scaler, 'best_scaler.pkl')
            joblib.dump(best_pca, 'best_pca.pkl')
            np.save('best_train_pca.npy', best_train_pca)
            np.save('best_train_labels.npy', best_train_labels)
        
        fold_num += 1
    
    avg_val_retrieval_accuracy = sum(val_accuracies) / len(val_accuracies)
    print(f"Average Validation retrieval Accuracy across all folds: {avg_val_retrieval_accuracy}%")
    print(f"Best Validation retrieval Accuracy: {best_accuracy}%")
    
    test_features, test_labels = load_test_dataset(test_data_path, labels, max_length, use_histogram_equalization=True)
    best_scaler = joblib.load('best_scaler.pkl')
    best_pca = joblib.load('best_pca.pkl')
    best_train_pca = np.load('best_train_pca.npy')
    best_train_labels = np.load('best_train_labels.npy')
    
    test_features = best_scaler.transform(test_features)
    test_features_pca = best_pca.transform(test_features)
    
    all_test_neighbors = []
    retrieval_start = time.time()
    
    with open('c1_t2_a1.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        for query_index in range(len(test_features_pca)):
            query_feature = test_features_pca[query_index]
            neighbor_indices = find_neighbors_cosine(best_train_pca, query_feature, top_n=10)
            neighbor_labels = best_train_labels[neighbor_indices]
            neighbor_labels_decoded = le.inverse_transform(neighbor_labels)  # Convert encoded labels back to original labels
            writer.writerow([f'test_query{query_index+1}.png'] + neighbor_labels_decoded.tolist())
            all_test_neighbors.append(neighbor_labels_decoded.tolist())
    
    retrieval_end = time.time()
    print(f"Image retrieval time: {retrieval_end - retrieval_start} seconds")
    
    test_retrieval_accuracy = calculate_accuracy(all_test_neighbors, test_labels)
    print(f"Test Image retrieval Accuracy: {test_retrieval_accuracy}%")
    
    overall_end_time = time.time()
    print(f"Total execution time: {overall_end_time - overall_start_time} seconds")

if __name__ == "__main__":
    main_torch()
