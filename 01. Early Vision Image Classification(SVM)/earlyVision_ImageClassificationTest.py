import numpy as np
import torch
import torch.nn.functional as F
import cv2
from skimage.feature import hog, local_binary_pattern, graycomatrix, graycoprops
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import time
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from scipy import signal as sg
import csv

Preprocessing = 1
FeatureExtraction = 1

# CUDA 사용 여부 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# GLCM 특징 추출 함수
def glcm_features(image, distances=[1], angles=[0], levels=256, symmetric=True, normed=True):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    glcm = graycomatrix(gray, distances, angles, levels=levels, symmetric=symmetric, normed=normed)
    features = []
    for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']:
        feat = graycoprops(glcm, prop)
        features.append(feat.flatten())
    return np.concatenate(features)

# Law's texture 특징 추출 함수
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

# Law's texture 색상 이미지에 대한 특징 추출 함수
def laws_texture_color(image):
    channels = cv2.split(image)
    features = []
    for channel in channels:
        features.extend(laws_texture_channel(channel))
    return np.array(features)

# HOG 특징 추출 함수
def hog_features_torch(image):
    image = torch.from_numpy(image).float().permute(2, 0, 1).unsqueeze(0).to(device)
    image = F.interpolate(image, size=(120, 120), mode='bilinear', align_corners=False)
    image = image.squeeze(0).permute(1, 2, 0).cpu().numpy()
    fd, _ = hog(image, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=True, channel_axis=-1)
    return fd

# 색상 히스토그램 특징 추출 함수
def color_histogram(image, bins=(8, 8, 8)):
    hist = cv2.calcHist([image], [0, 1, 2], None, bins, [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist

# LBP 특징 추출 함수
def lbp_features(image):
    radius = 1
    n_points = 8 * radius
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray_image, n_points, radius, method="uniform")
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)
    return hist

# Canny 에지 검출 함수
def canny_edge_detection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return edges.flatten()

# ORB 특징 추출 함수
def orb_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    if descriptors is None:
        return np.zeros(128)
    return np.mean(descriptors, axis=0)

# GPU 가속화 이미지 전처리 함수
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

# 특징 추출 함수
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
    
    combined_features = np.hstack((glcm_feature, laws_feature, hog_feature, color_hist_feature, lbp_feature, canny_feature, orb_feature))
    
    print(f"image_{FeatureExtraction}의 feature vector 차원 : {len(combined_features)}")
    FeatureExtraction += 1
    
    return combined_features

# 특징 벡터를 최대 길이에 맞게 패딩
def pad_features(features, max_length=None):
    if max_length is None:
        max_length = max(len(f) for f in features)
    padded_features = np.array([np.pad(f, (0, max_length - len(f)), 'constant') for f in features])
    return padded_features, max_length

# 테스트 데이터셋 로드 함수
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

# 혼동 행렬 시각화 및 저장 함수
def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.figure(figsize=(10, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(f'{title}.png')
    plt.show()

# 모델 및 메타데이터 로드 함수
def load_model_and_metadata(filename='best_svm_model.pkl'):
    data = joblib.load(filename)
    return data['model'], data['scaler'], data['pca'], data['max_length']

# 독립적인 테스트 함수
def independent_test(test_data_path, model_path, labels):
    print("Independent test start")
    
    # 모델, 스케일러, PCA 및 max_length 로드
    best_model, scaler, pca, max_length = load_model_and_metadata(model_path)
    
    # 테스트 데이터 로드 및 특징 추출
    test_features, test_labels = load_test_dataset(test_data_path, labels, max_length, use_histogram_equalization=True)
    
    # 전처리
    test_features = scaler.transform(test_features)
    
    # PCA 적용
    test_features_pca = pca.transform(test_features)
    
    le = LabelEncoder()
    le.fit(labels)
    test_labels_encoded = le.transform(test_labels)
    
    # 테스트 데이터 평가
    test_start = time.time()
    test_pred_labels = best_model.predict(test_features_pca)
    test_end = time.time()
    
    print("Test set classification report:")
    print(classification_report(test_labels_encoded, test_pred_labels, target_names=le.classes_))
    print(f"Test set evaluation time: {test_end - test_start} seconds")
    
    cm_test = confusion_matrix(test_labels_encoded, test_pred_labels)
    plot_confusion_matrix(cm_test, le.classes_, title='Confusion matrix for test set')
    
    test_accuracy = accuracy_score(test_labels_encoded, test_pred_labels)
    print(f"Test set accuracy: {test_accuracy}")
    
    with open('c1_t1_a1_test.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        for i, predict_label in enumerate(test_pred_labels):
            writer.writerow([f'query{i+1}.png', le.inverse_transform([predict_label])[0]])

if __name__ == "__main__":
    test_data_path = "C:/Users/L/Desktop/test_data"
    model_path = "best_svm_model.pkl"
    labels = ['Bicycle', 'Bridge', 'Bus', 'Car', 'Chimney', 'Crosswalk', 'Hydrant', 'Motorcycle', 'Palm', 'Traffic Light']
    
    independent_test(test_data_path, model_path, labels)
