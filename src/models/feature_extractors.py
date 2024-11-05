"""
Model Initializations
Feature Extraction Functions (which use the models)
    parameters: image_path and preprocess (even if preprocess is not used)
    output:     features
Clustering functions to separate features
"""
import os
import cv2
import torch
import numpy as np
import timm
from PIL import Image
from torchvision.models import resnet50, resnet101
from torchvision.models import vgg16, vgg19
from torchvision.models import inception_v3
from torchvision.models import densenet121, densenet201
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------Model Initializations--------------
print("Load models")
print("ResNet50 and ResNet101, removing the final layer")
resnet50_model = torch.nn.Sequential(*(list(resnet50(pretrained=True).children())[:-1])).to(device).eval()
resnet101_model = torch.nn.Sequential(*(list(resnet101(pretrained=True).children())[:-1])).to(device).eval()

print("VGG16 and VGG19")
vgg16_model = vgg16(pretrained=True).features.to(device).eval()
vgg19_model = vgg19(pretrained=True).features.to(device).eval()

print("InceptionV3, removing the final layer")
inception_model = torch.nn.Sequential(
    *(list(inception_v3(pretrained=True, aux_logits=False).children())[:-1])).to(device).eval()

print("DenseNet models, removing the classifier")
densenet121_model = torch.nn.Sequential(*(list(densenet121(pretrained=True).children())[:-1])).to(device).eval()
densenet201_model = torch.nn.Sequential(*(list(densenet201(pretrained=True).children())[:-1])).to(device).eval()

print("EfficientNet (e.g., EfficientNet-B0)")
efficientnet_model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=0).to(device).eval()

print("Swin Transformer using timm")
swin_model = timm.create_model('swin_base_patch4_window7_224', pretrained=True, num_classes=0).to(device).eval()


# --------------Feature Extraction Functions--------------
def color_histogram_features(image_path, preprocess_img):
    """
    Extract color histogram features from an image.
    :param image_path: Path to the image.
    :param bins: Tuple specifying histogram bins for each color channel.
    :return: Flattened color histogram.
    """
    bins = (8, 8, 8)
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    hist = cv2.calcHist([image], [0, 1, 2], None, bins, [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist


def extract_features_vgg16(image_path, preprocess_img):
    image = Image.open(image_path).convert('RGB')
    image = preprocess_img(image).unsqueeze(0).to(device)  # Add batch dimension
    with torch.no_grad():
        features = vgg16(image).flatten().cpu().numpy()
    return features


def extract_features_vgg19(image_path, preprocess_img):
    image = Image.open(image_path).convert('RGB')
    image = preprocess_img(image).unsqueeze(0).to(device)  # Add batch dimension
    with torch.no_grad():
        features = vgg19(image).flatten().cpu().numpy()
    return features



def extract_features_resnet_50(image_path, preprocess_img):
    image = Image.open(image_path).convert('RGB')
    image = preprocess_img(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = resnet50_model(image).flatten().cpu().numpy()
    return features


def extract_features_resnet_101(image_path, preprocess_img):
    image = Image.open(image_path).convert('RGB')
    image = preprocess_img(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = resnet101_model(image).flatten().cpu().numpy()
    return features


def extract_features_inception(image_path, preprocess_img):
    image = Image.open(image_path).convert('RGB')
    image = preprocess_img(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = inception_model(image).flatten().cpu().numpy()
    return features


def extract_features_swin(image_path, preprocess_img):
    image = Image.open(image_path).convert('RGB')
    image = preprocess_img(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = swin_model(image).flatten().cpu().numpy()
    return features


def extract_features_efficientnet(image_path, preprocess_img):
    image = Image.open(image_path).convert('RGB')
    image = preprocess_img(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = efficientnet_model(image).flatten().cpu().numpy()
    return features


def extract_features_densenet121(image_path, preprocess_img):
    image = Image.open(image_path).convert('RGB')
    image = preprocess_img(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = densenet121_model(image).flatten().cpu().numpy()
    return features


def extract_features_densenet201(image_path, preprocess_img):
    image = Image.open(image_path).convert('RGB')
    image = preprocess_img(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = densenet201_model(image).flatten().cpu().numpy()
    return features


def load_or_calculate_features(image_paths, preprocess, extract_features_function, features_file_path, feature_extractor):
    """
    Save features for all images in a single .npz file. Skip recalculating if the file already exists.
    :param image_paths: List of image paths.
    :param extract_features_function: Function to extract features from an image.
    :param features_file_path: Path to save/load the features .npz file.
    :return: Dictionary mapping image paths to feature vectors.
    """
    # Check if features file already exists
    if os.path.exists(features_file_path):
        # Load existing features
        data = np.load(features_file_path, allow_pickle=True)
        features_dict = data['features'].item()
        print(f"Loaded features from {features_file_path}")
    else:
        print(f"Running {feature_extractor} feature extractor")

        # Calculate and save features
        features_dict = {}
        for img_path in image_paths:
            features_dict[img_path] = extract_features_function(img_path, preprocess)

        # Save features to .npz file
        np.savez_compressed(features_file_path, features=features_dict)
        print(f"Saved all features to {features_file_path}")

    return features_dict


# --------------Clustering Methods--------------
def generic_clustering(image_paths, preprocess, features_output_path, extract_features_function,
                       extraction_method, n_clusters=5, clustering_method='kmeans', **kwargs):
    """
    Cluster images using specified clustering method and feature extraction function.
    :param image_paths: List of paths to image files.
    :param features_output_path: Directory path to save/load the features file.
    :param extract_features_function: Function to extract features from an image.
    :param extraction_method: Name of the feature extraction method.
    :param n_clusters: Number of clusters (if applicable to the clustering method).
    :param clustering_method: Clustering method ('kmeans', 'agglomerative', or 'dbscan').
    :param kwargs: Additional keyword arguments for the clustering algorithm.
    :return: Trained clustering model, dictionary mapping image paths to cluster labels.
    """
    # Extract features for all images
    features_file_path = os.path.join(features_output_path, f"{extraction_method}.npz")
    features_dict = load_or_calculate_features(image_paths, preprocess, extract_features_function,
                                               features_file_path, extraction_method)

    # Extract feature vectors as a list from the dictionary
    features = list(features_dict.values())

    # Select clustering model based on the specified method
    if clustering_method == 'kmeans':
        model = KMeans(n_clusters=n_clusters, random_state=42, **kwargs)
    elif clustering_method == 'agglomerative':
        model = AgglomerativeClustering(n_clusters=n_clusters, **kwargs)
    elif clustering_method == 'dbscan':
        model = DBSCAN(**kwargs)
    else:
        raise ValueError("Invalid clustering method. Choose from 'kmeans', 'agglomerative', or 'dbscan'.")

    # Fit and predict cluster labels
    labels = model.fit_predict(features)

    # Map image paths to their cluster labels
    image_clusters = {img_path: label for img_path, label in zip(image_paths, labels)}
    return model, image_clusters