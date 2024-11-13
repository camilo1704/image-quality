"""
Use PREPROCESS to change transformations on the image before entering the feature extractor models

FEATURE_EXTRACTORS used are listed as (name, extractor function from models.feature_extractors)

CLUSTERING_METHOD use kmeans by default but could be changed to agglomerative or dbscan
"""
import torchvision.transforms as transforms
from models.feature_extractors import (color_histogram_features,
                                       extract_features_resnet_50,
                                       extract_features_resnet_101,
                                       extract_features_inception,
                                       extract_features_vgg16,
                                       extract_features_vgg19,
                                       extract_features_efficientnet,
                                       extract_features_swin)

PREPROCESS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # imagenet mean and std
])

FEATURE_EXTRACTORS = [
    # Methods that work decently with just CPU
    ("color_histogram", color_histogram_features),
    ("ResNet50", extract_features_resnet_50),
    ("resnet_101", extract_features_resnet_101),
    ("inception", extract_features_inception),
    ("vgg16", extract_features_vgg16),
    ("vgg19", extract_features_vgg19),
    ("efficientnet", extract_features_efficientnet),
    ("swin", extract_features_swin)

    # Methods that are very slow to be run without GPU
    # ("densenet121", extract_features_densenet121),
    # ("densenet201", extract_features_densenet201),
]

# Options: kmeans, agglomerative, dbscan
CLUSTERING_METHOD = "kmeans"
