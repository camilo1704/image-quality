"""
ATTRIBUTES dictionary is used to configure the attribute score functions being used

-score_function: Function to extract the attribute score from an image path
-thresholds: List of thresholds to separate clusters according to score distribution
              will be showed in the distribution grraph as red lines
-labels: List of names of the clusters or tags for each group (Example: "very_dark_images", "dark_images",etc)
-x_limits_for_distribution_graph: X axis limits to zoom in the distribution graph for readability. Example: [0, 10]

CLUSTERING_METHOD use kmeans by default but could be changed to agglomerative or dbscan
"""
from src.models.image_attributes_score_algorithms import *

ATTRIBUTES = {
        "Brightness": {
            "thresholds": [0.1, 0.25, 0.77, 0.83],
            "labels": ["very_dark_images",
                       "dark_images",
                       "normal_images",
                       "bright_images",
                       "very_bright_images"],
            "score_function": image_lightness,
            "x_limits_for_distribution_graph": None
        },
        "Blur fft": {
            "thresholds": [5.5, 6.1],
            "labels": ["very_blurred_images",
                       "blurred_images",
                       "normal_images"],
            "score_function": calculate_blur_fft,
            "x_limits_for_distribution_graph": None
        },
        "Blur sobel": {
            "thresholds": [50, 100],
            "labels": ["very_blurred_images",
                       "blurred_images",
                       "normal_images"],
            "score_function": calculate_blur_sobel,
            "x_limits_for_distribution_graph": None
        },
        "Blackness": {
            "thresholds": [2, 5, 20, 99],
            "labels": ["0-2%",
                       "2-5%",
                       "5-20%",
                       "20-99%",
                       "99-100%"],
            "score_function": calculate_black_pixel_percentage,
            "x_limits_for_distribution_graph": [0, 10]
        },
        "Aspect Ratio": {
            "thresholds": [1.85, 3.45, 4],
            "labels": ["slightly_long_images",
                       "normal_images",
                       "long_images",
                       "very_long_images"],
            "score_function": image_aspect_ratio,
            "x_limits_for_distribution_graph": [0, 10]
        },
        "Dominant Color": {
            "thresholds": [15, 20, 40, 70],
            "labels": ["0-15", "15-20", "20-40", "40-70", "70+"],
            "score_function": calculate_dominant_color,
            "x_limits_for_distribution_graph": [0, 80]
        },
        "Contrast": {
            "thresholds": [10, 20, 30, 50],
            "labels": ["0-10", "10-20", "20-30", "30-50", "50+"],
            "score_function": calculate_contrast,
            "x_limits_for_distribution_graph": None
        },
        "Saturation": {
            "thresholds": [35, 80, 120, 180],
            "labels": ["0-35", "35-80", "80-120", "120-180", "180+"],
            "score_function": calculate_saturation,
            "x_limits_for_distribution_graph": None
        },
    }
