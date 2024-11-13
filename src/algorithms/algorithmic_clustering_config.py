"""
ATTRIBUTES dictionary is used to configure the attribute score functions being used

-score_function: Function to extract the attribute score from an image path
-thresholds: List of thresholds to separate clusters according to score distribution
              will be showed in the distribution graph as red lines
-labels: List of names of the clusters or tags for each group (Example: "very_dark_images", "dark_images",etc.)
-x_limits_for_distribution_graph: X axis limits to zoom in the distribution graph for readability. Example: [0, 10]

CLUSTERING_METHOD use kmeans by default but could be changed to agglomerative or dbscan
"""
from src.algorithms.image_attributes_score_algorithms import (image_lightness,
                                                              calculate_colorfulness,
                                                              calculate_color_entropy,
                                                              calculate_hue_variability,
                                                              calculate_brown_spectrum_detection,
                                                              calculate_earth_tones_percentage,
                                                              calculate_red_spectrum_dominance,
                                                              calculate_contrast,
                                                              calculate_saturation,
                                                              calculate_blur_fft,
                                                              calculate_blur_sobel,
                                                              calculate_dominant_color,
                                                              calculate_black_pixel_percentage,
                                                              image_aspect_ratio)

ATTRIBUTES = {
    "Brightness": {
        # "thresholds": [0.1, 0.25, 0.77, 0.83],    # Maturity
        "thresholds": [0.2, 0.35, 0.6, 0.75],       # Stand-count
        "labels": ["very_dark_images",
                   "dark_images",
                   "normal_images",
                   "bright_images",
                   "very_bright_images"],
        "score_function": image_lightness,
        "x_limits_for_distribution_graph": None
    },
    "Blur fft": {
        # "thresholds": [5.5, 6.1],     # Maturity
        "thresholds": [6.5, 8],         # Stand-count
        "labels": ["very_blurred_images",
                   "blurred_images",
                   "normal_images"],
        "score_function": calculate_blur_fft,
        "x_limits_for_distribution_graph": None
    },
    "Blur sobel": {
        # "thresholds": [50, 100],      # Maturity
        "thresholds": [40, 80],         # Stand-count
        "labels": ["very_blurred_images", "blurred_images", "normal_images"],
        "score_function": calculate_blur_sobel,
        "x_limits_for_distribution_graph": None
    },
    "Blackness": {
        # "thresholds": [2, 5, 20, 99],                             # Maturity
        # "labels": ["0-2%", "2-5%", "5-20%", "20-99%", "99-100%"], # Maturity
        "thresholds": [2, 5, 20],                                   # Stand-count
        "labels": ["0-2%", "2-5%", "5-20%", "20-100%"],             # Stand-count
        "score_function": calculate_black_pixel_percentage,
        "x_limits_for_distribution_graph": [0, 10]
    },
    "Aspect Ratio": {
        # "thresholds": [1.85, 3.45, 4],    # Maturity
        "thresholds": [7.5, 12, 23],        # Stand-count
        "labels": ["slightly_long_images", "normal_images", "long_images", "very_long_images"],
        "score_function": image_aspect_ratio,
        "x_limits_for_distribution_graph": [0, 10]
    },
    "Dominant Color": {
        # "thresholds": [15, 20, 40, 70],       # Maturity
        "thresholds": [12, 28, 38, 65],         # Stand-count
        "labels": ["0-12", "12-28", "28-38", "38-65", "65+"],
        "score_function": calculate_dominant_color,
        "x_limits_for_distribution_graph": [0, 80]
    },
    "Contrast": {
        # "thresholds": [10, 20, 30, 50],                           # Maturity
        # "labels": ["0-10", "10-20", "20-30", "30-50", "50+"],     # Maturity
        "thresholds": [15, 25, 50, 65],                             # Stand-count
        "labels": ["0-15", "15-25", "25-50", "50-65", "65+"],       # Stand-count
        "score_function": calculate_contrast,
        "x_limits_for_distribution_graph": None
    },
    "Saturation": {
        # "thresholds": [35, 80, 120, 180],                         # Maturity
        # "labels": ["0-25", "25-40", "40-72", "72-120", "120+"],   # Maturity
        "thresholds": [25, 40, 72, 120],                            # Stand-count
        "labels": ["0-35", "35-80", "80-120", "120-180", "180+"],  # Stand-count
        "score_function": calculate_saturation,
        "x_limits_for_distribution_graph": None
    },
    "Hue_variability": {
        # "thresholds": [35, 80, 120, 180],                         # Maturity
        # "labels": ["0-25", "25-40", "40-72", "72-120", "120+"],   # Maturity
        "thresholds": [9, 25],                                      # Stand-count
        "labels": ["0-9", "9-25", "25+"],                           # Stand-count
        "score_function": calculate_hue_variability,
        "x_limits_for_distribution_graph": None
    },
    "Colorfulness": {
        # "thresholds": [35, 80, 120, 180],                         # Maturity
        # "labels": ["0-25", "25-40", "40-72", "72-120", "120+"],   # Maturity
        "thresholds": [22, 38],                                     # Stand-count
        "labels": ["0-22", "22-38", "38+"],                         # Stand-count
        "score_function": calculate_colorfulness,
        "x_limits_for_distribution_graph": None
    },
    "Color_entropy": {
        # "thresholds": [35, 80, 120, 180],                         # Maturity
        # "labels": ["0-25", "25-40", "40-72", "72-120", "120+"],   # Maturity
        "thresholds": [2.4, 4.5],                                   # Stand-count
        "labels": ["0-2.4", "2.4-4.5", "4.5+"],                     # Stand-count
        "score_function": calculate_color_entropy,
        "x_limits_for_distribution_graph": None
    },
    "earth_tone_percentage": {
        # "thresholds": [35, 80, 120, 180],                         # Maturity
        # "labels": ["0-25", "25-40", "40-72", "72-120", "120+"],   # Maturity
        "thresholds": [0.2, 0.8],                                   # Stand-count
        "labels": ["0-2.4", "2.4-4.5", "4.5+"],                     # Stand-count
        "score_function": calculate_earth_tones_percentage,
        "x_limits_for_distribution_graph": None
    },
    "Brown_spectrum_dominance": {
        # "thresholds": [35, 80, 120, 180],                         # Maturity
        # "labels": ["0-25", "25-40", "40-72", "72-120", "120+"],   # Maturity
        "thresholds": [0.34, 0.8],                                  # Stand-count
        "labels": ["0-2.4", "2.4-4.5", "4.5+"],                     # Stand-count
        "score_function": calculate_brown_spectrum_detection,
        "x_limits_for_distribution_graph": None
    },
    "Red_spectrum_dominance": {
        # "thresholds": [35, 80, 120, 180],                         # Maturity
        # "labels": ["0-25", "25-40", "40-72", "72-120", "120+"],   # Maturity
        "thresholds": [0.1, 0.5],                                   # Stand-count
        "labels": ["0-2.4", "2.4-4.5", "4.5+"],                     # Stand-count
        "score_function": calculate_red_spectrum_dominance,
        "x_limits_for_distribution_graph": None
    }
}


def get_algorithmic_clustering_attributes():
    return ATTRIBUTES
