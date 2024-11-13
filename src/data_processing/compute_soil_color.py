from src.algorithms.image_attributes_score_algorithms import (calculate_dominant_color, calculate_saturation,
                                                              calculate_colorfulness, calculate_red_spectrum_dominance,
                                                              calculate_brown_spectrum_detection,
                                                              calculate_earth_tones_percentage)
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Functions to be computed measuring the color features of the image
measure_functions = [
    ("dominant_color", calculate_dominant_color),
    ("saturation", calculate_saturation),
    ("colorfulness", calculate_colorfulness),
    ("red_spectrum", calculate_red_spectrum_dominance),
    ("brown_spectrum", calculate_brown_spectrum_detection),
    ("earth_tones", calculate_earth_tones_percentage),
]


def apply_all_soil_color_measures(df_images):
    """
    Apply attributes measures to the images
    :param df_images: df with images path as rows
    :return df with attributes features columns added
    """
    for name, func in measure_functions:
        df_images[name] = df_images.file_name.apply(lambda row: func(row))
    return df_images
