from src.algorithms.blur import (blur_kurtosis, calculate_gradient_energy, mten_focus_measure, variance_of_laplacian,
                                 calculate_blurriness_fft, calculate_blurriness_wavelet, calculate_contrast_operator,
                                 calculate_blurriness_steerable)
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Functions to be computed measuring the blurriness of the image
measure_functions = [
    ("kurt", blur_kurtosis),
    ("tef", mten_focus_measure),
    ("blurriness_fft", calculate_blurriness_fft),
    ("gradient_energy", calculate_gradient_energy),
    ("variance_of_laplacian", variance_of_laplacian),
    ("contrast_operator", calculate_contrast_operator),
    ("blurriness_wavelet", calculate_blurriness_wavelet),
    ("blurriness_steerable", calculate_blurriness_steerable)
]


def apply_all_blur_measures(df_images):
    """
    Apply attributes measures to the images
    :param df_images: df with images path as rows
    :return df with attributes features columns added
    """
    for name, func in measure_functions:
        df_images[name] = df_images.file_name.apply(lambda row: func(row))
    return df_images
