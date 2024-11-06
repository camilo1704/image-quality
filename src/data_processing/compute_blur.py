import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.algorithms.blur import is_blurry, blur_kurtosis, calculate_gradient_energy, mten_focus_measure, mxml_focus_measure, variance_of_laplacian, calculate_msvd_blur, calculate_blurriness_entropy, calculate_blurriness_fft, calculate_blurriness_wavelet, calculate_contrast_operator,calculate_blurriness_steerable
   
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
imgs_path = ""

def calculate_blurriness(row, measure_function):
    img_path = os.path.join(imgs_path, row)
    return measure_function(img_path)

def apply_all_measures(df_images):

    for name, func in measure_functions:
        df_images[name] = df_images.img_name.apply(lambda row: calculate_blurriness(row, func))
    return df_images