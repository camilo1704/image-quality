import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import joblib
import pandas as pd

from src.utils.models import load_model
from src.utils.files import list_imgs_files
from src.algorithms.blur import blur_kurtosis, calculate_gradient_energy, mten_focus_measure, mxml_focus_measure, variance_of_laplacian, calculate_msvd_blur, calculate_blurriness_entropy, calculate_blurriness_fft, calculate_blurriness_wavelet, calculate_contrast_operator,calculate_blurriness_steerable


#Functions to be computed measuring the bluriness of the image
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

#Features to be used in blur model
feature_list = ['gradient_energy', 'kurt', 'tef', "variance_of_laplacian", "blurriness_fft", "blurriness_wavelet", "contrast_operator", "blurriness_steerable"]

def apply_all_measures(df_images):
    """
    Apply bluriness measures to the images
    :param df_images: df with images path as rows
    :return df with bluriness feature columns added
    """
    for name, func in measure_functions:
        df_images[name] = df_images.img_name.apply(lambda row: func(row))
    return df_images


def compute_blur_from_model(images_path, blur_artifacts_path):
    """
    Apply trained blur model to images.
    :param images_path: images folder path
    :param blur_artifacts_path: path to blur model artifacts
    :returns: dict with image path as key and 1/0 for bluriness
    """

    list_images_path = list_imgs_files(images_path)
    df_images = pd.DataFrame(columns=["img_name"], data = list_images_path)
    df_images = apply_all_measures(df_images)

    scaler_model_path = os.path.join(blur_artifacts_path, "scaler.pkl")
    scaler = joblib.load(scaler_model_path)

    blur_model_path =  os.path.join(blur_artifacts_path, "blur_model.pkl")
    trained_blur_model = load_model(blur_model_path)

    df_x = df_images[feature_list]
    df_x = scaler.transform(df_x)
    blur_predict = trained_blur_model.predict(df_x)
    blur_result_dict = {image_path:{"blur":y_predict} for image_path, y_predict in zip(list_images_path, blur_predict)}
    
    return blur_result_dict
