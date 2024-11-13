from src.utils.models import load_model
import os
import sys
import joblib
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


def compute_from_model(list_images_path, model_artifacts_path, model_classes_name,
                       feature_list, apply_measures_function):
    """
    Apply trained model to images.
    :param list_images_path: list of images paths
    :param model_artifacts_path: path to model artifacts. Dir includes scaler.pkl and name_model.pkl
    :param model_classes_name: name of the class property name. Example 'blur', 'soil_color'
    :param feature_list: list of the names of the features being analysed, derived from the measure_functions.
        Example ['gradient_energy', 'kurt', "blurriness_fft"...]
    :param apply_measures_function: function to apply measuring of scores on the function. Defined on compute_...py
    :returns: dict with image path as key and 0/1/2... values for the class predicted
    """

    df_images = pd.DataFrame(columns=["file_name"], data=list_images_path)
    df_images = apply_measures_function(df_images)

    scaler_model_path = os.path.join(model_artifacts_path, "scaler.pkl")
    scaler = joblib.load(scaler_model_path)

    model_path = os.path.join(model_artifacts_path, f"{model_classes_name}_model.pkl")
    trained_model = load_model(model_path)

    df_x = df_images[feature_list]
    df_x = scaler.transform(df_x)
    class_predict = trained_model.predict(df_x)
    result_dict = {image_path: {model_classes_name: y_predict} for image_path, y_predict in
                   zip(list_images_path, class_predict)}

    return result_dict
