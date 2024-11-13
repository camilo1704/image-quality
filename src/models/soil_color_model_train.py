from src.utils.files import list_imgs_files, mkdir_p
from src.data_processing.compute_soil_color import measure_functions, apply_all_soil_color_measures
from src.models.training_functions import prepare_dataframe, train_generic_classifier
import os
import sys
import argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images_path', type=str,
                        help='Path with all images')
    parser.add_argument('--save_artifacts_path', type=str,
                        help='Path to save model artifacts')

    args = parser.parse_args()
    imgs_path = args.images_path
    save_artifacts_path = args.save_artifacts_path

    mkdir_p(save_artifacts_path)
    save_scaler_path = os.path.join(save_artifacts_path, "scaler.pkl")
    save_model_path = os.path.join(save_artifacts_path, "soil_color_model.pkl")

    # Load images from three classes
    red_soil_files = list_imgs_files(os.path.join(imgs_path, "red_soil"))
    dark_soil_files = list_imgs_files(os.path.join(imgs_path, "dark_soil"))
    not_red_dark_soil_files = list_imgs_files(os.path.join(imgs_path, "not_red_dark_soil"))

    # Assign labels: 0 for red_soil, 1 for dark_soil, 2 for not_red_dark_soil
    labels = ([0] * len(red_soil_files) +
              [1] * len(dark_soil_files) +
              [2] * len(not_red_dark_soil_files))
    file_names = red_soil_files + dark_soil_files + not_red_dark_soil_files
    df = prepare_dataframe(file_names, labels, classification_name="soil_color",
                           feature_function=apply_all_soil_color_measures)

    feature_list = [item[0] for item in measure_functions]

    train_generic_classifier(df, feature_list, save_scaler_path, save_model_path, classification_name="soil_color",
                             target_names=["red_soil", "dark_soil", "not_red_dark_soil"])
