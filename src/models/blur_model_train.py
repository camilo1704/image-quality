from src.utils.files import list_imgs_files, mkdir_p
from src.data_processing.compute_blur import measure_functions, apply_all_measures
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
    save_model_path = os.path.join(save_artifacts_path, "blur_model.pkl")

    blur_imgs_files = list_imgs_files(os.path.join(imgs_path, "blur"))
    normal_imgs_files = list_imgs_files(os.path.join(imgs_path, "normal"))

    labels = [1] * len(blur_imgs_files) + [0] * len(normal_imgs_files)
    file_names = blur_imgs_files + normal_imgs_files
    df = prepare_dataframe(file_names, labels, classification_name="blur",
                           feature_function=apply_all_measures)

    feature_list = [item[0] for item in measure_functions]

    train_generic_classifier(df, feature_list, save_scaler_path, save_model_path,
                             classification_name="blur", target_names=["normal", "blur"])
