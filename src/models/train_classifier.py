"""
Train a classifier, binary or multiclass
    Necessary:
        Directory with representative images to train the classifier (images_path)
        Add attribute measuring functions or feature extractors to data_processing module in format of example modules:
            src.data_processing.compute_soil_color and src.data_processing.compute_blur
            must have measure_functions list and apply_all_measures function

Usage:
    python3 generic_classifier.py --images_path /path/to/images --save_artifacts_path /path/to/artifacts
                                  --classification_name blur --feature_module src.data_processing.compute_blur
                                  --classes "blur:1 normal:0"
"""
from src.utils.files import list_imgs_files, mkdir_p
from src.models.training_functions import prepare_dataframe, train_generic_classifier
import os
import sys
import argparse
import importlib
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


def import_measure_functions(module_path, function_name):
    module = importlib.import_module(module_path)
    return getattr(module, function_name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generic script for training classifiers.")
    parser.add_argument('--images_path', type=str,
                        help='Path with all images', required=True)
    parser.add_argument('--save_artifacts_path', type=str,
                        help='Path to save model artifacts', required=True)
    parser.add_argument('--classification_name', type=str,
                        help='Name of the classification task. Example: blur, soil_color', required=True)
    parser.add_argument('--feature_module', type=str, required=True,
                        help='Module path for feature computation. '
                             'Example: src.data_processing.compute_soil_color')
    parser.add_argument('--classes', nargs='+', type=str,
                        help='List of classes (e.g., "red_soil:0 dark_soil:1 not_red_dark_soil:2")', required=True)

    args = parser.parse_args()

    imgs_path = args.images_path
    save_artifacts_path = args.save_artifacts_path
    classification_name = args.classification_name

    # Dynamically import feature functions and measures
    apply_features = import_measure_functions(args.feature_module, "apply_all_measures")
    measure_functions = import_measure_functions(args.feature_module, "measure_functions")

    # Parse class labels and directories
    class_mapping = {item.split(':')[0]: int(item.split(':')[1]) for item in args.classes}

    mkdir_p(save_artifacts_path)
    save_scaler_path = os.path.join(save_artifacts_path, "scaler.pkl")
    save_model_path = os.path.join(save_artifacts_path, f"{classification_name}_model.pkl")

    # Prepare data
    file_names = []
    labels = []
    for class_name, label in class_mapping.items():
        class_files = list_imgs_files(os.path.join(imgs_path, class_name))
        file_names.extend(class_files)
        labels.extend([label] * len(class_files))

    df = prepare_dataframe(file_names, labels, classification_name=classification_name,
                           feature_function=apply_features)

    feature_list = [item[0] for item in measure_functions]

    train_generic_classifier(df, feature_list, save_scaler_path, save_model_path,
                             classification_name=classification_name,
                             target_names=list(class_mapping.keys()))
