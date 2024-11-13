"""
Test the soil color model with a directory of images.
Generate a grid with examples of the classifications, a dictionary with the model results
 and optionally folders with all the images in different folders for each class
"""
from src.data_processing.classification_processing import compute_from_model
from src.data_processing.compute_soil_color import measure_functions, apply_all_measures
from src.utils.graphs import grid_of_cluster_examples
from src.utils.files import (mkdir_p, get_image_paths, convert_to_serializable,
                             organize_images_by_classification, convert_image_dict_to_columns)
import os
import sys
import json
import argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images_path', type=str,
                        help='Path with all images')
    parser.add_argument('--soil_color_artifacts_path', type=str,
                        help='Path to save graphs and exploratory analysis of the whole dataset')
    parser.add_argument('--output_path', type=str,
                        help='Path to save graphs and exploratory analysis of the whole dataset')
    parser.add_argument('--save_images', action='store_true', default=False,
                        help='Copy images on 2 different folders: red_soil/dark_soil/not_red_dark_soil')

    args = parser.parse_args()
    image_paths = get_image_paths(args.images_path)
    soil_color_artifacts_path = args.soil_color_artifacts_path
    output_path = args.output_path
    save_images = args.save_images
    images_results_path = os.path.join(os.path.join(output_path, "images_results"))
    mkdir_p(images_results_path)

    # Compute results
    feature_list = [item[0] for item in measure_functions]
    soil_color_dict = compute_from_model(image_paths, soil_color_artifacts_path, model_classes_name="soil_color",
                                   feature_list=feature_list, apply_measures_function=apply_all_measures)

    # Save results dict
    with open(os.path.join(output_path, "soil_color_dict.json"), "w") as json_file:
        json.dump(soil_color_dict, json_file, indent=4, default=convert_to_serializable)

    # Grid of class examples
    organize_images_by_classification(image_dict=soil_color_dict, output_folder=images_results_path,
                                      property_name='soil_color', class_names=["red_soil", "dark_soil",
                                                                               "not_red_dark_soil"])

    column_lists, column_names = convert_image_dict_to_columns(image_dict=soil_color_dict,
                                                               property_name='soil_color',
                                                               class_names=["red_soil", "dark_soil",
                                                                            "not_red_dark_soil"])

    # Save copies of images in class directories
    grid_of_cluster_examples("soil_color_model", 'soil_color',
                             column_lists, column_names, saving_dir=output_path, number_of_clusters=2)
