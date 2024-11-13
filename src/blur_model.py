"""
Test the blur model with a directory of images.
Generate a grid with examples of the classifications, a dictionary with the model results
 and optionally folders with all the images in different folders for each class
"""
from src.data_processing.compute_blur import compute_blur_from_model
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
    parser.add_argument('--blur_artifacts_path', type=str,
                        help='Path to save graphs and exploratory analysis of the whole dataset')
    parser.add_argument('--output_path', type=str,
                        help='Path to save graphs and exploratory analysis of the whole dataset')
    parser.add_argument('--save_images', action='store_true', default=False,
                        help='Copy images on 2 different folders: blur/not_blur')

    args = parser.parse_args()
    image_paths = get_image_paths(args.images_path)
    blur_artifacts_path = args.blur_artifacts_path
    output_path = args.output_path
    save_images = args.save_images

    images_results_path = os.path.join(os.path.join(output_path, "images_results"))
    mkdir_p(images_results_path)

    blur_dict = compute_blur_from_model(image_paths, blur_artifacts_path)

    with open(os.path.join(output_path, "blur_dict.json"), "w") as json_file:
        json.dump(blur_dict, json_file, indent=4, default=convert_to_serializable)

    organize_images_by_classification(image_dict=blur_dict, output_folder=images_results_path,
                                      property_name='blur', class_names=["not_blur", "blur"])

    column_lists, column_names = convert_image_dict_to_columns(image_dict=blur_dict,
                                                               property_name='blur', class_names=["not_blur", "blur"])

    grid_of_cluster_examples("blur_model", 'blur',
                             column_lists, column_names, saving_dir=output_path, number_of_clusters=2)
