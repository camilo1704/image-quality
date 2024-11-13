from data_processing.compute_blur import compute_blur_from_model
from src.utils.files import mkdir_p, get_image_paths, convert_to_serializable
from src.algorithms.algorithmic_clustering import image_attribute_scores
from src.algorithms.algorithmic_clustering_config import get_algorithmic_clustering_attributes
import os
import sys
import json
import argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images_path', type=str,
                        help='Path with all images')
    parser.add_argument('--output_path', type=str,
                        help='Path to save graphs and exploratory analysis of the whole dataset')
    parser.add_argument('--blur_artifacts_path', type=str,
                        help='Path to save graphs and exploratory analysis of the whole dataset')

    args = parser.parse_args()
    image_paths = get_image_paths(args.images_path)
    blur_artifacts_path = args.blur_artifacts_path
    output_path = args.output_path

    score_results_path = os.path.join(os.path.join(output_path, "score_results"))
    mkdir_p(score_results_path)

    result_dict = compute_blur_from_model(image_paths, blur_artifacts_path)
    print(result_dict)

    attribute_degrees = {}
    ATTRIBUTES = get_algorithmic_clustering_attributes()
    for an_name, attribute_data in ATTRIBUTES.items():
        attribute_degrees[an_name] = image_attribute_scores(image_paths, an_name, attribute_data["labels"],
                                                            attribute_data["thresholds"],
                                                            attribute_data["score_function"], output_path,
                                                            attribute_data["x_limits_for_distribution_graph"]
                                                            )

    brightness_results = json.load(open(os.path.join(score_results_path, "brightness_results.json")))
    blackness_results = json.load(open(os.path.join(score_results_path, "blackness_results.json")))
    aspect_ratio_results = json.load(open(os.path.join(score_results_path, "aspect_ratio_results.json")))

    for img_arr in brightness_results:
        img_path, value = img_arr
        result_dict[img_path]["dark"] = 1 if value < 0.25 else 0
        result_dict[img_path]["brigth"] = 1 if value > 0.77 else 0

    for img_arr in blackness_results:
        img_path, value = img_arr
        result_dict[img_path]["black"] = 1 if value > 20 else 0

    for img_arr in aspect_ratio_results:
        img_path, value = img_arr
        result_dict[img_path]["long"] = 1 if value > 3.45 else 0

    with open(os.path.join(score_results_path, "results_dict.json"), "w") as json_file:
        json.dump(result_dict, json_file, indent=4, default=convert_to_serializable)
