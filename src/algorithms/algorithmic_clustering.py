"""
This command takes a directory path with images, gives a score for each attribute for each image and generates the
distribution graph of the scores of each attribute. According to thresholds for the scores it clusters the images
in groups and creates a grid of examples.

--images_path: Path with all images
--output_path: Path to save graphs

Output:
    Score files from each model
    Grid of examples of each cluster

To add a new method, add it to utils.image_attributes_score_algorithms and use it in algorithmic_clustering_config
"""

import json
import os
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.utils.files import mkdir_p, get_image_paths
from src.utils.graphs import grid_of_cluster_examples, image_attribute_score_distribution
from src.algorithms.algorithmic_clustering_config import get_algorithmic_clustering_attributes


def categorize_images_with_thresholds(attribute_results_float32, thresholds, labels):
    """
    Categorize images based on specified thresholds.
    :param attribute_results_float32: List of image attribute measurements. [(image_path, attribute_measurement),...]
    :param thresholds: Threshold values for categorization. Eg: [5.5, 6.1]
    :param labels: List of labels for each category. ["very_blurred_images", "blurred_images", "normal_images"]
    :return: List of image_paths ordered corresponding to each label
    """
    categories = [[] for _ in range(len(thresholds) + 1)]

    for image_value, attribute_score in attribute_results_float32:
        for i, thresh in enumerate(thresholds):
            if attribute_score <= thresh:
                categories[i].append(image_value)
                break
        else:
            categories[-1].append(image_value)

    for label, category in zip(labels, categories):
        print(f"{label}\t-> {len(category)}")

    print("-----------------------------------")

    return categories


def image_attribute_scores(dataset_image_paths, attribute_name, threshold_labels,
                     threshold_values, measure_function, output_path, x_limits_for_distribution_graph=None):
    """
    Process and categorize attribute images. Create a distribution graph of the attribute
    :param attribute_name: Name of the attribute.
    :param threshold_labels: List of labels for threshold levels.
    :param threshold_values: List of threshold values.
    :param measure_function: Function to measure attributes.
    :param output_path: Path for saving results and visualizations.
    :return: List of categorized images.
    """
    results = []
    file_path = os.path.join(output_path, "score_results", f"{attribute_name.lower().replace(' ', '_')}_results.json")
    print("attribute  ", attribute_name)
    if not os.path.exists(file_path):
        for im_path in dataset_image_paths:
            try:
                result = [im_path, measure_function(im_path)]
                print(im_path, measure_function(im_path))
                results.append(result)
            except Exception:
                print(f"Exception on path {im_path}")

        results_float32 = [[item[0], float(item[1])] for item in results]

        with open(file_path, "w") as json_file:
            json.dump(results_float32, json_file)
    else:
        with open(file_path, "r") as json_file:
            results_float32 = json.load(json_file)

    degrees = categorize_images_with_thresholds(attribute_results_float32=results_float32,
                                                thresholds=threshold_values, labels=threshold_labels)

    image_attribute_score_distribution(results_float32, attribute_name=attribute_name, saving_dir=output_path,
                                     thresholds=threshold_values, labels=threshold_labels,
                                     xlim=x_limits_for_distribution_graph)

    return degrees


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images_path', type=str,
                        help='Path with all images')
    parser.add_argument('--output_path', type=str,
                        help='Path to save graphs and exploratory analysis of the whole dataset')

    args = parser.parse_args()
    output_path = args.output_path
    image_paths = get_image_paths(args.images_path)

    mkdir_p(os.path.join(output_path, "score_results"))
    ATTRIBUTES = get_algorithmic_clustering_attributes()
    attribute_degrees = {}
    for an_name, attribute_data in ATTRIBUTES.items():
        attribute_degrees[an_name] = image_attribute_scores(image_paths, an_name, attribute_data["labels"],
                                                            attribute_data["thresholds"],
                                                            attribute_data["score_function"], output_path,
                                                            attribute_data["x_limits_for_distribution_graph"]
                                                            )
        grid_of_cluster_examples("threshold_clustering", an_name, attribute_degrees[an_name], attribute_data["labels"],
                                 saving_dir=output_path, number_of_clusters=len(attribute_degrees[an_name]))
