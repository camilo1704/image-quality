"""
This command takes a directory path with images and generates a clustering according to
different feature extractor models. The models come from utils.feature_extractors and are specified on config

--images_path: Path with all images
--number_of_clusters: 5
--output_path: Path to save graphs

Output:
    Features files from each model
    Grid of examples of each cluster

To add a new feature extractor model, add it to utils.feature_extractors and use it in deep_learning_clustering_config
"""
import os
import argparse
from deep_learning_clustering_config import CLUSTERING_METHOD, PREPROCESS, FEATURE_EXTRACTORS
from models.feature_extractors import generic_clustering
from src.utils.files import mkdir_p
from src.utils.graphs import grid_of_cluster_examples


def prepare_column_lists_for_clusters(image_clusters, n_clusters):
    """
    Organize image paths by clusters for visualization.
    :param image_clusters: Dictionary mapping image paths to cluster labels.
    :param n_clusters: Number of clusters.
    :return: List of lists, where each sublist contains image paths for a cluster.
    """
    column_lists = [[] for _ in range(n_clusters)]
    for img_path, label in image_clusters.items():
        column_lists[label].append(img_path)
    return column_lists


def run_clustering_and_plot(image_paths, features_files_output_path, output_path, preprocess,
                            feature_extractor, clustering_method, extract_features_function, n_clusters=5):
    """
    Runs clustering and plots the results in a grid.
    :param image_paths: List of paths to image files.
    :param features_files_output_path: Directory to save/load feature files.
    :param output_path: Directory to save the grid plot.
    :param feature_extractor: Name of the feature extraction method.
    :param clustering_method: Clustering method (e.g., 'kmeans').
    :param extract_features_function: Function used to extract features from images.
    :param n_clusters: Number of clusters.
    """
    # Run clustering
    model, image_clusters = generic_clustering(
        image_paths,
        preprocess,
        features_output_path=features_files_output_path,
        extract_features_function=extract_features_function,
        extraction_method=feature_extractor,
        n_clusters=n_clusters,
        clustering_method=clustering_method
    )

    # Prepare lists for plotting
    column_lists = prepare_column_lists_for_clusters(image_clusters, n_clusters)
    column_names = [f"Cluster {i}" for i in range(n_clusters)]

    # Plot and save the grid of images
    grid_of_cluster_examples(clustering_method, feature_extractor,
                                column_lists, column_names, output_path, n_clusters)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images_path', type=str,
                        help='Path with all images')
    parser.add_argument('--number_of_clusters', type=int, default=5,
                        help='Path with all images')
    parser.add_argument('--output_path', type=str,
                        help='Path to save graphs and exploratory analysis of the whole dataset')

    args = parser.parse_args()
    output_path = args.output_path
    n_clusters = args.number_of_clusters  # Define the number of clusters

    images_path = args.images_path
    image_paths = [os.path.join(images_path, f) for f in os.listdir(images_path) if f.endswith(('.jpg', '.png', '.jpeg'))]

    features_files_output_path = os.path.join(output_path, "features")
    mkdir_p(features_files_output_path)

    for feature_extractor, extractor_function in FEATURE_EXTRACTORS:
        run_clustering_and_plot(
            image_paths=image_paths,
            features_files_output_path=features_files_output_path,
            output_path=output_path,
            feature_extractor=feature_extractor,
            preprocess=PREPROCESS,
            clustering_method=CLUSTERING_METHOD,
            extract_features_function=extractor_function,
            n_clusters=n_clusters
        )
