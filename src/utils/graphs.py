import os
import cv2
from src.utils.files import save_graph_and_clear_memory
from random import shuffle
from matplotlib import pyplot as plt
import seaborn as sns


def grid_of_cluster_examples(clustering_method, score_or_features_extractor, column_lists, column_names, saving_dir,
                             number_of_clusters, max_columns_per_image=5):
    """
    Create a grid of images displaying columns of images for each cluster, split into multiple images if necessary.
    :param clustering_method: Name of the clustering method used.
    :param score_or_features_extractor: Name of the feature extraction/image attribute score method used.
    :param column_lists: Lists of images for each cluster.
        format: [["path_to_image1_cluster1.jpg", "path_to_image2_cluster1.jpg"],   # Images for Cluster 1
                 ["path_to_image1_cluster2.jpg", "path_to_image2_cluster2.jpg"],   # Images for Cluster 2
                 ["path_to_image1_cluster3.jpg", "path_to_image2_cluster3.jpg"]]   # Images for Cluster 3
    :param column_names: Names of clusters.
        format: ["Cluster 1", "Cluster 2", "Cluster 3"]
    :param saving_dir: Directory to save the graphs.
    :param number_of_clusters: Total number of clusters.
    :param max_columns_per_image: Maximum number of columns (clusters) per saved image.
    """
    # Split columns into chunks of max_columns_per_image
    num_parts = (number_of_clusters + max_columns_per_image - 1) // max_columns_per_image
    cluster_chunks = [column_lists[i * max_columns_per_image:(i + 1) * max_columns_per_image] for i in range(num_parts)]
    name_chunks = [column_names[i * max_columns_per_image:(i + 1) * max_columns_per_image] for i in range(num_parts)]

    for part, (columns, names) in enumerate(zip(cluster_chunks, name_chunks), start=1):
        # Shuffle images within each column
        for degree_id, degree in enumerate(columns):
            shuffle(columns[degree_id])

        num_rows = 40
        fig, axes = plt.subplots(num_rows, len(columns), figsize=(50, 80))

        for row in range(num_rows):
            for degree in range(len(columns)):
                if len(columns[degree]) > row:
                    img_path = columns[degree][row]
                else:
                    break
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                axes[row, degree].imshow(img)
                axes[row, degree].text(0.5, -0.15, os.path.basename(img_path), size=15, ha="center",
                                       transform=axes[row, degree].transAxes)
                axes[row, degree].axis('off')

                # Get the height and size of the image
                height, width, _ = img.shape
                size_text = f"Size: {width}x{height}"
                axes[row, degree].text(
                    0.5, 1.05, size_text, size=15, ha="center", transform=axes[row, degree].transAxes
                )

        for ax, col in zip(axes[0], names):
            ax.set_title(col.replace('_', ' '), fontsize=60, y=1.4, fontweight="bold")

        plt.subplots_adjust(wspace=0, hspace=0)
        plt.suptitle(f"{clustering_method} with {score_or_features_extractor} - Clusters {part} of {num_parts}",
                     fontsize=80, fontweight="bold", y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        # Save the plot with a descriptive filename, including the part number
        filename = f"{clustering_method.lower()}_{score_or_features_extractor.lower()}_{number_of_clusters}_clusters_partition_part_{part}.jpg"
        save_graph_and_clear_memory(saving_dir, filename)


def image_attribute_score_distribution(measurements_results, attribute_name, saving_dir,
                                       thresholds=None, labels=None, xlim=None):
    """
    Plots the distribution of attribute measurements with optional red vertical lines for thresholds.
    If xlim is provided, creates two plots: one with and one without the xlim limits.
    :param measurements_results: List of [image_path, measurement_value] pairs.
    :param attribute_name: Name of the attribute.
    :param saving_dir: Directory to save the plot.
    :param thresholds: List of threshold values to plot as vertical lines.
    :param labels: List of labels for each threshold.
    :param xlim: Optional x-axis limits.
    """
    # Extract measurement values from the results
    measurements = [entry[1] for entry in measurements_results]

    def create_plot(limited=False):
        plt.figure(figsize=(10, 6))
        sns.histplot(data=measurements, bins=15, kde=True, color='blue')

        # Add title and axis labels
        title = f"Distribution of {attribute_name}"
        if limited and xlim:
            title += f" - Zoom {xlim[0]} - {xlim[1]}"
        plt.title(title, fontsize=16)
        plt.xlabel(f"{attribute_name}", fontsize=14)
        plt.ylabel("Number of images", fontsize=14)

        # Add vertical lines for each threshold with labels outside the graph
        if thresholds and labels:
            for i, threshold in enumerate(thresholds):
                plt.axvline(x=threshold, color='red', linestyle='--', linewidth=1.5)
                # Place the label outside the plot, above the ylim and rotated 30 degrees
                plt.text(threshold, plt.ylim()[1] * 1.05, labels[i], color='red', ha='right',
                         va='bottom', fontsize=10, rotation=30, clip_on=False)

        # Apply x-axis limits if the plot is zoomed
        if limited and xlim:
            plt.xlim(xlim[0], xlim[1])

        # Determine file name
        filename = f"{attribute_name.lower()}_measurements_distribution"
        if limited and xlim:
            filename += f"_zoom_{xlim[0]}-{xlim[1]}"
        filename += ".jpg"

        # Save the plot and clear memory
        save_graph_and_clear_memory(saving_dir, filename)

    # Create both plots if xlim is provided
    create_plot(limited=False)
    if xlim:
        create_plot(limited=True)
