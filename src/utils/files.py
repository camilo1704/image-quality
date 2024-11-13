import os
import errno
import shutil
import numpy as np
from matplotlib import pyplot as plt


def list_imgs_files(imgs_path):
    imgs_files = os.listdir(imgs_path)
    imgs_files = [os.path.join(imgs_path, x) for x in imgs_files]

    return imgs_files


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise exc


def get_image_paths(directory):
    extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
    image_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            if any(file.lower().endswith(ext) for ext in extensions):
                image_paths.append(os.path.join(root, file))
    return image_paths

    # Convert the data to JSON serializable types


def convert_to_serializable(obj):
    if isinstance(obj, np.integer):  # Handle NumPy integers
        return int(obj)
    elif isinstance(obj, np.floating):  # Handle NumPy floats
        return float(obj)
    elif isinstance(obj, np.ndarray):  # Handle NumPy arrays
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def save_graph_and_clear_memory(save_path, save_file_name):
    mkdir_p(save_path)
    plt.savefig(os.path.join(save_path, save_file_name))
    plt.cla()
    plt.close()


def convert_image_dict_to_columns(image_dict, property_name, class_names):
    """
    Converts an image_dict to column_lists and column_names format for clustering visualization.

    Args:
        image_dict (dict): Dictionary where keys are image paths and values are dictionaries with property data.
        property_name (str): Name of the property used for clustering.
        class_names (list): List of class names corresponding to the property values.

    Returns:
        tuple: (column_lists, column_names)
            column_lists: List of lists of image paths, grouped by clusters.
            column_names: List of cluster names.
    """
    # Initialize a dictionary to group images by property value
    grouped_images = {class_name: [] for class_name in class_names}

    # Group image paths by the property value
    for image_path, properties in image_dict.items():
        property_value = properties.get(property_name)
        if property_value is not None and property_value < len(class_names):
            grouped_images[class_names[property_value]].append(image_path)
        else:
            print(f"Skipping {image_path}: unexpected or missing property value {property_value}")

    # Prepare column_lists and column_names
    column_lists = list(grouped_images.values())
    column_names = list(grouped_images.keys())

    return column_lists, column_names


def organize_images_by_classification(image_dict, output_folder, property_name, class_names):
    """
    Organizes images into folders based on a specified class value in the dictionary.

    Args:
        image_dict (dict): Dictionary where keys are image paths and values are dictionaries with property data
            format: {'image_path_1': {'class_name': 0}, 'image_path_2': {'class_name': 1},...}
        output_folder (str): Path to the output folder where subfolders will be created for each class.
        property_name (str): Name of the property to use for classification (e.g., 'blur').
        class_names (list): List of class names corresponding to the property values.

        usage example: organize_images_by_property(image_dict, output_folder, "blur", ["not_blur", "blur"])
    """
    # Get the unique property values from the dictionary
    property_values = {properties.get(property_name) for properties in image_dict.values()}

    # Check if the number of class names matches the number of unique property values
    if len(property_values) != len(class_names):
        raise ValueError(
            f"The number of class names ({len(class_names)}) does not match the number of unique property values ({len(property_values)})."
        )

    # Map property values to class names
    value_to_class = {value: name for value, name in zip(sorted(property_values), class_names)}

    # Create the output directories if they don't exist
    for class_name in class_names:
        os.makedirs(os.path.join(output_folder, class_name), exist_ok=True)

    # Organize images into the corresponding folders
    for image_path, properties in image_dict.items():
        property_value = properties.get(property_name)
        if property_value in value_to_class:
            dest_folder = os.path.join(output_folder, value_to_class[property_value])
        else:
            print(f"Skipping {image_path}: unexpected property value {property_value}")
            continue

        # Copy the image to the appropriate folder
        try:
            shutil.copy(image_path, dest_folder)
        except FileNotFoundError:
            print(f"File not found: {image_path}")
        except Exception as e:
            print(f"Error copying {image_path}: {e}")
