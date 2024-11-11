import os
import errno
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
