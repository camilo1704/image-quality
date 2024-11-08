import os
import errno


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