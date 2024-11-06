import pandas as pd
from utils.files import list_blur_images, list_normal_imgs
from data_processing.compute_blur import apply_all_measures
from models.blur_model import blur_classifier



blur_imgs_files = list_blur_images()
normal_imgs_files = list_normal_imgs()


labels = [1]*len(blur_imgs_files) + [0]*len(normal_imgs_files)
file_names = blur_imgs_files + normal_imgs_files


def df_blur(file_names, labels):
    data = [[x,y] for x,y in zip(file_names, labels)]
    df_images = pd.DataFrame(columns=["img_name","blur"], data = data)
    df_images = apply_all_measures(df_images)
    return df_images

df = df_blur(file_names, labels)

feature_list = ['gradient_energy', 'kurt', 'tef', "variance_of_laplacian", "blurriness_fft", "blurriness_wavelet", "contrast_operator", "blurriness_steerable"]
blur_classifier(df, feature_list)
