import os

blur_imgs_path = "/home/cuchu/documents/eiwa/img_char/data/super_blur"
normal_imgs_path = "/home/cuchu/documents/eiwa/img_char/data/normal"


def list_blur_images():
    blur_imgs_files = os.listdir(blur_imgs_path)
    blur_imgs_files = [os.path.join(blur_imgs_path, x) for x in blur_imgs_files]

    return blur_imgs_files

def list_normal_imgs():

    normal_imgs_files = os.listdir(normal_imgs_path)
    normal_imgs_files = [os.path.join(normal_imgs_path, x) for x in normal_imgs_files]
    return normal_imgs_files
