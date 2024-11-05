import cv2
import numpy as np


def calculate_dominant_color(image_path):
    """
    Calculate the dominant color of an image using the mean hue value.
    :param image_path: The path to the image file.
    :return: The mean hue value as the dominant color score.
    """
    image = cv2.imread(image_path)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue_channel = hsv_image[:, :, 0]
    dominant_color_score = np.mean(hue_channel)
    return dominant_color_score


def calculate_contrast(image_path):
    """
    Calculate the contrast of an image using the standard deviation of intensity.
    :param image_path: The path to the image file.
    :return: The contrast score of the image.
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    contrast_score = np.std(image)
    return contrast_score


def calculate_saturation(image_path):
    """
    Calculate the saturation of an image using the mean saturation in HSV.
    :param image_path: The path to the image file.
    :return: The saturation score of the image.
    """
    image = cv2.imread(image_path)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation_channel = hsv_image[:, :, 1]
    saturation_score = np.mean(saturation_channel)
    return saturation_score


def calculate_noise(image_path):
    """
    Calculate the noise of an image using the variance of the Laplacian.
    :param image_path: The path to the image file.
    :return: The noise score of the image.
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    noise_score = laplacian.var()
    return noise_score


def calculate_sharpness(image_path):
    """
    Calculate the sharpness of an image using the variance of gradient magnitude.
    :param image_path: The path to the image file.
    :return: The sharpness score of the image.
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = cv2.magnitude(sobel_x, sobel_y)
    sharpness_score = np.var(gradient_magnitude)
    return sharpness_score


def image_lightness(image_path):
    """
    Calculate the lightness of an image by Max RGB value. There are multiple ways of measuring brightness/lightness,
    this one was subjectively the most effective for maturity images
    :param image_path: The path to the image file.
    :return: The image lightness score.
    """
    img = cv2.imread(image_path)
    img_array = np.array(img, dtype=np.float32)
    nr_of_pixels = img_array.shape[0] * img_array.shape[1]
    pixel_sum = np.sum(np.max(img_array, axis=2))
    return pixel_sum / (255 * float(nr_of_pixels))


def calculate_blur_fft(image_path):
    """
    Calculate the blur score of an image using Fast Fourier Transform (FFT).
    :param image_path: The path to the image file.
    :return: The image blur score.
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    f_transform = np.fft.fft2(image)
    f_shift = np.fft.fftshift(f_transform)
    magnitude_spectrum = np.log(np.abs(f_shift) + 1)  # Avoid log(0)
    blur_score = np.sum(magnitude_spectrum) / (image.shape[0] * image.shape[1])
    return blur_score


def calculate_blur_sobel(image_path):
    """
    Calculate the blur score of an image using Sobel gradient magnitude.
    :param image_path: The path to the image file.
    :return: The image blur score.
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = cv2.magnitude(sobel_x, sobel_y)
    blur_score = np.mean(gradient_magnitude)
    return blur_score


def calculate_black_pixel_percentage(image_path):
    """
    Calculate the percentage of complete black pixels on the image, not the shadows of darker pixels,
    only complete blacks.
    :param image_path: The path to the image file.
    :return: The image complete blackness score.
    """
    image = cv2.imread(image_path)
    total_pixels = image.shape[0] * image.shape[1]
    black_pixels = (image == [0, 0, 0]).all(axis=2).sum()
    percentage = (black_pixels / total_pixels) * 100
    return percentage


def image_aspect_ratio(image_path):
    image = cv2.imread(image_path)
    return np.shape(image)[1] / np.shape(image)[0]