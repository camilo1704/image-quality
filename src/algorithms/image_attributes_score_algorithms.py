import cv2
import numpy as np
from skimage.measure import shannon_entropy


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


def calculate_hue_variability(image_path):
    image = cv2.imread(image_path)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue_channel = hsv_image[:, :, 0]
    hue_variability = np.std(hue_channel)
    return hue_variability


def calculate_colorfulness(image_path):
    image = cv2.imread(image_path)
    (B, G, R) = cv2.split(image.astype("float"))
    rg = np.abs(R - G)
    yb = np.abs(0.5 * (R + G) - B)
    std_rg = np.std(rg)
    std_yb = np.std(yb)
    mean_rg = np.mean(rg)
    mean_yb = np.mean(yb)
    colorfulness = np.sqrt(std_rg ** 2 + std_yb ** 2) + 0.3 * np.sqrt(mean_rg ** 2 + mean_yb ** 2)
    return colorfulness


def calculate_color_entropy(image_path):
    image = cv2.imread(image_path)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist, _ = np.histogram(hsv_image[:, :, 0], bins=180, range=[0, 180])
    color_entropy = shannon_entropy(hist)
    return color_entropy


def calculate_earth_tones_percentage(image_path):
    """
    This metric calculates the proportion of earth tones (e.g., browns, tans, and greens)
    in an image. Earth tones typically occupy specific ranges in the HSV color space.
    """
    image = cv2.imread(image_path)
    # Convert to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define HSV ranges for earth tones
    lower_brown = np.array([10, 20, 20])
    upper_brown = np.array([30, 255, 200])
    lower_green = np.array([30, 20, 20])
    upper_green = np.array([85, 255, 200])

    # Create masks for brown and green earth tones
    brown_mask = cv2.inRange(hsv_image, lower_brown, upper_brown)
    green_mask = cv2.inRange(hsv_image, lower_green, upper_green)
    earth_mask = cv2.bitwise_or(brown_mask, green_mask)

    # Calculate percentage of earth tones
    earth_tones_ratio = np.sum(earth_mask > 0) / (hsv_image.shape[0] * hsv_image.shape[1])
    return earth_tones_ratio


def calculate_brown_spectrum_detection(image_path):
    """
    This metric measures the proportion of brown tones in the image, often used to identify earth-like or natural colors.
    """
    image = cv2.imread(image_path)
    # Convert to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define HSV range for brown tones
    lower_brown = np.array([10, 20, 20])
    upper_brown = np.array([30, 255, 200])

    # Create mask for brown tones
    brown_mask = cv2.inRange(hsv_image, lower_brown, upper_brown)

    # Calculate percentage of brown tones
    brown_ratio = np.sum(brown_mask > 0) / (hsv_image.shape[0] * hsv_image.shape[1])
    return brown_ratio


def calculate_red_spectrum_dominance(image_path):
    """
    This metric measures the dominance of red hues in the image. It can be useful for identifying red tones
    related to vegetation stress or warm tones in artistic photography.
    """
    image = cv2.imread(image_path)
    # Convert to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define HSV range for red tones (both ends of the hue spectrum)
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 50, 50])
    upper_red2 = np.array([180, 255, 255])

    # Create mask for red tones
    red_mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
    red_mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)

    # Calculate percentage of red tones
    red_ratio = np.sum(red_mask > 0) / (hsv_image.shape[0] * hsv_image.shape[1])
    return red_ratio
