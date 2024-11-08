import cv2
import pywt
import numpy as np
from scipy.linalg import lstsq
from scipy.stats import entropy, kurtosis
from skimage.util import view_as_windows
from scipy.ndimage import convolve, gaussian_filter


from joblib import Parallel, delayed


def is_blurry(img_path, threshold=100):
    image = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
    variance = cv2.Laplacian(image,cv2.CV_64F).var()
    return variance


def blur_kurtosis(image_path):
    # Read the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Image not found or cannot be opened.")
    
    # Apply the Laplacian to get edges
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    
    # Flatten the Laplacian to a 1D array for kurtosis calculation
    laplacian_flat = laplacian.flatten()
    
    # Calculate and return kurtosis
    return kurtosis(laplacian_flat, fisher=True)


def calculate_gradient_energy(image_path):
    # Read the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Image not found or cannot be opened.")
    
    # Calculate gradients in x and y directions
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)  # Gradient in x-direction
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)  # Gradient in y-direction
    
    # Compute Gradient Energy as the sum of squares of gradients
    gradient_energy = np.sum(grad_x**2 + grad_y**2)
    
    return gradient_energy

def mten_focus_measure(image_path):
    # Read the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Image not found or cannot be opened.")
    
    # Compute the gradients in the x and y directions using the Sobel operator
    Sx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)  # Gradient in x-direction
    Sy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)  # Gradient in y-direction
    
    # Calculate MTEN by summing the squares of Sx and Sy gradients
    mten_value = np.sum(Sx**2 + Sy**2)
    
    return mten_value


def mxml_focus_measure(image_path):
    # Read the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Image not found or cannot be opened.")
    
    # Define Laplacian filters for x, y, and diagonal directions
    Lx = np.array([[1, -2, 1], [2, -4, 2], [1, -2, 1]])  # Horizontal mask
    Ly = np.array([[1, 2, 1], [-2, -4, -2], [1, 2, 1]])  # Vertical mask
    Ld1 = np.array([[2, -1, 0], [-1, -4, 1], [0, 1, 2]])  # Diagonal (\) mask
    Ld2 = np.array([[0, 1, 2], [1, -4, -1], [2, -1, 0]])  # Diagonal (/) mask
    
    # Convolve the image with each Laplacian filter
    Ix = cv2.filter2D(image, cv2.CV_64F, Lx)
    Iy = cv2.filter2D(image, cv2.CV_64F, Ly)
    Id1 = cv2.filter2D(image, cv2.CV_64F, Ld1)
    Id2 = cv2.filter2D(image, cv2.CV_64F, Ld2)
    
    # Calculate the MXML focus measure
    MXML_value = np.sum(np.abs(Ix) + np.abs(Iy) + np.abs(Id1) + np.abs(Id2))
    
    return MXML_value


def variance_of_laplacian(image_path):
    """
    Calculates the variance of an image's laplacian
    :param image_path: path to image file
    :returns mvl_value: variance of laplacian
    """
    # Read the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Image not found or cannot be opened.")
    
    # Apply Laplacian filter to the image
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    
    # Calculate the mean of the Laplacian
    mean_laplacian = np.mean(laplacian)
    
    # Calculate the variance of the Laplacian as MVL
    mvl_value = np.mean((laplacian - mean_laplacian) ** 2)
    
    return mvl_value


def calculate_msvd_blur(image_path, window_size=3, k=2):
    """
    Calculates the MSVD blur measure for an image.
    :param image_path: path to image file
    :param window_size: int, the size of the neighborhood window (e.g., 3 for a 3x3 window)
    :param k: int, the number of largest singular values to consider
    :return blur_map: np.array, blur measure for each pixel in the image
    """
    # Ensure image is grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Pad the image to handle border pixels
    padded_image = cv2.copyMakeBorder(image, window_size//2, window_size//2, 
                                      window_size//2, window_size//2, cv2.BORDER_REFLECT)
    
    blur_map = np.zeros(image.shape, dtype=np.float32)
    rows, cols = image.shape
    
    # Iterate through each pixel in the image
    for y in range(rows):
        for x in range(cols):
            # Extract the neighborhood window
            window = padded_image[y:y + window_size, x:x + window_size].astype(np.float32)
            
            # Perform Singular Value Decomposition on the window
            _, s, _ = np.linalg.svd(window)
            
            # Calculate MSVD(x, y)
            largest_k_sum = np.sum(s[:k])
            total_sum = np.sum(s)
            msvd = 1 - (largest_k_sum / total_sum)
            
            # Store the result in the blur map
            blur_map[y, x] = msvd
    
    return blur_map


def calculate_histogram_entropy(patch, num_bins=20):
    """
    Calculate the histogram entropy for a given patch.
    :param patch: np.array, the input image patch (grayscale).
    :param num_bins: int, the number of bins for the histogram (default is 256 for 8-bit images).
    :return entropy_value: float, the entropy of the histogram for the patch.
    """
    # Calculate the histogram of the patch
    hist, _ = np.histogram(patch, bins=num_bins, range=(0, 256), density=True)
    
    # Calculate entropy
    entropy_value = -np.sum(hist * np.log2(hist + 1e-10))  # Adding a small epsilon to avoid log(0)
    
    return entropy_value


def calculate_blurriness_entropy(image_path, window_size=5, num_bins=20):
    """
    Calculates a single blurriness score for an image based on histogram entropy.
    :param- image_path: path to image file
    :param window_size: int, size of the neighborhood window.
    :param num_bins: int, number of bins for the histogram.
    :return blurriness_score: float, the average entropy across all patches as a measure of blurriness.
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # Ensure image is grayscale
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Extract all patches of the specified window size
    patches = view_as_windows(image, (window_size, window_size))
    
    # Flatten patches to compute entropy on each patch
    entropy_values = [
        calculate_histogram_entropy(patch.flatten(), num_bins=num_bins)
        for row in patches
        for patch in row
    ]
    
    # Calculate the mean entropy across all patches as the blurriness score
    blurriness_score = np.mean(entropy_values)
    
    return blurriness_score


def calculate_blurriness_fft(image_path):
    """
    Calculates a blurriness score based on the Fast Fourier Transform (FFT) of the image.
    :param image_path: path to image file
    :return blurriness_score: float, a score where lower values indicate more blurriness.
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Ensure the image is grayscale
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply FFT and shift the zero frequency component to the center
    fft_image = np.fft.fft2(image)
    fft_shifted = np.fft.fftshift(fft_image)
    
    # Calculate the magnitude spectrum
    magnitude_spectrum = np.abs(fft_shifted)
    
    # Calculate a blurriness score based on the high frequencies
    # Summing up the frequencies in the high-frequency area only
    rows, cols = magnitude_spectrum.shape
    center_x, center_y = rows // 2, cols // 2
    high_freq_area = magnitude_spectrum[center_x-10:center_x+10, center_y-10:center_y+10]
    high_freq_sum = np.sum(high_freq_area)
    
    # Calculate the total magnitude and compute the ratio
    total_sum = np.sum(magnitude_spectrum)
    blurriness_score = 1 - (high_freq_sum / total_sum)
    
    return blurriness_score


def calculate_blurriness_wavelet(image_path, wavelet='db1'):
    """
    Calculates a blurriness score for an image based on the sum of the wavelet coefficients.
    :params image_path: path to image file
    :param wavelet: str, the type of wavelet to use (default is 'db1').
    :returns blurriness_score: float, the sum of the absolute wavelet coefficients in the detail sub-bands.
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # Ensure the image is grayscale
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Perform a single-level 2D Discrete Wavelet Transform (DWT)
    coeffs = pywt.dwt2(image, wavelet)
    # Extract the detail sub-bands (LH, HL, HH)
    _, (LH, HL, HH) = coeffs
    
    # Calculate the sum of absolute values in the detail sub-bands
    blurriness_score = np.sum(np.abs(LH)) + np.sum(np.abs(HL)) + np.sum(np.abs(HH))
    
    return blurriness_score


def calculate_contrast_operator(image_path, window_size=3):
    """
    Calculates a blurriness score based on Nanda et al.'s contrast operator.
    :params image_path: path to image file
    :param window_size: int, the size of the neighborhood window (default is 3x3).
    :return blurriness_score: float, the average contrast across all pixels as a measure of blurriness.
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Ensure image is grayscale
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Define the kernel for calculating contrast
    kernel = np.ones((window_size, window_size))
    kernel[window_size // 2, window_size // 2] = -(window_size * window_size - 1)
    
    # Calculate local contrast for each pixel
    local_contrast = convolve(image.astype(float), kernel, mode='reflect')
    local_contrast = np.abs(local_contrast)
    
    # Sum local contrast values within the window for each pixel
    blurriness_score = np.mean(local_contrast)
    
    return blurriness_score


def calculate_patch_curvature(window):
    """
    Calculate the curvature measure for a given patch.
    :param window: np.array, the input image patch.
    :returns curvature_measure: float, the curvature measure for the patch.
    """
    # Set up a system of equations for quadratic fitting
    window_size = window.shape[0]
    X, Y = np.meshgrid(range(window_size), range(window_size))
    X = X.flatten()
    Y = Y.flatten()
    Z = window.flatten()
    
    # Matrix A for solving Ax = b in least-squares sense
    A = np.column_stack((np.ones_like(X), X, Y, X**2, Y**2, X * Y))
    coeffs, _, _, _ = lstsq(A, Z)
    
    # Sum of the absolute values of the relevant coefficients (c1, c2, c3, c5)
    curvature_measure = np.abs(coeffs[1]) + np.abs(coeffs[2]) + np.abs(coeffs[3]) + np.abs(coeffs[5])
    return curvature_measure

def calculate_blurriness_curvature(image_path, window_size=5, stride=3, num_jobs=-1):
    """
    Calculates a blurriness score based on the curvature operator
    :params image_path :path to image file
    :param window_size: int, size of the neighborhood window.
    :param stride: int, the step size for sampling patches.
    :param num_jobs: int, number of parallel jobs (use -1 for all available CPUs).
    :return blurriness_score: float, the average curvature across sampled patches as a measure of blurriness.
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # Ensure the image is grayscale
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Extract all patches of the specified window size, with specified stride
    patches = view_as_windows(image, (window_size, window_size), step=stride)
    patches = patches.reshape(-1, window_size, window_size)
    
    # Parallel processing for each patch
    curvature_measures = Parallel(n_jobs=num_jobs)(
        delayed(calculate_patch_curvature)(patch) for patch in patches
    )
    
    # Calculate the mean curvature measure as the final blurriness score
    blurriness_score = np.mean(curvature_measures)
    
    return blurriness_score


def steerable_filter_responses(image, sigma=1, num_orientations=4):
    """
    Apply steerable filters to an image and get the maximum response across orientations.
    :param image: np.array, the input grayscale image.
    :param sigma: float, the standard deviation for the Gaussian kernel.
    :param num_orientations: int, the number of orientations for the filters.
    :returns max_response: np.array, the maximum filter response for each pixel.
    """
    # Define orientations
    angles = np.linspace(0, np.pi, num_orientations, endpoint=False)
    responses = []

    # Apply filters at each orientation
    for theta in angles:
        # Compute Gaussian derivatives in x and y directions
        dx = gaussian_filter(image, sigma=sigma, order=[0, 1])
        dy = gaussian_filter(image, sigma=sigma, order=[1, 0])
        
        # Rotate derivatives to the orientation theta
        response = np.abs(dx * np.cos(theta) + dy * np.sin(theta))
        responses.append(response)

    # Stack responses and take the maximum across orientations
    max_response = np.max(np.stack(responses, axis=-1), axis=-1)
    return max_response


def calculate_blurriness_steerable(image_path, sigma=1, num_orientations=4):
    """
    Calculates a blurriness score for an image using steerable filters.
    :param image_path: path to image
    :param sigma: float, the standard deviation for the Gaussian kernel.
    :param num_orientations: int, the number of orientations for the filters.
    :returns blurriness_score: float, the sum of the maximum responses as a measure of blurriness.
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # Ensure the image is grayscale
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Get maximum response from steerable filters
    max_response = steerable_filter_responses(image, sigma=sigma, num_orientations=num_orientations)
    
    # Sum the maximum responses to obtain the blurriness score
    blurriness_score = np.sum(max_response)
    
    return blurriness_score
