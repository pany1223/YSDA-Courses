import numpy as np
from scipy.fft import fft2, ifft2


def gaussian_kernel(size, sigma):
    """
    Построение ядра фильтра Гаусса.

    @param  size  int    размер фильтра (нечетный)
    @param  sigma float  параметр размытия
    @return numpy array  фильтр Гаусса размером size x size
    """
    center = size // 2
    x_grid, y_grid = np.mgrid[:size, :size]
    dists = np.sqrt((x_grid - center)**2 + (y_grid - center)**2)
    gaussian_filter = 1 / (2 * np.pi * sigma**2) * np.exp(- dists**2 / (2 * sigma**2))
    gaussian_filter = gaussian_filter / np.sum(gaussian_filter)
    return gaussian_filter


def fourier_transform(h, shape):
    """
    Получение Фурье-образа искажающей функции

    @param  h            numpy array  искажающая функция h (ядро свертки)
    @param  shape        list         требуемый размер образа
    @return numpy array  H            Фурье-образ искажающей функции h
    """
    return fft2(h, s=shape)


def inverse_kernel(H, threshold=1e-10):
    """
    Получение H_inv

    @param  H            numpy array    Фурье-образ искажающей функции h
    @param  threshold    float          порог отсечения для избежания деления на 0
    @return numpy array  H_inv
    """
    H_inv = np.zeros(H.shape, dtype=np.complex64)
    mask = (np.abs(H) > threshold)
    H_inv[mask] = 1 / H[mask]
    return H_inv


def inverse_filtering(blurred_img, h, threshold=1e-10):
    """
    Метод инверсной фильтрации

    @param  blurred_img    numpy array  искаженное изображение
    @param  h              numpy array  искажающая функция
    @param  threshold      float        параметр получения H_inv
    @return numpy array                 восстановленное изображение
    """
    G = fourier_transform(blurred_img, blurred_img.shape)
    H = fourier_transform(h, G.shape)
    H_inv = inverse_kernel(H, threshold)
    F = G * H_inv
    f = np.abs(ifft2(F))
    return f


def wiener_filtering(blurred_img, h, K=1e-4):
    """
    Винеровская фильтрация

    @param  blurred_img    numpy array  искаженное изображение
    @param  h              numpy array  искажающая функция
    @param  K              float        константа из выражения (8)
    @return numpy array                 восстановленное изображение
    """
    G = fourier_transform(blurred_img, blurred_img.shape)
    H = fourier_transform(h, G.shape)
    H_conj = np.conjugate(H)
    H_square = H_conj * H
    F = H_conj / (H_square + K) * G
    f = np.abs(ifft2(F))
    return f


def compute_psnr(img1, img2):
    """
    PSNR metric

    @param  img1    numpy array   оригинальное изображение
    @param  img2    numpy array   искаженное изображение
    @return float   PSNR(img1, img2)
    """
    h, w = img1.shape
    MSE = np.sum((img1 - img2)**2) / (h*w)
    PSNR = 20 * np.log10(255 / np.sqrt(MSE))
    return PSNR
