import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from scipy.ndimage import gaussian_filter
from skimage.metrics import peak_signal_noise_ratio
# !Этих импортов достаточно для решения данного задания, нельзя использовать другие библиотеки!


def pca_compression(matrix, p):
    """ Сжатие изображения с помощью PCA
    Вход: двумерная матрица (одна цветовая компонента картинки), количество компонент
    Выход: собственные векторы, проекция матрицы на новое пр-во и средние значения до центрирования
    """
    # Отцентруем каждую строчку матрицы
    rows_centers = np.mean(matrix, axis=1)
    matrix_centered = matrix - rows_centers[:, None]
    # Найдем матрицу ковариации
    covariance = np.cov(matrix_centered, ddof=0)
    # Ищем собственные значения и собственные векторы матрицы ковариации, используйте linalg.eigh из numpy
    eig_val, eig_vec = np.linalg.eigh(covariance)
    # Посчитаем количество найденных собственных векторов
    n_eig_vec = eig_vec.shape[1]
    # Сортируем собственные значения в порядке убывания
    eig_val_sorted_indices = np.argsort(eig_val)[::-1]
    # Сортируем собственные векторы согласно отсортированным собственным значениям
    # !Это все для того, чтобы мы производили проекцию в направлении максимальной дисперсии!
    eig_vec_sorted = eig_vec[:, eig_val_sorted_indices]
    # Оставляем только p собственных векторов
    eig_vec_p = eig_vec_sorted[:, :p]
    # Проекция данных на новое пространство
    projection = eig_vec_p.T @ matrix_centered
    
    return eig_vec_p, projection, rows_centers


def pca_decompression(compressed):
    """ Разжатие изображения
    Вход: список кортежей из собственных векторов и проекций для каждой цветовой компоненты
    Выход: разжатое изображение
    """
    result_img = []
    for i, comp in enumerate(compressed):
        # Матрично умножаем собственные векторы на проекции и прибавляем среднее значение по строкам исходной матрицы
        # !Это следует из описанного в самом начале примера!
        eig_vec, projection, rows_centers = comp
        decompressed = eig_vec @ projection + rows_centers[:, None]
        decompressed = np.clip(decompressed, 0, 255).astype(np.uint8)
        result_img.append(decompressed)
    return np.dstack(result_img)


def pca_visualize():
    plt.clf()
    img = imread('cat.jpg')
    if len(img.shape) == 3:
        img = img[..., :3]
    fig, axes = plt.subplots(3, 3)
    fig.set_figwidth(12)
    fig.set_figheight(12)

    for i, p in enumerate([1, 5, 10, 20, 50, 100, 150, 200, 256]):
        compressed = []
        for j in range(0, 3):
            compressed.append(pca_compression(img[:, :, j], p))
            
        axes[i // 3, i % 3].imshow(pca_decompression(compressed))
        axes[i // 3, i % 3].set_title('Компонент: {}'.format(p))

    fig.savefig("pca_visualization.png")


def rgb2ycbcr(img):
    """ Переход из пр-ва RGB в пр-во YCbCr
    Вход: RGB изображение
    Выход: YCbCr изображение
    """
    R, G, B = img[..., 0], img[..., 1], img[..., 2]
    Y = 0.299*R + 0.587*G + 0.114*B
    C_b = 128. - 0.1687*R - 0.3313*G + 0.5*B
    C_r = 128. + 0.5*R - 0.4187*G - 0.0813*B
    return np.dstack((Y, C_b, C_r))


def ycbcr2rgb(img):
    """ Переход из пр-ва YCbCr в пр-во RGB
    Вход: YCbCr изображение
    Выход: RGB изображение
    """
    Y, C_b, C_r = img[..., 0], img[..., 1], img[..., 2]
    R = Y + 1.402*(C_r - 128)
    G = Y - 0.34414*(C_b - 128) - 0.71414*(C_r - 128)
    B = Y + 1.77*(C_b - 128)
    return np.dstack((R, G, B))


def get_gauss_1():
    plt.clf()
    rgb_img = imread('Lenna.png')
    if len(rgb_img.shape) == 3:
        rgb_img = rgb_img[..., :3]
        
    YCbCr = rgb2ycbcr(rgb_img)
    Y, C_b, C_r = YCbCr[..., 0], YCbCr[..., 1], YCbCr[..., 2]
    C_b_blurred = gaussian_filter(input=C_b, sigma=10)
    C_r_blurred = gaussian_filter(input=C_r, sigma=10)
    restored = np.clip(ycbcr2rgb(np.dstack((Y, C_b_blurred, C_r_blurred))), 0, 255).astype(np.uint8)
    
    plt.imshow(restored)
    plt.savefig("gauss_1.png")


def get_gauss_2():
    plt.clf()
    rgb_img = imread('Lenna.png')
    if len(rgb_img.shape) == 3:
        rgb_img = rgb_img[..., :3]
        
    YCbCr = rgb2ycbcr(rgb_img)
    Y, C_b, C_r = YCbCr[..., 0], YCbCr[..., 1], YCbCr[..., 2]
    Y_blurred = gaussian_filter(input=Y, sigma=10)
    restored = np.clip(ycbcr2rgb(np.dstack((Y_blurred, C_b, C_r))), 0, 255).astype(np.uint8)
    
    plt.imshow(restored)
    plt.savefig("gauss_2.png")


def downsampling(component):
    """Уменьшаем цветовые компоненты в 2 раза
    Вход: цветовая компонента размера [A, B, 1]
    Выход: цветовая компонента размера [A // 2, B // 2, 1]
    """
    blurred = gaussian_filter(input=component, sigma=10)
    return blurred[::2, ::2]


def alpha(x):
    return 1/np.sqrt(2) if x == 0 else 1


def dct(block):
    """Дискретное косинусное преобразование
    Вход: блок размера 8x8
    Выход: блок размера 8x8 после ДКП
    """
    res = np.zeros((8, 8))
    for u in range(8):
        for v in range(8):
            alpha_u, alpha_v = alpha(u), alpha(v)
            cosines_sum = 0
            for x in range(8):
                for y in range(8):
                    cosines_sum += block[x][y] * np.cos((2*x+1)*u*np.pi/16) * np.cos((2*y+1)*v*np.pi/16)
            res[u][v] = 1/4 * alpha_u * alpha_v * cosines_sum
    return res


# Матрица квантования яркости
y_quantization_matrix = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
])

# Матрица квантования цвета
color_quantization_matrix = np.array([
    [17, 18, 24, 47, 99, 99, 99, 99],
    [18, 21, 26, 66, 99, 99, 99, 99],
    [24, 26, 56, 99, 99, 99, 99, 99],
    [47, 66, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99]
])


def quantization(block, quantization_matrix):
    """Квантование
    Вход: блок размера 8x8 после применения ДКП; матрица квантования
    Выход: блок размера 8x8 после квантования. Округление осуществляем с помощью np.round
    """
    return np.round(block / quantization_matrix)


def own_quantization_matrix(default_quantization_matrix, q):
    """Генерация матрицы квантования по Quality Factor
    Вход: "стандартная" матрица квантования; Quality Factor
    Выход: новая матрица квантования
    Hint: если после проделанных операций какие-то элементы обнулились, то замените их единицами
    """
    assert 1 <= q <= 100
    if q < 50:
        s = 5000 / q
    elif q < 100:
        s = 200 - 2 * q
    else:
        s = 1
    Q = np.floor((50 + s * default_quantization_matrix) / 100)
    Q[Q == 0] = 1
    return Q


def zigzag(block):
    """Зигзаг-сканирование
    Вход: блок размера 8x8
    Выход: список из элементов входного блока, получаемый после его обхода зигзаг-сканированием
    """
    res = []
    size = block.shape[0]
    block = np.flip(block, axis=0)
    for step, offset in enumerate(range(1-size, size)):
        res.append(np.diagonal(block, offset)[::(-1)**step])
    return np.concatenate(res)


def compression(zigzag_list):
    """Сжатие последовательности после зигзаг-сканирования
    Вход: список после зигзаг-сканирования
    Выход: сжатый список в формате, который был приведен в качестве примера в самом начале данного пункта
    """
    res = []
    zeros_cnt = 0
    for x in zigzag_list:
        if x == 0:
            zeros_cnt += 1
        else:
            if zeros_cnt > 0:
                res += [0, zeros_cnt]
                zeros_cnt = 0
            res += [x]
    if zeros_cnt > 0:
        res += [0, zeros_cnt]
    return np.array(res)


def split_into_blocks(matrix, block_shape):
    xx, yy = block_shape
    return (
        matrix
        .reshape(matrix.shape[0] // xx, yy, -1, yy)
        .swapaxes(1, 2)
        .reshape(-1, xx, yy)
    )


def jpeg_compression(img, quantization_matrixes):
    """JPEG-сжатие
    Вход: цветная картинка, список из 2-ух матриц квантования
    Выход: список списков со сжатыми векторами: [[compressed_y1,...], [compressed_Cb1,...], [compressed_Cr1,...]]
    """
    # Переходим из RGB в YCbCr
    YCbCr = rgb2ycbcr(img)
    Y, C_b, C_r = YCbCr[..., 0], YCbCr[..., 1], YCbCr[..., 2]
    # Уменьшаем цветовые компоненты
    C_b, C_r = downsampling(C_b), downsampling(C_r)
    # Делим все компоненты на блоки 8x8 и все элементы блоков переводим из [0, 255] в [-128, 127]
    Y = split_into_blocks(Y, (8, 8)) - 128
    C_b = split_into_blocks(C_b, (8, 8)) - 128
    C_r = split_into_blocks(C_r, (8, 8)) - 128
    # Применяем ДКП, квантование, зизгаз-сканирование и сжатие
    Y = [compression(zigzag(quantization(dct(block), quantization_matrixes[0]))) for block in Y]
    C_b = [compression(zigzag(quantization(dct(block), quantization_matrixes[1]))) for block in C_b]
    C_r = [compression(zigzag(quantization(dct(block), quantization_matrixes[1]))) for block in C_r]
    return [Y, C_b, C_r]


def inverse_compression(compressed_list):
    """Разжатие последовательности
    Вход: сжатый список
    Выход: разжатый список
    """
    res = []
    i = 0
    while i < len(compressed_list):
        if compressed_list[i] == 0:
            res += [0] * int(compressed_list[i+1])
            i += 2
        else:
            res += [compressed_list[i]]
            i += 1
    return res


def nth_matrix_diag(matrix, n):
    xx, yy = np.diag_indices_from(matrix)
    if n > 0:
        return xx[:-n], yy[n:]
    elif n < 0:
        return xx[-n:], yy[:n]
    else:
        return xx, yy

        
def inverse_zigzag(input):
    """Обратное зигзаг-сканирование
    Вход: список элементов
    Выход: блок размера 8x8 из элементов входного списка, расставленных в матрице в порядке их следования в зигзаг-сканировании
    """
    size = int(np.sqrt(len(input)))
    matrix = np.zeros((size, size))
    i = j = 0
    for step, offset in enumerate(range(1-size, size)):
        indices = nth_matrix_diag(matrix, offset)
        delta = len(indices[0])
        matrix[indices] = input[i:j+delta][::(-1)**step]
        i = j = j + delta
    matrix = np.flip(matrix, axis=0)
    return matrix


def inverse_quantization(block, quantization_matrix):
    """Обратное квантование
    Вход: блок размера 8x8 после применения обратного зигзаг-сканирования; матрица квантования
    Выход: блок размера 8x8 после квантования. Округление не производится
    """
    return block * quantization_matrix


def inverse_dct(block):
    """Обратное дискретное косинусное преобразование
    Вход: блок размера 8x8
    Выход: блок размера 8x8 после обратного ДКП. Округление осуществляем с помощью np.round
    """
    res = np.zeros((8, 8))
    for x in range(8):
        for y in range(8):
            cosines_sum = 0
            for u in range(8):
                for v in range(8):
                    alpha_u, alpha_v = alpha(u), alpha(v)
                    cosines_sum += alpha_u * alpha_v * block[u][v] * np.cos((2*x+1)*u*np.pi/16) * np.cos((2*y+1)*v*np.pi/16)
            res[x][y] = 1/4 * cosines_sum
    res = np.round(res)
    return res


def upsampling(component):
    """Увеличиваем цветовые компоненты в 2 раза
    Вход: цветовая компонента размера [A, B, 1]
    Выход: цветовая компонента размера [2 * A, 2 * B, 1]
    """
    columns_restored = np.repeat(component, repeats=2, axis=1)
    all_restored = np.repeat(columns_restored, repeats=2, axis=0)
    return all_restored


def inverse_split_into_blocks(blocks):
    length, xx, yy = blocks.shape
    size = int(length**0.5)
    h = w = int((length*xx*yy)**0.5)
    return (
        blocks
        .reshape(size, size, xx, yy)
        .swapaxes(2, 1)
        .reshape(h, w)
    )


def jpeg_decompression(result, result_shape, quantization_matrixes):
    """Разжатие изображения
    Вход: result список сжатых данных, размер ответа, список из 2-ух матриц квантования
    Выход: разжатое изображение
    """
    [Y, C_b, C_r] = result
    Y = [inverse_dct(inverse_quantization(inverse_zigzag(inverse_compression(y)), quantization_matrixes[0])) for y in Y]
    C_b = [inverse_dct(inverse_quantization(inverse_zigzag(inverse_compression(cb)), quantization_matrixes[1])) for cb in C_b]
    C_r = [inverse_dct(inverse_quantization(inverse_zigzag(inverse_compression(cr)), quantization_matrixes[1])) for cr in C_r]
    Y = inverse_split_into_blocks(np.array(Y)) + 128
    C_b = inverse_split_into_blocks(np.array(C_b)) + 128
    C_r = inverse_split_into_blocks(np.array(C_r)) + 128
    C_b = upsampling(C_b)
    C_r = upsampling(C_r)
    YCbCr = np.dstack([Y, C_b, C_r])
    RGB = ycbcr2rgb(YCbCr)
    RGB = np.clip(RGB, 0, 255).astype(np.uint8)
    return RGB


def jpeg_visualize():
    plt.clf()
    img = imread('Lenna.png')
    if len(img.shape) == 3:
        img = img[..., :3]
    fig, axes = plt.subplots(2, 3)
    fig.set_figwidth(12)
    fig.set_figheight(12)

    for i, p in enumerate([1, 10, 20, 50, 80, 100]):
        y_quantization = own_quantization_matrix(y_quantization_matrix, p)
        color_quantization = own_quantization_matrix(color_quantization_matrix, p)
        matrixes = [y_quantization, color_quantization]
        
        compressed = jpeg_compression(img, matrixes)
        img_restored = jpeg_decompression(compressed, img.shape, matrixes)

        axes[i // 3, i % 3].imshow(img_restored)
        axes[i // 3, i % 3].set_title('Quality Factor: {}'.format(p))

    fig.savefig("jpeg_visualization.png")


def compression_pipeline(img, c_type, param=1):
    """Pipeline для PCA и JPEG
    Вход: исходное изображение; название метода - 'pca', 'jpeg'; 
    param - кол-во компонент в случае PCA, и Quality Factor для JPEG
    Выход: изображение; количество бит на пиксель
    """
    
    assert c_type.lower() == 'jpeg' or c_type.lower() == 'pca'
    
    if c_type.lower() == 'jpeg':
        y_quantization = own_quantization_matrix(y_quantization_matrix, param)
        color_quantization = own_quantization_matrix(color_quantization_matrix, param)
        matrixes = [y_quantization, color_quantization]
        
        compressed = jpeg_compression(img, matrixes)
        img = jpeg_decompression(compressed, img.shape, matrixes)
    elif c_type.lower() == 'pca':
        compressed = []
        for j in range(0, 3):
            compressed.append((pca_compression(img[:, :, j].astype(np.float64).copy(), param)))
            
        img = pca_decompression(compressed)
        compressed.extend([np.mean(img[:, :, 0], axis=1), np.mean(img[:, :, 1], axis=1), np.mean(img[:, :, 2], axis=1)])
        
    if 'tmp' not in os.listdir() or not os.path.isdir('tmp'):
        os.mkdir('tmp')
        
    np.savez_compressed(os.path.join('tmp', 'tmp.npz'), compressed)
    size = os.stat(os.path.join('tmp', 'tmp.npz')).st_size * 8
    os.remove(os.path.join('tmp', 'tmp.npz'))
        
    return img, size / (img.shape[0] * img.shape[1])


def calc_metrics(img_path, c_type, param_list):
    """Подсчет PSNR и Rate-Distortion для PCA и JPEG. Построение графиков
    Вход: пусть до изображения; тип сжатия; список параметров: кол-во компонент в случае PCA, и Quality Factor для JPEG
    """
    
    assert c_type.lower() == 'jpeg' or c_type.lower() == 'pca'
    
    img = imread(img_path)
    if len(img.shape) == 3:
        img = img[..., :3]
    
    outputs = []
    for param in param_list:
        outputs.append(compression_pipeline(img.copy(), c_type, param))
     
    psnr = [peak_signal_noise_ratio(img, output[0]) for output in outputs]
    rate = [output[1] for output in outputs]
    
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
    fig.set_figwidth(20)
    fig.set_figheight(5)
    
    ax1.set_title('PSNR for {}'.format(c_type.upper()))
    ax1.plot(param_list, psnr, 'tab:orange')
    ax1.set_xlabel('Quality Factor')
    ax1.set_ylabel('PSNR')
    
    ax2.set_title('Rate-Distortion for {}'.format(c_type.upper()))
    ax2.plot(psnr, rate, 'tab:red')
    ax2.set_xlabel('Distortion')
    ax2.set_ylabel('Rate')
    return fig


def get_pca_metrics_graph():
    plt.clf()
    fig = calc_metrics('Lenna.png', 'pca', [1, 5, 10, 20, 50, 100, 150, 200, 256])
    fig.savefig("pca_metrics_graph.png")


def get_jpeg_metrics_graph():
    plt.clf()
    fig = calc_metrics('Lenna.png', 'jpeg', [1, 10, 20, 50, 80, 100])
    fig.savefig("jpeg_metrics_graph.png")

