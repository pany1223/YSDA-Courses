import numpy as np
from scipy.signal import convolve2d

def get_bayer_masks(n_rows, n_cols):
    bayer = np.zeros((n_rows, n_cols, 3), 'bool')
    bayer[::2, 1::2, 0] = 1  # Red
    bayer[::2, ::2, 1] = 1   # Green
    bayer[1::2, 1::2, 1] = 1 # Green
    bayer[1::2, ::2, 2] = 1  # Blue
    return bayer


def get_colored_img(raw_img):
    mask = get_bayer_masks(*raw_img.shape)
    return np.dstack([raw_img * mask[...,i] for i in range(3)])


def bilinear_interpolation(colored_img):
    w, h, dim = colored_img.shape
    mask = get_bayer_masks(w, h)
    kernel = np.ones((3, 3))
    for c in range(dim):
        conv = (convolve2d(colored_img[...,c], kernel, mode='same') / \
                convolve2d(mask[...,c], kernel, mode='same')).astype('uint8')
        conv[mask[...,c]] = 0
        colored_img[...,c] += conv
    return colored_img


def improved_interpolation(raw_img):
    '''
    Inspired by and adjusted implementation:
    https://github.com/colour-science/colour-demosaicing/blob/master/colour_demosaicing/bayer/demosaicing/malvar2004.py
    '''
    # G at R locations / G at B locations
    GR_GB = np.array([[0.0, 0.0, -1.0, 0.0, 0.0],
                       [0.0, 0.0, 2.0, 0.0, 0.0],
                       [-1.0, 2.0, 4.0, 2.0, -1.0],
                       [0.0, 0.0, 2.0, 0.0, 0.0],
                       [0.0, 0.0, -1.0, 0.0, 0.0]]) / 8

    # R at green in R row, B column / B at green in B row, R column
    Rg_RB_Bg_BR = np.array([[0.0, 0.0, 0.5, 0.0, 0.0],
                            [0.0, -1.0, 0.0, -1.0, 0.0],
                            [-1.0, 4.0, 5.0, 4.0, -1.0],
                            [0.0, -1.0, 0.0, -1.0, 0.0],
                            [0.0, 0.0, 0.5, 0.0, 0.0]]) / 8
    
    # R at green in B row, R column / B at green in R row, B column
    Rg_BR_Bg_RB = Rg_RB_Bg_BR.T

    # R at blue in B row, B column / B at red in R row, R column 
    Rb_BB_Br_RR = np.array([[0.0, 0.0, -1.5, 0.0, 0.0],
                            [0.0, 2.0, 0.0, 2.0, 0.0],
                            [-1.5, 0.0, 6.0, 0.0, -1.5],
                            [0.0, 2.0, 0.0, 2.0, 0.0],
                            [0.0, 0.0, -1.5, 0.0, 0.0]]) / 8
    
    raw_img = raw_img.astype(np.float64)
    colored_img = get_colored_img(raw_img)
    w, h, dim = colored_img.shape
    mask = get_bayer_masks(w, h)
    
    R, G, B = [colored_img[...,i] for i in range(3)]
    R_mask, G_mask, B_mask = [mask[...,i] for i in range(3)]
    
    G = np.where(np.logical_or(R_mask == 1, B_mask == 1), convolve2d(raw_img, GR_GB, mode='same'), G)
    
    RBg_RBBR = convolve2d(raw_img, Rg_RB_Bg_BR, mode='same')
    RBg_BRRB = convolve2d(raw_img, Rg_BR_Bg_RB, mode='same')
    RBgr_BBRR = convolve2d(raw_img, Rb_BB_Br_RR, mode='same')
    
    R_rows = np.transpose(np.any(R_mask == 1, axis=1)[np.newaxis]) * np.ones(R.shape)
    R_columns = np.any(R_mask == 1, axis=0)[np.newaxis] * np.ones(R.shape)
    B_rows = np.transpose(np.any(B_mask == 1, axis=1)[np.newaxis]) * np.ones(B.shape)
    B_columns = np.any(B_mask == 1, axis=0)[np.newaxis] * np.ones(B.shape)

    R = np.where(np.logical_and(R_rows == 1, B_columns == 1), RBg_RBBR, R)
    R = np.where(np.logical_and(B_rows == 1, R_columns == 1), RBg_BRRB, R)
    B = np.where(np.logical_and(B_rows == 1, R_columns == 1), RBg_RBBR, B)
    B = np.where(np.logical_and(R_rows == 1, B_columns == 1), RBg_BRRB, B)
    R = np.where(np.logical_and(B_rows == 1, B_columns == 1), RBgr_BBRR, R)
    B = np.where(np.logical_and(R_rows == 1, R_columns == 1), RBgr_BBRR, B)
        
    result = np.clip(np.dstack([R, G, B]), 0, 255)
    return result.astype('uint8')


def compute_psnr(img_pred, img_gt):
    w, h, c = img_pred.shape
    pred = img_pred.astype(np.float64)
    gt = img_gt.astype(np.float64)
    MSE = np.sum((pred - gt)**2) / (c*h*w)
    if MSE == 0:
        raise ValueError
    PSNR = 10*np.log10(np.max(gt)**2 / MSE)
    return PSNR
