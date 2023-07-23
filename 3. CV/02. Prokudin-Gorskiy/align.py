import numpy as np


def cut_img(raw_img, cut_edges=0.1):
    h, w = raw_img.shape
    hh = h//3
    h_cut, w_cut = int(hh * cut_edges), int(w * cut_edges)
    B = raw_img[:hh][h_cut:-h_cut, w_cut:-w_cut]
    G = raw_img[hh:2*hh][h_cut:-h_cut, w_cut:-w_cut]
    R = raw_img[2*hh:3*hh][h_cut:-h_cut, w_cut:-w_cut]
    return B, G, R


def shift_image(image, dv, du):
    image = np.roll(image, du, axis=0)
    image = np.roll(image, dv, axis=1)
    if du > 0:
        image[:du,:] = 0
    else:
        image[du:,:] = 0
    if dv > 0:
        image[:,:dv] = 0
    else:
        image[:,dv:] = 0
    return image


def normalize_shift(u, v, h, w):
    if np.abs(u) > h/2:
        u = u - h * np.sign(u)
    if np.abs(v) > w/2:
        v = v - w * np.sign(v)
    return u, v


def get_shift_Fourier_transform(I1, I2):
    # h,w ~ u,v
    h, w = I1.shape
    FI1 = np.fft.fft2(I1)
    FI2_conj = np.conjugate(np.fft.fft2(I2))
    C = np.fft.ifft2(FI1 * FI2_conj)
    u, v = np.unravel_index(np.argmax(C), (h, w))
    u, v = normalize_shift(u, v, h, w)
    return u, v


def align(img, g_coord):
    '''
    Get raw image and Green point, return aligned image and corresponding points in Blue and Red channels
    Channel order: Blue, Green, Red (BGR)
    '''
    h_shift = img.shape[0]//3
    g_row, g_col = g_coord
    B, G, R = cut_img(img)
    R_u, R_v = get_shift_Fourier_transform(G, R)
    B_u, B_v = get_shift_Fourier_transform(G, B)
    aligned_img = np.dstack([shift_image(R, R_v, R_u),
                             G,
                             shift_image(B, B_v, B_u)])
    # row,col ~ u,v
    b_row, b_col = g_row - B_u - h_shift, g_col - B_v
    r_row, r_col = g_row - R_u + h_shift, g_col - R_v
    return aligned_img, (b_row, b_col), (r_row, r_col)
