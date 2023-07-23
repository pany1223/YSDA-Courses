import numpy as np


def get_Y(raw_img):
    img = np.array(raw_img, dtype='float64')
    R, G, B = raw_img[...,0], raw_img[...,1], raw_img[...,2]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    return Y

def get_derivative(intensity, axis):
    # 1 ~ x, 0 ~ y
    intensity = np.pad(intensity, pad_width=1, mode='edge')
    derivative = (np.roll(intensity, shift=-1, axis=axis) - 
                  np.roll(intensity, shift=1, axis=axis))[1:-1,1:-1]
    return derivative

def get_energy(intensity):
    # energy = norm of gradient
    x_diff = get_derivative(intensity, axis=1)
    y_diff = get_derivative(intensity, axis=0)
    energy = np.sqrt(x_diff ** 2 + y_diff ** 2)
    return energy

#------------------------------------------------------------------------

def global_index(triplet_index, row_index, max_index):
    # map local index of minimum energy into global index of image
    return {0: max(0, row_index - 1), 
            1: row_index, 
            2: min(row_index + 1, max_index)}[triplet_index]

def get_all_seams(energy):
    h, w = energy.shape
    seams = np.zeros((h, w), dtype='float64')
    seams[0,:] = energy[0,:]
    min_energy_pointers = []
    for i in range(1, h):
        current_row = energy[i,:]
        upper_row = seams[i-1,:]
        minimums, indices = [], []
        for j in range(w):
            triplet = upper_row[max(0, j-1):min(w, j+2)]
            minimums.append(np.min(triplet))
            if j == 0:
                indices.append(np.argmin(triplet))
            else:
                # argmin returns index of the most left minimum if there're many the same
                indices.append(global_index(np.argmin(triplet), j, w-1))
        seams[i,:] += minimums + current_row
        min_energy_pointers.append(indices)
    return seams, np.array(min_energy_pointers, dtype='int')

def get_minimal_seam(seams, min_energy_pointers):
    seam_mask = np.zeros(seams.shape, dtype='uint8')
    current_pointer = np.argmin(seams[-1,:])
    seam_mask[-1,current_pointer] = 1
    # from bottom to top
    for i, row in enumerate(reversed(min_energy_pointers)):
        current_pointer = row[current_pointer]
        seam_mask[-2-i,current_pointer] = 1
    return np.array(seam_mask, dtype='uint8')

#------------------------------------------------------------------------

def apply_mask(raw_img, mask):
    h, w, c = raw_img.shape
    return np.dstack([raw_img[...,i][~mask.astype(bool)].reshape((h, w-1)) for i in range(c)])

def shrink_horizontal(img, mask):
    h, w, _ = img.shape
    y = get_Y(img)
    energy = get_energy(y)
    if mask is not None:
        energy += np.array(mask, dtype='float64') * h * w * 256
    seams, min_energy_pointers = get_all_seams(energy)
    seam_mask = get_minimal_seam(seams, min_energy_pointers)
    shrinked = apply_mask(img, seam_mask)
    if mask is not None:
        mask = mask[~seam_mask.astype(bool)].reshape((h, w-1))
    return shrinked, mask, seam_mask

def rotate_flip(img):
    img = np.rot90(img, axes=(1,0))
    img = np.flip(img, axis=1)
    return img

def rotate_flip_reverse(img):
    img = np.flip(img, axis=1)
    img = np.rot90(img, axes=(0,1))
    return img

def shrink_vertical(img, mask):
    img_rotated = rotate_flip(img)
    if mask is not None:
        mask = rotate_flip(mask)
    shrinked_rotated, mask, seam_mask_rotated = shrink_horizontal(img_rotated, mask)
    shrinked = rotate_flip_reverse(shrinked_rotated)
    seam_mask = rotate_flip_reverse(seam_mask_rotated)
    if mask is not None:
        mask = rotate_flip_reverse(mask)
    return shrinked, mask, seam_mask

#------------------------------------------------------------------------

def insert_into_array(array, insert_mask, values):
    h, w = array.shape
    array_new = np.zeros((h,w+1))
    for i in range(h):
        array_new[i] = np.insert(array[i], 
                                 np.where(insert_mask[i])[0][0], 
                                 values[i])
    return array_new

def insert_new_seam(raw_img, seam_mask):
    h, w, c = raw_img.shape
    img = np.array(raw_img, dtype='float64')
    last_column = seam_mask[:,-1:]
    seam_mask_padded = np.hstack([seam_mask, last_column])
    neighbour_mask = np.roll(seam_mask_padded, shift=1, axis=1)[:,:-1]
    img_channels = []
    for ch in range(c):
        channel = img[...,ch]
        new_seam_values = (channel[seam_mask.astype(bool)] + 
                           channel[neighbour_mask.astype(bool)]) / 2
        channel_new = insert_into_array(channel, neighbour_mask, new_seam_values)
        img_channels.append(channel_new)
    img_new = np.array(np.dstack(img_channels), dtype='uint8')
    return img_new, neighbour_mask

def expand_horizontal(img, mask):
    h, w, _ = img.shape
    y = get_Y(img)
    energy = get_energy(y)
    if mask is not None:
        energy += np.array(mask, dtype='float64') * h * w * 256
    seams, min_energy_pointers = get_all_seams(energy)
    seam_mask = get_minimal_seam(seams, min_energy_pointers)
    img_new, neighbour_mask = insert_new_seam(img, seam_mask)
    if mask is not None:
        mask = mask + seam_mask 
        mask = insert_into_array(mask, neighbour_mask, np.zeros(h))
    return img_new, mask, seam_mask

def expand_vertical(img, mask):
    img_rotated = rotate_flip(img)
    if mask is not None:
        mask = rotate_flip(mask)
    expanded_rotated, mask, seam_mask_rotated = expand_horizontal(img_rotated, mask)
    expanded = rotate_flip_reverse(expanded_rotated)
    seam_mask = rotate_flip_reverse(seam_mask_rotated)
    if mask is not None:
        mask = rotate_flip_reverse(mask)
    return expanded, mask, seam_mask

#------------------------------------------------------------------------

def seam_carve(img, mode, mask=None):
    if mode == 'horizontal shrink':
        img_new, mask_new, seam_mask = shrink_horizontal(img, mask)
    if mode == 'vertical shrink':
        img_new, mask_new, seam_mask = shrink_vertical(img, mask)
    if mode == 'horizontal expand':
        img_new, mask_new, seam_mask = expand_horizontal(img, mask)
    if mode == 'vertical expand':
        img_new, mask_new, seam_mask = expand_vertical(img, mask)
        
    return img_new, mask_new, seam_mask