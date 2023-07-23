import torch
from torch import nn
from torch.nn import Sequential
from torchvision import transforms
import numpy as np
from collections import OrderedDict

# ============================== 1 Classifier model ============================

def get_cls_model(input_shape):
    """
    :param input_shape: tuple (n_rows, n_cols, n_channels)
            input shape of image for classification
    :return: nn model for classification
    """
    # your code here \/
    H, W, C = input_shape
    model = Sequential(OrderedDict([
        ('conv', nn.Conv2d(C, 8, (3, 3), stride=1, padding='same')),
        ('relu_1', nn.ReLU()),
        ('bn', nn.BatchNorm2d(8)),
        ('maxpool', nn.MaxPool2d((4, 4), stride=4)),
        ('flatten', nn.Flatten(1, -1)),
        ('fc1', nn.Linear(8*(H//4)*(W//4), 128, bias=False)),
        ('relu_2', nn.ReLU()),
        ('dropout', nn.Dropout(0.1)),
        ('fc2', nn.Linear(128, 2, bias=False))
    ]))
    return model


def fit_cls_model(X, y):
    """
    :param X: 4-dim tensor with training images
    :param y: 1-dim tensor with labels for training
    :return: trained nn model
    """
    # your code here \/
    N, C, H, W = X.shape
    model = get_cls_model((H, W, C))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    N_EPOCH = 20
    BATCH_SIZE = 64
    ITERS = N // BATCH_SIZE + 1

    for epoch in range(N_EPOCH):
        running_loss = 0

        for i in range(ITERS):
            data = X[BATCH_SIZE*i:BATCH_SIZE*(i+1)]
            labels = y[BATCH_SIZE*i:BATCH_SIZE*(i+1)]

            augmentation = transforms.Compose([
                transforms.RandomInvert(p=0.5),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=(0.75, 1.25))
            ])
            data = augmentation(data)

            optimizer.zero_grad()
            logits = model(data)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

    return model


# ============================ 2 Classifier -> FCN =============================
def get_detection_model(cls_model):
    """
    :param cls_model: trained cls model
    :return: fully convolutional nn model with weights initialized from cls
             model
    """
    # your code here \/
    conv_fc1 = nn.Conv2d(8, 128, (10, 25), padding='valid') # no padding
    conv_fc2 = nn.Conv2d(128, 2, (1, 1), padding='valid') # no padding
    
    with torch.no_grad():
        conv_fc1.weight = torch.nn.Parameter(cls_model.fc1.weight.reshape(128, 8, 10, 25))
        conv_fc2.weight = torch.nn.Parameter(cls_model.fc2.weight.reshape(2, 128, 1, 1))

    detection_model = Sequential(OrderedDict([
        ('conv', cls_model.conv),
        ('relu_1', cls_model.relu_1),
        ('bn', cls_model.bn),
        ('maxpool', cls_model.maxpool),  
        # shape after CNN part = [batch_size, 8, 10, 25]
        # it's 8 feature maps sized 10x25
        ('conv_fc1', conv_fc1),
        ('relu_2', cls_model.relu_2),
        ('dropout', cls_model.dropout),
        ('conv_fc2', conv_fc2),
        ('activation', nn.Softmax(dim=1)),
    ]))
    # output shape = [batch_size, 2, 1, 1]
    return detection_model


# ============================ 3 Simple detector ===============================
def get_detections(detection_model, dictionary_of_images):
    """
    :param detection_model: trained fully convolutional detector model
    :param dictionary_of_images: dictionary of images in format
        {filename: ndarray}
    :return: detections in format {filename: detections}. detections is a N x 5
        array, where N is number of detections. Each detection is described
        using 5 numbers: [row, col, n_rows, n_cols, confidence].
    """
    # your code here \/
    def pad_image(image):
        h, w = image.shape
        padded = np.zeros((220, 370), dtype='float64')
        padded[:h,:w] = image
        return padded
    
    STRIDE = 4
    RECEPTIVE_H, RECEPTIVE_W = 40, 100
    THRESHOLD = 0.9

    detections = {}
    detection_model.eval()
    images = [(pad_image(img), img.shape) for img in dictionary_of_images.values()]

    for filename, (image_padded, original_shape) in zip(dictionary_of_images.keys(), images):
        img_tensor = torch.tensor(image_padded).expand(1, 1, *image_padded.shape)

        detection = detection_model(img_tensor.float())
        detection = detection[0][1].detach().numpy()
        detected = np.argwhere(detection >= THRESHOLD)
        bboxes = [[h*STRIDE, w*STRIDE, RECEPTIVE_H, RECEPTIVE_W, detection[h,w]] for h, w in detected]
        bboxes_filtered = [
            bb for bb in bboxes 
            if (bb[0]+RECEPTIVE_H <= original_shape[0]) & 
               (bb[1]+RECEPTIVE_W <= original_shape[1])
        ]
        detections[filename] = bboxes_filtered
        assert len(bboxes_filtered) > 0, 'Filtered too many predicted bboxes!'
    
    return detections


# =============================== 5 IoU ========================================
def calc_iou(first_bbox, second_bbox):
    """
    :param first bbox: bbox in format (row, col, n_rows, n_cols)
    :param second_bbox: bbox in format (row, col, n_rows, n_cols)
    :return: iou measure for two given bboxes
    """
    # your code here \/
    row_1, col_1, n_rows_1, n_cols_1 = map(int, first_bbox)
    row_2, col_2, n_rows_2, n_cols_2 = map(int, second_bbox)
    
    # if negative - shift bboxes into positive area
    row_shift = max(0, -row_1, -row_2)
    col_shift = max(0, -col_1, -col_2)

    bbox_1 = np.zeros((220, 370), dtype=bool)
    bbox_1[row_1+row_shift:row_1+row_shift+n_rows_1, 
           col_1+col_shift:col_1+col_shift+n_cols_1] = 1

    bbox_2 = np.zeros((220, 370), dtype=bool)
    bbox_2[row_2+row_shift:row_2+row_shift+n_rows_2, 
           col_2+col_shift:col_2+col_shift+n_cols_2] = 1

    intersection = (bbox_1 & bbox_2).sum()
    union = (bbox_1 | bbox_2).sum()

    return intersection / union


# =============================== 6 AUC ========================================
def calc_auc(pred_bboxes, gt_bboxes):
    """
    :param pred_bboxes: dict of bboxes in format {filename: detections}
        detections is a N x 5 array, where N is number of detections. Each
        detection is described using 5 numbers: [row, col, n_rows, n_cols,
        confidence].
    :param gt_bboxes: dict of bboxes in format {filenames: bboxes}. bboxes is a
        list of tuples in format (row, col, n_rows, n_cols)
    :return: auc measure for given detections and gt
    """
    # your code here \/
    IOU_THR = 0.5
    TP, FP = {}, {}
    N_rectangles = 0

    for filename, detections in pred_bboxes.items():
        # (1.a)
        detections_sorted = sorted(detections, key=lambda x: x[-1], reverse=True)
        true_bboxes = gt_bboxes[filename].copy()
        N_rectangles += len(true_bboxes)
        # (1.b)
        TP[filename] = []
        FP[filename] = []
        for pred in detections_sorted:
            iou_best = 0
            index_best = None
            for i, true in enumerate(true_bboxes):
                iou = calc_iou(pred[:4], true)
                if iou > iou_best:
                    iou_best = iou
                    index_best = i
            # (1.c)(1.d)
            confidence = pred[-1]
            if iou_best >= IOU_THR:
                TP[filename].append(confidence)
                true_bboxes.pop(index_best)
            else:
                FP[filename].append(confidence)
    # (2)
    TP = sum([tp for tp in TP.values()], [])
    FP = sum([fp for fp in FP.values()], [])
    # (3)
    TP_FP = sorted(TP + FP)
    TP = sorted(TP)
    # (4)
    i = 0
    PR_curve = []
    last_c = None
    for j, c in enumerate(TP_FP):
        if last_c != c:
            tp_fp_above_c = len(TP_FP) - j
            while (i < len(TP)) and (TP[i] < c):
                i += 1
            tp_above_c = len(TP) - i
            # (5)
            recall = tp_above_c / N_rectangles
            precision = tp_above_c / tp_fp_above_c
            # (6)
            PR_curve.append((recall, precision, c))
            last_c = c
    
    PR_curve.append((0, 1, 1))
    PR_curve = np.array(PR_curve)
    yy = PR_curve[:,1] # precision
    xx = PR_curve[:,0] # recall
    AUC = sum(0.5 * (yy[:-1] + yy[1:]) * (xx[:-1] - xx[1:]))
    return AUC


# =============================== 7 NMS ========================================
def nms(detections_dictionary, iou_thr=0.05):
    """
    :param detections_dictionary: dict of bboxes in format {filename: detections}
        detections is a N x 5 array, where N is number of detections. Each
        detection is described using 5 numbers: [row, col, n_rows, n_cols,
        confidence].
    :param iou_thr: IoU threshold for nearby detections
    :return: dict in same format as detections_dictionary where close detections
        are deleted
    """
    # your code here \/
    detections_filtered = {}

    for filename, detections in detections_dictionary.items():
        # (1)
        detections_sorted = sorted(detections, key=lambda x: x[-1], reverse=True)
        # (2)
        N = len(detections_sorted)
        mask = [True] * N
        for i in range(N):
            if mask[i]:
                for j in range(i+1, N):
                    current = detections_sorted[i][:-1]
                    next = detections_sorted[j][:-1]
                    iou = calc_iou(current, next)
                    if iou > iou_thr:
                        mask[j] = False
        filtered = [d for m, d in zip(mask, detections_sorted) if m]
        detections_filtered[filename] = filtered
    return detections_filtered
