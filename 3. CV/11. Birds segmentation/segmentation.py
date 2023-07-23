import os
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from collections import Counter, OrderedDict

import cv2
import skimage
from skimage import img_as_ubyte
from skimage.io import imread
from PIL import Image

import torch
from torch import nn
from torch.nn import Sequential
from torch.nn import functional as F
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint


class BirdsDataset(Dataset):
    def __init__(self, 
                 mode: str,
                 fast_train: bool = True,
                 train_data_path: str = '00_test_val_input/train',
                 val_images_path: str = '00_test_val_input/test/images',
                 vat_gt_path: str = '00_test_val_gt',
                 normalize: bool = True
                ):
        self._items = []
        self._normalize = normalize
        
        train_path = os.path.join(train_data_path, 'images')
        gt_train_path = os.path.join(train_data_path, 'gt')

        birds_classes = sorted(os.listdir(train_path))

        # take only 1 class out of 200
        if fast_train:
            birds_classes = birds_classes[:1]

        for bird in birds_classes:
            if mode == 'train':
                filenames = sorted(os.listdir(os.path.join(train_path, bird)))
                for filename in filenames:
                    img_path = os.path.join(train_path, bird, filename)
                    gt_path = os.path.join(gt_train_path, bird, filename.replace('.jpg', '.png'))
                    self._items.append((img_path, gt_path))
            elif mode == 'val':
                filenames = sorted(os.listdir(os.path.join(val_images_path, bird)))
                for filename in filenames:
                    img_path = os.path.join(val_images_path, bird, filename)
                    gt_path = os.path.join(vat_gt_path, bird, filename.replace('.jpg', '.png'))
                    self._items.append((img_path, gt_path))
            else:
                raise ValueError('Wrong mode!')

    def __len__(self):
        return len(self._items)

    def __getitem__(self, index):
        img_path, gt_path = self._items[index]

        image = np.array(imread(img_path)).astype(np.float32)
        gt = np.array(img_as_ubyte(imread(gt_path, as_gray=True)) > 127).astype(np.float32)

        # resizing to 352x352
        image = cv2.resize(image, (352, 352))
        gt = cv2.resize(gt, (352, 352))

        # single channel image
        if len(image.shape) == 2:
            image = np.dstack([image]*3)

        # normalize and to tensor
        image = transforms.ToTensor()(image)
        gt = transforms.ToTensor()(gt)
        if self._normalize:
            image = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(image)

        return image, gt
    

def dice_loss(pred, target, smooth=1.):
    pred = pred.contiguous()
    target = target.contiguous()    
    intersection = (pred * target).sum(dim=2).sum(dim=2)
    summ = pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2)
    loss = (1 - (2 * intersection + smooth) / (summ + smooth)).mean()
    return loss

    
def convrelu(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.ReLU(inplace=True),
    )


class ResNetUNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.base_model = models.resnet18(pretrained=False)
        self.base_layers = list(self.base_model.children())

        self.layer0 = nn.Sequential(*self.base_layers[:3]) # size=(N, 64, x.H/2, x.W/2)
        self.layer0_1x1 = convrelu(64, 64, 1, 0)
        self.layer1 = nn.Sequential(*self.base_layers[3:5]) # size=(N, 64, x.H/4, x.W/4)
        self.layer1_1x1 = convrelu(64, 64, 1, 0)
        self.layer2 = self.base_layers[5]  # size=(N, 128, x.H/8, x.W/8)
        self.layer2_1x1 = convrelu(128, 128, 1, 0)
        self.layer3 = self.base_layers[6]  # size=(N, 256, x.H/16, x.W/16)
        self.layer3_1x1 = convrelu(256, 256, 1, 0)
        self.layer4 = self.base_layers[7]  # size=(N, 512, x.H/32, x.W/32)
        self.layer4_1x1 = convrelu(512, 512, 1, 0)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_up3 = convrelu(256 + 512, 512, 3, 1)
        self.conv_up2 = convrelu(128 + 512, 256, 3, 1)
        self.conv_up1 = convrelu(64 + 256, 256, 3, 1)
        self.conv_up0 = convrelu(64 + 256, 128, 3, 1)

        self.conv_original_size0 = convrelu(3, 64, 3, 1)
        self.conv_original_size1 = convrelu(64, 64, 3, 1)
        self.conv_original_size2 = convrelu(64 + 128, 64, 3, 1)

        self.conv_last = nn.Conv2d(64, 1, 1)

    def forward(self, input):
        x_original = self.conv_original_size0(input)
        x_original = self.conv_original_size1(x_original)

        layer0 = self.layer0(input)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        layer4 = self.layer4_1x1(layer4)
        x = self.upsample(layer4)
        layer3 = self.layer3_1x1(layer3)
        x = torch.cat([x, layer3], dim=1)
        x = self.conv_up3(x)

        x = self.upsample(x)
        layer2 = self.layer2_1x1(layer2)
        x = torch.cat([x, layer2], dim=1)
        x = self.conv_up2(x)

        x = self.upsample(x)
        layer1 = self.layer1_1x1(layer1)
        x = torch.cat([x, layer1], dim=1)
        x = self.conv_up1(x)

        x = self.upsample(x)
        layer0 = self.layer0_1x1(layer0)
        x = torch.cat([x, layer0], dim=1)
        x = self.conv_up0(x)

        x = self.upsample(x)
        x = torch.cat([x, x_original], dim=1)
        x = self.conv_original_size2(x)

        out = self.conv_last(x)

        return out

    
class MyModel(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.model = ResNetUNet()
        self.bce_weight = 0.6

        # freeze all backbone layers
        for l in self.model.base_layers:
            for param in l.parameters():
                param.requires_grad = False

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_logit = self(x)
        bce = F.binary_cross_entropy_with_logits(y_logit, y)
        pred = torch.sigmoid(y_logit)
        dice = dice_loss(pred, y)
        loss = bce * self.bce_weight + dice * (1 - self.bce_weight) * y.size(0)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_logit = self(x)
        bce = F.binary_cross_entropy_with_logits(y_logit, y)
        pred = torch.sigmoid(y_logit)
        dice = dice_loss(pred, y)
        # 0.6 weight for BCE, 0.4 for Dice
        loss = bce * self.bce_weight + dice * (1 - self.bce_weight) * y.size(0)
        return {'val_loss': loss, 'logs': {'dice': dice, 'bce': bce}}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        print(f"| Train_loss: {avg_loss:.3f}" )
        self.log('train_loss', avg_loss, prog_bar=True, on_epoch=True, on_step=False)
     
    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_dice = torch.stack([x['logs']['dice'] for x in outputs]).mean()
        avg_bce = torch.stack([x['logs']['bce'] for x in outputs]).mean()
        print(f"[Epoch {self.trainer.current_epoch:3}] Val_loss: {avg_loss:.3f}, Val_dice: {avg_dice:.3f}, Val_bce: {avg_bce:.3f}", end= " ")
        self.log('val_loss', avg_loss, prog_bar=True, on_epoch=True, on_step=False)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=5e-4)
        
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,
                                                              gamma=0.96,
                                                              verbose=True)
        lr_dict = {
            "scheduler": lr_scheduler,
            "interval": "epoch",
            "frequency": 3,
            "monitor": "val_loss"
        } 
        return [optimizer], [lr_dict]

    
def get_model():
    return MyModel()


def predict(model, img_path):
    '''
    Returns matrix HxW with probabilities Object/Background
    '''
    image = np.array(imread(img_path)).astype(np.float32)
    shape = tuple(image.shape[:2])

    image = cv2.resize(image, (352, 352))
    if len(image.shape) == 2:
        image = np.dstack([image]*3)
    image = transforms.ToTensor()(image)
    image = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(image)
    image = image.unsqueeze(0)
    
    model.eval()
    with torch.no_grad():
        logits = model(image)

    pred = torch.sigmoid(logits)[0][0]
    pred = cv2.resize(pred.cpu().detach().numpy(), shape[::-1])
    return pred


def train_segmentation_model(train_data_path):
    '''
    Train segmentation model
    '''
    ds_train = BirdsDataset(mode="train", fast_train=False, train_data_path=train_data_path)
    ds_val = BirdsDataset(mode="val", fast_train=False, train_data_path=train_data_path)

    dl_train = DataLoader(ds_train, batch_size=32, shuffle=True, num_workers=2)
    dl_val = DataLoader(ds_val, batch_size=32, shuffle=False, num_workers=2)

    model = get_model()

    MyModelCheckpoint = ModelCheckpoint(dirpath='runs/pl_segmentation',
                                        filename='{epoch}-{val_loss:.3f}',
                                        monitor='val_loss', 
                                        mode='min', 
                                        save_top_k=1)

    MyEarlyStopping = EarlyStopping(monitor="val_loss",
                                    mode="min",
                                    patience=5,
                                    verbose=True)
    
    trainer = pl.Trainer(max_epochs=30,
                         accelerator='gpu',
                         devices=1,
                         precision=16,
                         logger=True,
                         enable_checkpointing=True,
                         callbacks=[MyEarlyStopping, MyModelCheckpoint])

    trainer.fit(model, dl_train, dl_val)

    # setting weights <1e-38 to zero, if there're
    with torch.no_grad():
        for name, param in model.model.named_parameters():
            if not name.startswith('base_model'):
                param[param.abs() < np.finfo('float32').tiny] = 0.

    torch.save(model.to('cpu').state_dict(), "segmentation_model.pth")

    return model
