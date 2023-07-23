import os
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import cv2
from skimage.io import imread
import albumentations as A
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint


class FacepointsDataset(Dataset):
    def __init__(self, 
                 mode: str, 
                 gt: dict,
                 img_dir: str,
                 fraction: float = 0.8, 
                 transform = None,
                ):
        self._items = []
        self._transform = transform

        imgs = sorted(os.listdir(img_dir))
        split = int(fraction * len(imgs))
        
        if mode == 'train':
            img_names = imgs[:split]
        elif mode == 'val':
            img_names = imgs[split:]
        else:
            raise ValueError(f'Wrong mode = {mode}!')

        for img in img_names:
            self._items.append((os.path.join(img_dir, img), gt[img]))

    def __len__(self):
        return len(self._items)

    def __getitem__(self, index):
        img_path, facepoints = self._items[index]
        facepoints = np.array(facepoints).astype(np.float32)

        image = imread(img_path)
        image = np.array(image).astype(np.float32)
        x, y = image.shape[1], image.shape[0]

        # resizing
        size = 100
        image = cv2.resize(image, (size,size))
        facepoints[::2] = facepoints[::2] * (size / x)
        facepoints[1::2] = facepoints[1::2] * (size / y)

        # single channel image
        if len(image.shape) == 2:
            image = np.stack([image]*3).transpose(1, 2, 0)

        # augmentation
        if self._transform:
            keypoints = list(zip(facepoints[::2], facepoints[1::2]))
            transformed = self._transform(image=image, keypoints=keypoints)
            image = transformed['image']
            facepoints = np.ravel(transformed['keypoints']).astype(np.float32)

        image = torch.from_numpy(image.transpose(2, 0, 1))
        facepoints = torch.from_numpy(facepoints)

        # normalization
        mean = image.mean([1,2]).reshape(3,1,1)
        std = image.std([1,2]).reshape(3,1,1)
        image = (image - mean) / std

        return image, facepoints


class FaceModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        """Define computations here"""
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv5 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv6 = nn.Conv2d(128, 128, 3, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(32)
        self.batchnorm2 = nn.BatchNorm2d(32)
        self.batchnorm3 = nn.BatchNorm2d(64)
        self.batchnorm4 = nn.BatchNorm2d(64)
        self.batchnorm5 = nn.BatchNorm2d(128)
        self.batchnorm6 = nn.BatchNorm2d(128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128*12*12, 128)
        self.fc2 = nn.Linear(128, 28)
    
    def forward(self, x):
        """ Use for inference only (separate from training_step)"""
        x = self.batchnorm1(self.relu(self.conv1(x)))
        x = self.pool(self.batchnorm2(self.relu(self.conv2(x))))
        x = self.batchnorm3(self.relu(self.conv3(x)))
        x = self.pool(self.batchnorm4(self.relu(self.conv4(x))))
        x = self.batchnorm5(self.relu(self.conv5(x)))
        x = self.pool(self.batchnorm6(self.relu(self.conv6(x))))
        x = torch.flatten(x, 1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x
    
    def training_step(self, batch, batch_idx):
        """The full training loop"""
        x, y = batch
        loss = F.mse_loss(self(x), y)
        return {'loss': loss}
    
    def configure_optimizers(self):
        """Define optimizers and LR schedulers"""
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=5e-4)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                                  mode='min', 
                                                                  factor=0.2, 
                                                                  patience=3, 
                                                                  verbose=True)
        lr_dict = {
            "scheduler": lr_scheduler,
            "interval": "epoch",
            "frequency": 1,
            "monitor": "val_loss"
        } 
        return [optimizer], [lr_dict]
    
    def validation_step(self, batch, batch_idx):
        """The full validation loop"""
        x, y = batch
        loss = F.mse_loss(self(x), y)
        return {'val_loss': loss}

    def training_epoch_end(self, outputs):
        """Log and display average train loss across epoch"""
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        print(f"| Train_loss: {avg_loss:.2f}" )
        self.log('train_loss', avg_loss, prog_bar=True, on_epoch=True, on_step=False)
     
    def validation_epoch_end(self, outputs):
        """Log and display average val loss"""
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        print(f"[Epoch {self.trainer.current_epoch:3}] Val_loss: {avg_loss:.2f}", end= " ")
        self.log('val_loss', avg_loss, prog_bar=True, on_epoch=True, on_step=False)


def train_detector(train_gt: dict, 
                   train_img_dir: str,
                   fast_train=True):
    '''Train model on train (with fast_train <5 min)'''
    MyTransform = A.Compose([
        A.Rotate(limit=70, p=0.3),
    ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

    ds_train = FacepointsDataset(mode="train", gt=train_gt, img_dir=train_img_dir, transform=MyTransform)
    ds_val = FacepointsDataset(mode="val", gt=train_gt, img_dir=train_img_dir)

    dl_train = DataLoader(ds_train, batch_size=32, shuffle=True, num_workers=2)
    dl_val = DataLoader(ds_val, batch_size=16, shuffle=False, num_workers=2)

    model = FaceModel()

    if fast_train:
        trainer = pl.Trainer(max_epochs=1,
                             accelerator='cpu', 
                             devices=1,
                             logger=False,
                             enable_checkpointing=False)
        trainer.fit(model, dl_train, dl_val)
    else:
        checkpoint = ModelCheckpoint(dirpath='pl_model',
                                     filename='{epoch}-{val_loss:.3f}',
                                     monitor='val_loss',
                                     mode='min',
                                     save_top_k=1)
        earlystopping = EarlyStopping(monitor="val_loss",
                                      mode="min",
                                      patience=8,
                                      verbose=True)
        trainer = pl.Trainer(max_epochs=50,
                             accelerator='gpu', 
                             devices=1,
                             logger=True,
                             enable_checkpointing=True,
                             callbacks=[checkpoint, earlystopping])
        trainer.fit(model, dl_train, dl_val)
        torch.save(model.to('cpu').state_dict(), "facepoints_model.ckpt")

    return model


def detect(model_filename: str, 
           test_img_dir: str):
    '''
    Returns dict with predictions sized N: 
        {image_1: [x_1, y_1, ..., x_14, y_14], 
         image_2: ...}
    '''
    model = FaceModel()
    model.load_state_dict(torch.load(model_filename))
    model.eval()
    imgs_names = sorted(os.listdir(test_img_dir))
    predictions = {}

    # batch size = 1
    for img_name in imgs_names:
        img_path = os.path.join(test_img_dir, img_name)
        image = imread(img_path)
        image = np.array(image).astype(np.float32)
        x, y = image.shape[1], image.shape[0]

        # resizing
        size = 100
        image = cv2.resize(image, (size,size))
        
        # single channel image
        if len(image.shape) == 2:
            image = np.stack([image]*3).transpose(1, 2, 0)
        
        # normalization
        image = torch.from_numpy(image.transpose(2, 0, 1))
        mean = image.mean([1,2]).reshape(3,1,1)
        std = image.std([1,2]).reshape(3,1,1)
        image = (image - mean) / std
        
        # prediction
        image = image.unsqueeze(0)
        pred = model(image).detach().numpy()[0]
        pred[::2] = pred[::2] * (x / size)
        pred[1::2] = pred[1::2] * (y / size)
        predictions.update({img_name: pred})

    return predictions
