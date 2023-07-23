import os
import numpy as np
from itertools import chain

import cv2
import albumentations as A
from skimage.io import imread

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint


RGB_MEAN = (125.005, 130.891, 119.178)
RGB_STD = (46.192, 46.081, 48.974)


class BirdsDataset(Dataset):
    def __init__(self, 
                 mode: str, 
                 gt: dict,
                 img_dir: str,
                 fraction: float = 0.8, 
                 transform = None,
                 normalize = True,
                ):
        self._items = []
        self._transform = transform
        self._normalize = normalize
        
        # invert to {0: [1.jpg, 2.jpg, ...]}
        inv_gt = {}
        for k, v in gt.items():
            inv_gt[v] = inv_gt.get(v, []) + [k]

        split = int(fraction * len(inv_gt[next(iter(inv_gt))]))
        if mode == 'train':
            img_names = chain(*[imgs[:split] for imgs in inv_gt.values()])
        elif mode == 'val':
            img_names = chain(*[imgs[split:] for imgs in inv_gt.values()])
        elif mode == 'sample':
            # 2 images per class, 100 in total, for fast_train=True
            img_names = chain(*[imgs[:2] for imgs in inv_gt.values()])
        else:
            raise ValueError(f'Wrong mode = {mode}!')

        for img in img_names:
            self._items.append((os.path.join(img_dir, img), gt[img]))

    def __len__(self):
        return len(self._items)

    def __getitem__(self, index):
        img_path, label = self._items[index]

        image = imread(img_path)
        image = np.array(image).astype(np.float32)

        # resizing to minimal side = 300 and crop center to square 300x300
        h, w = image.shape[0], image.shape[1]
        if h < w:
            h_new, w_new = 300, int(300 * w / h)
        else:
            h_new, w_new = int(300 * h / w), 300
        image = cv2.resize(image, (w_new, h_new))
        image = A.CenterCrop(width=300, height=300)(image=image)['image']

        # single channel image
        if len(image.shape) == 2:
            image = np.dstack([image]*3)

        # augmentation
        if self._transform:
            image = self._transform(image=image)['image']

        # normalize and to tensor
        if self._normalize:
            image = A.Normalize(mean=RGB_MEAN, std=RGB_STD)(image=image)['image']
        image = torch.from_numpy(image.transpose(2, 0, 1))

        return image, label
    

class EfficientNetB3Classifier(pl.LightningModule):
    def __init__(self, num_classes, pretrained=False, train_last_n_layers=1):
        super().__init__()        

        self.model = models.efficientnet_b3(pretrained=pretrained)
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, num_classes)

        # Unfreeze all layers
        for child in list(self.model.children()):
            for param in child.parameters():
                param.requires_grad = True

        # Freeze all except last N
        for child in list(self.model.children())[:-train_last_n_layers]:
            for param in child.parameters():
                param.requires_grad = False
        
    def forward(self, x):
        x = self.model(x)
        return F.log_softmax(x, dim=1)

    
class BirdsModel(pl.LightningModule):
    def __init__(self, pretrained=False, train_last_n_layers=1):
        super().__init__()
        self.model = EfficientNetB3Classifier(50, pretrained, train_last_n_layers)
        
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = F.nll_loss(logits, y)
        acc = torch.sum(logits.argmax(dim=1) == y) / y.shape[0]
        self.log('train_loss', loss, on_step=True, prog_bar=True)
        self.log('train_acc', acc, on_step=True, prog_bar=True)
        logs = {'train_loss': loss, 'train_acc': acc}
        return {'loss': loss, 'log': logs}
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = F.nll_loss(logits, y)
        acc = torch.sum(logits.argmax(dim=1) == y) / y.shape[0]
        self.log('val_loss', loss, on_step=True, on_epoch=False)
        self.log('val_acc', acc, on_step=True, on_epoch=False)
        return {'val_loss': loss, 'val_acc': acc}

    def training_epoch_end(self, outputs):
        avg_acc = torch.stack([x['log']['train_acc'] for x in outputs]).mean()
        print(f'| Accuracy (train): {avg_acc:.2f}', end= " ")
        self.log('train_acc', avg_acc, on_epoch=True, on_step=False)

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['val_acc'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss, 'val_acc': avg_acc}
        print(f'[Epoch {self.trainer.current_epoch:2}] Accuracy (val): {avg_acc:.2f}', end= " ")
        self.log('val_acc', avg_acc, on_epoch=True, on_step=False)
        return {'val_loss': avg_loss, 'val_acc': avg_acc, 'log': tensorboard_logs}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,
                                                              gamma=0.96,
                                                              verbose=True)
        lr_dict = {
            "scheduler": lr_scheduler,
            "interval": "epoch",
            "frequency": 3,
            "monitor": "val_acc"
        } 
        return [optimizer], [lr_dict]

    
def train_classifier(train_gt: dict, 
                     train_img_dir: str,
                     fast_train=True):
    MyTransform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=30, p=0.5),
        A.RandomBrightnessContrast(p=0.3),
    ])

    ds_train = BirdsDataset(mode="train", gt=train_gt, img_dir=train_img_dir, transform=MyTransform)
    ds_val = BirdsDataset(mode="val", gt=train_gt, img_dir=train_img_dir)
    ds_sample = BirdsDataset(mode="sample", gt=train_gt, img_dir=train_img_dir)

    dl_train = DataLoader(ds_train, batch_size=64, shuffle=True, num_workers=2)
    dl_val = DataLoader(ds_val, batch_size=32, shuffle=False, num_workers=2)
    dl_sample = DataLoader(ds_sample, batch_size=1, shuffle=False, num_workers=1)

    model = BirdsModel(pretrained=not fast_train, train_last_n_layers=3)

    if fast_train:
        trainer = pl.Trainer(max_epochs=1,
                             accelerator='cpu', 
                             devices=1,
                             logger=False,
                             enable_checkpointing=False)
        trainer.fit(model, dl_sample, dl_sample)
    else:
        checkpoint = ModelCheckpoint(dirpath='pl_model',
                                     filename='{epoch}-{val_acc:.3f}',
                                     monitor='val_acc',
                                     mode='max',
                                     save_top_k=1)
        earlystopping = EarlyStopping(monitor="val_acc",
                                      mode="max",
                                      patience=10,
                                      verbose=True)
        trainer = pl.Trainer(max_epochs=100,
                             accelerator='gpu', 
                             devices=1,
                             precision=16,
                             logger=True,
                             enable_checkpointing=True,
                             callbacks=[checkpoint, earlystopping])
        trainer.fit(model, dl_train, dl_val)
        torch.save(model.to('cpu').state_dict(), "birds_model.ckpt")

    return model


def classify(model_path: str, 
             test_img_dir: str):
    '''
    Returns dict with predictions sized N: 
        {image_1: label, 
         image_2: ...}
    '''
    model = BirdsModel(pretrained=False)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    imgs_names = sorted(os.listdir(test_img_dir))
    predictions = {}

    # batch size = 1
    for img_name in imgs_names:
        img_path = os.path.join(test_img_dir, img_name)
        image = imread(img_path)
        image = np.array(image).astype(np.float32)

        # resizing to minimal side = 300 and crop center to square 300x300
        h, w = image.shape[0], image.shape[1]
        if h < w:
            h_new, w_new = 300, int(300 * w / h)
        else:
            h_new, w_new = int(300 * h / w), 300
        image = cv2.resize(image, (w_new, h_new))
        image = A.CenterCrop(width=300, height=300)(image=image)['image']

        # single channel image
        if len(image.shape) == 2:
            image = np.dstack([image]*3)
        
        # normalize and to tensor
        image = A.Normalize(mean=RGB_MEAN, std=RGB_STD)(image=image)['image']
        image = torch.from_numpy(image.transpose(2, 0, 1))
        
        # prediction
        image = image.unsqueeze(0)
        pred = model(image).argmax(dim=1).detach().numpy()[0]
        predictions.update({img_name: pred})

    return predictions
