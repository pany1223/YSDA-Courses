# -*- coding: utf-8 -*-
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import models
import pytorch_lightning as pl
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image

import os
import csv
import json
import tqdm
import pickle
import typing
import random
from collections import OrderedDict
import matplotlib.pyplot as plt

import numpy as np
from concurrent.futures import ProcessPoolExecutor
from sklearn.neighbors import KNeighborsClassifier


CLASSES_CNT = 205


def calc_metric(y_true, y_pred, cur_type, class_name_to_type):
    """
    cur_type ~ ['all', 'rare', 'freq']
    """
    ok_cnt = 0
    all_cnt = 0
    for t, p in zip(y_true, y_pred):
        if cur_type == 'all' or class_name_to_type[t] == cur_type:
            all_cnt += 1
            if t == p:
                ok_cnt += 1
    return ok_cnt / max(1, all_cnt)


class DatasetRTSD(torch.utils.data.Dataset):
    """
    Класс для чтения и хранения датасета.
    :param root_folders: список путей до папок с данными
    :param path_to_classes_json: путь до classes.json
    """
    def __init__(self, root_folders, path_to_classes_json) -> None:
        super(DatasetRTSD, self).__init__()
        self.classes, self.class_to_idx = self.get_classes(path_to_classes_json)
        
        ### YOUR CODE HERE - список пар (путь до картинки, индекс класса)
        self.samples = []
        for root in root_folders:
            class_folders = os.listdir(root)
            for class_name in class_folders:
                path = os.path.join(root, class_name)
                self.samples += [(os.path.join(path, img), self.class_to_idx[class_name]) for img in os.listdir(path)]
        
        ### YOUR CODE HERE - cловарь из списков картинок для каждого класса, 
        ### classes_to_samples[индекс класса] = [список чисел-позиций картинок в self.samples]
        self.classes_to_samples = {c: [] for c in range(len(self.classes))}
        for i, (_, v) in enumerate(self.samples):
            self.classes_to_samples[v] += [i]
        
        ### YOUR CODE HERE - аугментации + нормализация + ToTensorV2
        self.transform = A.Compose([
            A.Resize(64, 64, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=30, p=0.5),
            A.RGBShift(p=0.5),
            A.Normalize(),
            ToTensorV2()
        ])

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        """
        Возвращает тройку: тензор с картинкой, путь до файла, номер класса файла (если нет разметки, то "-1").
        """
        ### YOUR CODE HERE
        img_path, class_index = self.samples[index]
        image = Image.open(img_path).convert('RGB')
        image = np.array(image).astype(np.float32)
        image = self.transform(image=image)['image']
        return image, img_path, class_index
    
    @staticmethod
    def get_classes(path_to_classes_json):
        """
        Считывает из classes.json информацию о классах.
        :param path_to_classes_json: путь до classes.json
        """
        with open(path_to_classes_json) as j:
            data = json.load(j)

        ### YOUR CODE HERE - словарь, class_to_idx['название класса'] = индекс
        class_to_idx = {k: v['id'] for k, v in data.items()} 

        ### YOUR CODE HERE - массив, classes[индекс] = 'название класса'
        classes = list(class_to_idx.keys())
        return classes, class_to_idx


class TestData(torch.utils.data.Dataset):
    """
    Класс для чтения и хранения тестового датасета.
    :param root: путь до папки с картинками знаков
    :param path_to_classes_json: путь до classes.json
    :param annotations_file: путь до .csv-файла с аннотациями (опциональный)
    """
    def __init__(self, root, path_to_classes_json, annotations_file=None):
        super(TestData, self).__init__()
        self.root = root
        
        ### YOUR CODE HERE - список путей до картинок
        self.samples = [img for img in os.listdir(root)]
        
        ### YOUR CODE HERE - преобразования: ресайз + нормализация + ToTensorV2
        self.transform = A.Compose([
            A.Resize(64, 64, p=1.0),
            A.Normalize(),
            ToTensorV2()
        ])
        
        ### YOUR CODE HERE - словарь, targets[путь до картинки] = индекс класса
        self.targets = None
        if annotations_file is not None:
            with open(path_to_classes_json) as j:
                data = json.load(j)
            class_to_idx = {k: v['id'] for k, v in data.items()} 
            
            self.targets = {}
            with open(annotations_file) as f:
                next(f)
                for line in f:
                    filename, class_name = line.rstrip('\n').split(',')
                    self.targets[filename] = class_to_idx[class_name]

    def __len__(self):
        return len(self.samples)                
                    
    def __getitem__(self, index):
        """
        Возвращает тройку: тензор с картинкой, путь до файла, номер класса файла (если нет разметки, то "-1").
        """
        ### YOUR CODE HERE
        filename = self.samples[index]
        img_path = os.path.join(self.root, filename)
        if self.targets:
            class_index = self.targets.get(filename, -1)
        else:
            class_index = -1
        image = Image.open(img_path).convert('RGB')
        image = np.array(image).astype(np.float32)
        image = self.transform(image=image)['image']
        return image, filename, class_index


class CustomNetwork(pl.LightningModule):
    """
    Класс, реализующий нейросеть для классификации.
    :param features_criterion: loss-функция на признаки, извлекаемые нейросетью перед классификацией (None когда нет такого лосса)
    :param internal_features: внутреннее число признаков
    """
    def __init__(self, features_criterion=None, internal_features=1024, pretrained=False):
        super(CustomNetwork, self).__init__()
        ### YOUR CODE HERE
        self.model = models.resnet50(pretrained=pretrained)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(in_features, internal_features),
            nn.ReLU(),
            nn.Linear(internal_features, CLASSES_CNT)
        )

        # Freeze all layers except last 4 layers
        for child in list(self.model.children()):
            for param in child.parameters():
                param.requires_grad = True
        for child in list(self.model.children())[:-4]:
            for param in child.parameters():
                param.requires_grad = False

    def forward(self, x):
        out = self.model(x)
        return F.log_softmax(out, dim=1)
    
    def training_step(self, batch):
        image, _, label = batch
        logits = self.forward(image)
        loss = F.nll_loss(logits, label)
        acc = (logits.argmax(dim=1) == label).sum().item() / len(logits)
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-5)
        return [optimizer]
        
    def predict(self, x):
        """
        Функция для предсказания классов-ответов. Возвращает np-массив с индексами классов.
        :param x: батч с картинками
        """
        ### YOUR CODE HERE
        out = self.forward(x)
        labels = out.argmax(dim=1).detach().cpu().numpy()
        return labels


def train_simple_classifier():
    """Функция для обучения простого классификатора на исходных данных."""
    ### YOUR CODE HERE
    ds_train = DatasetRTSD(root_folders=['./cropped-train'], path_to_classes_json='./classes.json')
    dl_train = DataLoader(ds_train, batch_size=32, shuffle=True, num_workers=2)
    
    trainer = pl.Trainer(max_epochs=3,
                         precision=16,
                         accelerator='gpu', 
                         devices=1,
                         logger=False,
                         enable_checkpointing=False)
    
    model = CustomNetwork(pretrained=True)
    trainer.fit(model, dl_train)
    torch.save(model.to('cpu').state_dict(), 'simple_model.pth')

    return model


def apply_classifier(model, test_folder, path_to_classes_json):
    """
    Функция, которая применяет модель и получает её предсказания.
    :param model: модель, которую нужно протестировать
    :param test_folder: путь до папки с тестовыми данными
    :param path_to_classes_json: путь до файла с информацией о классах classes.json
    """
    ### YOUR CODE HERE - список словарей вида {'filename': 'имя файла', 'class': 'строка-название класса'}
    results = []

    ds_test = TestData(test_folder, path_to_classes_json)
    dl_test = DataLoader(ds_test, batch_size=1, shuffle=False)
    classes, class_to_idx = DatasetRTSD.get_classes(path_to_classes_json)

    model.eval()
    for image, img_path, _ in dl_test:
        pred = model.predict(image)[0]
        filename = img_path[0].split('/')[-1]
        results.append({'filename': filename, 'class': classes[pred]})

    return results


def test_classifier(model, test_folder, path_to_classes_json, annotations_file):
    """
    Функция для тестирования качества модели.
    Возвращает точность на всех знаках, Recall на редких знаках и Recall на частых знаках.
    :param model: модель, которую нужно протестировать
    :param test_folder: путь до папки с тестовыми данными
    :param annotations_file: путь до .csv-файла с аннотациями (опциональный)
    """
    ### YOUR CODE HERE
    # predicted labels [{'0.png': '5.12.1'}, {...}]
    predicted = apply_classifier(model, test_folder, path_to_classes_json)
    pred = {d['filename']: d['class'] for d in predicted}

    # class name to type {'1.1': 'freq', ...}
    with open(path_to_classes_json) as j:
        data = json.load(j)
    class_name_to_type = {k: v['type'] for k, v in data.items()} 

    # true labels {'0.png': '5.12.1', ...}
    with open(annotations_file) as f:
        next(f)
        true = {}
        for line in f:
            filename, class_name = line.rstrip('\n').split(',')
            true[filename] = class_name

    # true/pred
    images = list(true.keys())
    y_true = [true[img] for img in images]
    y_pred = [pred[img] for img in images]

    # metrics
    total_acc = calc_metric(y_true, y_pred, 'all', class_name_to_type)
    rare_recall = calc_metric(y_true, y_pred, 'rare', class_name_to_type)
    freq_recall = calc_metric(y_true, y_pred, 'freq', class_name_to_type)

    return total_acc, rare_recall, freq_recall


class SignGenerator(object):
    """
    Класс для генерации синтетических данных.
    :param background_path: путь до папки с изображениями фона
    """
    def __init__(self, background_path):
        ### YOUR CODE HERE
        self.backgrounds = []
        for bg in os.listdir(background_path):
            self.backgrounds.append(os.path.join(background_path, bg))

    def __len__(self):
        return len(self.backgrounds) 

    def get_sample(self, icon):
        """
        Функция, встраивающая иконку на случайное изображение фона.
        :param icon: Массив с изображением иконки
        """
        ### YOUR CODE HERE
        icon = np.array(icon.convert("RGBA"))

        # 1. random resize
        size = random.randint(16, 128)
        icon = cv2.resize(icon, (size, size))

        # 2. padding
        pad = int(size * random.uniform(0.0, 0.15)) + 1
        padded = size + 2 * pad
        size += 2 * pad
        icon_padded = np.zeros((padded, padded, 4), dtype='uint8')
        icon_padded[pad:-pad, pad:-pad] = icon
        icon = icon_padded

        # 3. random color change
        icon_hsv = cv2.cvtColor(icon[..., :3], cv2.COLOR_RGB2HSV)
        icon_hsv[...,0] = np.clip(icon_hsv[...,0] + np.random.randint(-16, 16), 0, 179)
        icon_hsv[...,1] = np.clip(icon_hsv[...,1] + np.random.randint(-16, 16), 0, 255)
        icon_hsv[...,2] = np.clip(icon_hsv[...,2] + np.random.randint(-16, 16), 0, 255)
        icon[..., :3] = cv2.cvtColor(icon_hsv, cv2.COLOR_HSV2RGB)

        # 4. random rotation
        center = tuple(np.array(icon.shape[1::-1]) / 2)
        rotation = cv2.getRotationMatrix2D(center=center, 
                                           angle=random.randint(-15, 15), 
                                           scale=1.0)
        icon = cv2.warpAffine(icon, rotation, icon.shape[1::-1], flags=cv2.INTER_LINEAR)

        # 5. blur in motion
        kernel_size = random.randint(3, 9)
        kernel = np.zeros((kernel_size, kernel_size))
        kernel[(kernel_size-1)//2, :] = np.ones(kernel_size)
        kernel /= kernel_size
        icon[...,:3] = cv2.filter2D(src=icon[...,:3], 
                                    ddepth=-1, 
                                    kernel=kernel)

        # 6. Gauss filter
        icon[...,:3] = cv2.GaussianBlur(icon[...,:3], (3,3), 10)

        # 7. embedding into background
        bg_path = random.choice(self.backgrounds)
        bg = np.array(Image.open(bg_path).convert("RGB"))
        bg_h, bg_w = bg.shape[:2]
        h_crop = random.randint(0, bg_h - size - 1)
        w_crop = random.randint(0, bg_w - size - 1)
        crop = bg[h_crop:h_crop+size, w_crop:w_crop+size]
        mask = icon[...,3].astype('float32') / 255.
        mask = np.dstack([mask]*3)
        image = (mask * icon[...,:3] + (1 - mask) * crop).astype('uint8')
        return image


def generate_one_icon(args):
    """
    Функция, генерирующая синтетические данные для одного класса.
    :param args: Это список параметров: [путь до файла с иконкой, путь до выходной папки, путь до папки с фонами, число примеров каждого класса]
    """
    ### YOUR CODE HERE
    icon_path, output_folder, background_path, samples_per_class = args
    class_name = icon_path.split('/')[-1][:-4]
    sg = SignGenerator(background_path)
    for i in range(samples_per_class):
        image = Image.open(icon_path)
        sample = sg.get_sample(image)
        path = os.path.join(output_folder, class_name, f'{class_name}-{i}.png')
        cv2.imwrite(path, cv2.cvtColor(sample, cv2.COLOR_RGB2BGR))


def generate_all_data(output_folder, icons_path, background_path, samples_per_class=1000):
    """
    Функция, генерирующая синтетические данные.
    Эта функция запускает пул параллельно работающих процессов, каждый из которых будет генерировать иконку своего типа.
    Это необходимо, так как процесс генерации очень долгий.
    Каждый процесс работает в функции generate_one_icon.
    :param output_folder: Путь до выходной директории
    :param icons_path: Путь до директории с иконками
    :param background_path: Путь до директории с картинками фона
    :param samples_per_class: Количество примеров каждого класса, которые надо сгенерировать
    """
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)
    for icon in os.listdir(icons_path):
        icon_folder = os.path.join(output_folder, icon[:-4])
        if not os.path.isdir(icon_folder):
            os.mkdir(icon_folder)
    with ProcessPoolExecutor(8) as executor:
        params = [[os.path.join(icons_path, icon_file), output_folder, background_path, samples_per_class]
                  for icon_file in os.listdir(icons_path)]
        list(tqdm.tqdm(executor.map(generate_one_icon, params)))


def train_synt_classifier():
    """Функция для обучения простого классификатора на смеси исходных и синтетических данных."""
    ### YOUR CODE HERE
    ds_train = DatasetRTSD(root_folders=['./cropped-train', './synthetic-train'], path_to_classes_json='./classes.json')
    dl_train = DataLoader(ds_train, batch_size=32, shuffle=True, num_workers=2)

    trainer = pl.Trainer(max_epochs=1,
                         precision=16,
                         accelerator='gpu', 
                         devices=1,
                         logger=False,
                         enable_checkpointing=False)
    
    model = CustomNetwork(pretrained=True)
    trainer.fit(model, dl_train)
    torch.save(model.to('cpu').state_dict(), "simple_model_with_synt.pth")
    return model


class FeaturesLoss(torch.nn.Module):
    """
    Класс для вычисления loss-функции на признаки предпоследнего слоя нейросети.
    """
    def __init__(self, margin: float) -> None:
        super(FeaturesLoss, self).__init__()
        ### YOUR CODE HERE
        pass


class CustomBatchSampler(torch.utils.data.sampler.Sampler[typing.List[int]]):
    """
    Класс для семплирования батчей с контролируемым числом классов и примеров каждого класса.
    :param data_source: Это датасет RTSD
    :param elems_per_class: Число элементов каждого класса
    :param classes_per_batch: Количество различных классов в одном батче
    """
    def __init__(self, data_source, elems_per_class, classes_per_batch):
        ### YOUR CODE HERE
        pass
    def __iter__(self):
        ### YOUR CODE HERE
        pass


def train_better_model():
    """Функция для обучения классификатора на смеси исходных и ситетических данных с новым лоссом на признаки."""
    ### YOUR CODE HERE
    return model


class ModelWithHead:
    """
    Класс, реализующий модель с головой из kNN.
    :param n_neighbors: Количество соседей в методе ближайших соседей
    """
    def __init__(self, n_neighbors):
        ### YOUR CODE HERE
        pass

    def load_nn(self, nn_weights_path):
        """
        Функция, загружающая веса обученной нейросети.
        :param nn_weights_path: Это путь до весов обученной нейросети с улучшенными признаками на предпоследнем слое
        """
        ### YOUR CODE HERE
        pass

    def load_head(self, knn_path):
        """
        Функция, загружающая веса kNN (с помощью pickle).
        :param knn_path: Путь, откуда надо прочитать веса kNN
        """
        ### YOUR CODE HERE
        pass

    def predict(self, imgs):
        """
        Функция для предсказания классов-ответов. Возвращает np-массив с индексами классов.
        :param imgs: батч с картинками
        """
        features, model_pred = ... ### YOUR CODE HERE - предсказание нейросетевой модели
        features = features / np.linalg.norm(features, axis=1)[:, None]
        knn_pred = ... ### YOUR CODE HERE - предсказание kNN на features
        return knn_pred


class IndexSampler(torch.utils.data.sampler.Sampler[int]):
    """
    Класс для семплирования батчей с картинками индекса.
    :param data_source: Это датасет RTSD с синтетическими примерами
    :param examples_per_class: Число элементов каждого класса, которые должны попасть в индекс
    """
    def __init__(self, data_source, examples_per_class) -> None:
        ### YOUR CODE HERE
        pass
    def __iter__(self):
        """Функция, которая будет генерировать список индексов элементов в батче."""
        return ### YOUR CODE HERE


def train_head(nn_weights_path, examples_per_class = 20):
    """
    Функция для обучения kNN-головы классификатора.
    :param nn_weights_path: Это путь до весов обученной нейросети с улучшенными признаками на предпоследнем слое
    :param examples_per_class: Число элементов каждого класса, которые должны попасть в индекс
    """
    ### YOUR CODE HERE
