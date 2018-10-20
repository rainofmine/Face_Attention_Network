from __future__ import print_function, division
import sys
import os
import torch
import pandas as pd
import numpy as np
import random
import csv
import time
import cv2

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.utils.data.sampler import Sampler


import skimage.io
import skimage.transform
import skimage.color
import skimage

from PIL import Image, ImageEnhance, ImageFilter


class CSVDataset(Dataset):
    """CSV dataset."""

    def __init__(self, train_file, class_list, transform=None):
        """
        Args:
            train_file (string): CSV file with training annotations
            annotations (string): CSV file with class list
            test_file (string, optional): CSV file with testing annotations
        """
        self.train_file = train_file
        self.class_list = class_list
        self.transform = transform

        # parse the provided class file
        try:
            with self._open_for_csv(self.class_list) as file:
                self.classes = self.load_classes(csv.reader(file, delimiter=' '))
        except ValueError as e:
            raise(ValueError('invalid CSV class file: {}: {}'.format(self.class_list, e)), None)

        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

        # csv with img_path, x1, y1, x2, y2, class_name
        try:
            with self._open_for_csv(self.train_file) as file:
                self.image_data = self._read_annotations(csv.reader(file, delimiter=' '), self.classes)
        except ValueError as e:
            raise(ValueError('invalid CSV annotations file: {}: {}'.format(self.train_file, e)), None)
        self.image_names = list(self.image_data.keys())


    def _parse(self, value, function, fmt):
        """
        Parse a string into a value, and format a nice ValueError if it fails.
        Returns `function(value)`.
        Any `ValueError` raised is catched and a new `ValueError` is raised
        with message `fmt.format(e)`, where `e` is the caught `ValueError`.
        """
        try:
            return function(value)
        except ValueError as e:
            raise(ValueError(fmt.format(e)), None)

    def _open_for_csv(self, path):
        """
        Open a file with flags suitable for csv.reader.
        This is different for python2 it means with mode 'rb',
        for python3 this means 'r' with "universal newlines".
        """
        if sys.version_info[0] < 3:
            return open(path, 'rb')
        else:
            return open(path, 'r', newline='')


    def load_classes(self, csv_reader):
        result = {}

        for line, row in enumerate(csv_reader):
            line += 1

            try:
                class_name, class_id = row
            except ValueError:
                raise(ValueError('line {}: format should be \'class_name,class_id\''.format(line)), None)
            class_id = self._parse(class_id, int, 'line {}: malformed class ID: {{}}'.format(line))

            if class_name in result:
                raise ValueError('line {}: duplicate class name: \'{}\''.format(line, class_name))
            result[class_name] = class_id
        return result


    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):

        img = self.load_image(idx)
        annot = self.load_annotations(idx)
        sample = {'img': img, 'annot': annot, 'scale': 1}
        if self.transform:
            sample = self.transform(sample)

        return sample

    def load_image(self, image_index):
        
        #img = skimage.io.imread(self.image_names[image_index])
        img = cv2.imread(self.image_names[image_index])
        b,g,r = cv2.split(img)
        img = cv2.merge([r,g,b])
       

        if len(img.shape) == 2:
            img = skimage.color.gray2rgb(img)

        return img.astype(np.float32)/255.0

    def load_annotations(self, image_index):
        # get ground truth annotations
        annotation_list = self.image_data[self.image_names[image_index]]
        annotations     = np.zeros((0, 5))

        # some images appear to miss annotations (like image with id 257034)
        if len(annotation_list) == 0:
            return annotations

        # parse annotations
        for idx, a in enumerate(annotation_list):
            # some annotations have basically no width / height, skip them
            x1 = a['x1']
            x2 = a['x2']
            y1 = a['y1']
            y2 = a['y2']

            if (x2-x1) < 1 or (y2-y1) < 1:
                continue

            annotation        = np.zeros((1, 5))
            
            annotation[0, 0] = x1
            annotation[0, 1] = y1
            annotation[0, 2] = x2
            annotation[0, 3] = y2

            annotation[0, 4]  = self.name_to_label(a['class'])
            annotations       = np.append(annotations, annotation, axis=0)

        return annotations

    def _read_annotations(self, csv_reader, classes):
        result = {}
        for line, row in enumerate(csv_reader):
            line += 1

            try:
                img_file, x1, y1, x2, y2, class_name = row[:6]
            except ValueError:
                raise(ValueError('line {}: format should be \'img_file,x1,y1,x2,y2,class_name\' or \'img_file,,,,,\''.format(line)), None)

            if img_file not in result:
                result[img_file] = []

            # If a row contains only an image path, it's an image without annotations.
            if (x1, y1, x2, y2, class_name) == ('.', '.', '.', '.', '.'):
                continue

            x1 = self._parse(x1, int, 'line {}: malformed x1: {{}}'.format(line))
            y1 = self._parse(y1, int, 'line {}: malformed y1: {{}}'.format(line))
            x2 = self._parse(x2, int, 'line {}: malformed x2: {{}}'.format(line))
            y2 = self._parse(y2, int, 'line {}: malformed y2: {{}}'.format(line))

            if class_name !='ignore':
                # Check that the bounding box is valid.
                if x2 <= x1:
                    raise ValueError('line {}: x2 ({}) must be higher than x1 ({})'.format(line, x2, x1))
                if y2 <= y1:
                    raise ValueError('line {}: y2 ({}) must be higher than y1 ({})'.format(line, y2, y1))

                # check if the current class name is correctly present
                if class_name not in classes:
                    raise ValueError('line {}: unknown class name: \'{}\' (classes: {})'.format(line, class_name, classes))

            result[img_file].append({'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2, 'class': class_name})
        return result

    def name_to_label(self, name):
        return self.classes[name]

    def label_to_name(self, label):
        return self.labels[label]

    def num_classes(self):
        return max(self.classes.values()) + 1

    def image_aspect_ratio(self, image_index):
        image = Image.open(self.image_names[image_index])
        return float(image.width) / float(image.height)


def collater(data):

    imgs = [s['img'] for s in data]
    annots = [s['annot'] for s in data]
    scales = [s['scale'] for s in data]
        
    widths = [int(s.shape[0]) for s in imgs]
    heights = [int(s.shape[1]) for s in imgs]
    batch_size = len(imgs)

    max_width = np.array(widths).max()
    max_height = np.array(heights).max()

    padded_imgs = torch.zeros(batch_size, max_width, max_height, 3)

    for i in range(batch_size):
        img = imgs[i]
        padded_imgs[i, :int(img.shape[0]), :int(img.shape[1]), :] = img

    max_num_annots = max(annot.shape[0] for annot in annots)
    annot_padded = torch.ones((len(annots), max_num_annots, 5)) * -1
    #print(annot_padded.shape)
    if max_num_annots > 0:
        for idx, annot in enumerate(annots):
            #print(annot.shape)
            if annot.shape[0] > 0:
                annot_padded[idx, :annot.shape[0], :] = annot
    else:
        annot_padded = torch.ones((len(annots), 1, 5)) * -1


    padded_imgs = padded_imgs.permute(0, 3, 1, 2)

    return {'img': padded_imgs, 'annot': annot_padded, 'scale': scales}

class Resizer(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample, min_side=800, max_side=1400):
        
        image, annots, scale = sample['img'], sample['annot'], sample['scale']

        rows, cols, cns = image.shape

        #scale = min_side / rows


        smallest_side = min(rows, cols)

        # rescale the image so the smallest side is min_side
        scale = min_side / smallest_side

        # check if the largest side is now greater than max_side, which can happen
        # when images have a large aspect ratio
        largest_side = max(rows, cols)

        if largest_side * scale > max_side:
            scale = max_side / largest_side


        # resize the image with the computed scale

        image = cv2.resize(image, (int(round((cols*scale))), int(round((rows*scale)))))
        #image = skimage.transform.resize(image, (int(round(rows*scale)), int(round((cols*scale)))))
        
        rows, cols, cns = image.shape

        pad_w = 32 - rows%32
        pad_h = 32 - cols%32

        new_image = np.zeros((rows + pad_w, cols + pad_h, cns)).astype(np.float32)
        new_image[:rows, :cols, :] = image.astype(np.float32)

        annots[:, :4] *= scale
        
        return {'img': new_image, 'annot': annots, 'scale': scale}


class Augmenter(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample, flip_x=0.5):

        if np.random.rand() < flip_x:
            image, annots, scales = sample['img'], sample['annot'], sample['scale']
            image = image[:, ::-1, :]

            rows, cols, channels = image.shape

            x1 = annots[:, 0].copy()
            x2 = annots[:, 2].copy()
            
            x_tmp = x1.copy()

            annots[:, 0] = cols - x2
            annots[:, 2] = cols - x_tmp

            sample = {'img': image, 'annot': annots, 'scale': scales}

        return sample


class Random_crop(object):

    def __call__(self, sample):

        image, annots, scales = sample['img'], sample['annot'], sample['scale']

        if not annots.shape[0]:
            return {'img': image, 'annot': annots, 'scale': scales}
        if random.choice([0, 1]):
            return {'img': image, 'annot': annots, 'scale': scales}
        else:
            rows, cols, cns = image.shape
            flag = 0
            while True:
                flag += 1
                if flag > 10:
                    return {'img': image, 'annot': annots, 'scale': scales}

                crop_ratio = random.uniform(0.5, 1)
                rows_zero = int(rows * random.uniform(0, 1 - crop_ratio))
                cols_zero = int(cols * random.uniform(0, 1 - crop_ratio))
                crop_rows = int(rows * crop_ratio)
                crop_cols = int(cols * crop_ratio)
                '''
                new_image = image[rows_zero:rows_zero+crop_rows, cols_zero:cols_zero+crop_cols, :]
                new_image = cv2.resize(new_image, (cols, rows))
                #new_image = skimage.transform.resize(new_image, (rows, cols))

                new_annots = np.zeros((0, 5))
                for i in range(annots.shape[0]):
                    x1 = max(annots[i, 0] - cols_zero, 0)
                    y1 = max(annots[i, 1] - rows_zero, 0)
                    x2 = min(annots[i, 2] - cols_zero, crop_cols)
                    y2 = min(annots[i, 3] - rows_zero, crop_rows)
                    label = annots[i, 4]
                    if x1 + 10 < x2 and y1 + 10 < y2:
                        x1 /= crop_ratio
                        y1 /= crop_ratio
                        x2 /= crop_ratio
                        y2 /= crop_ratio
                        new_annots = np.append(new_annots, np.array([[x1, y1, x2, y2, label]]), axis=0)

                if not new_annots.shape[0]:
                    continue
                '''
                new_image = np.zeros((rows , cols , cns))
                new_image[rows_zero:rows_zero+crop_rows, cols_zero:cols_zero+crop_cols, :] = image[rows_zero:rows_zero+crop_rows, cols_zero:cols_zero+crop_cols, :]

                new_annots = np.zeros((0, 5))
                for i in range(annots.shape[0]):
                    x1 = max(cols_zero, annots[i, 0])
                    y1 = max(rows_zero, annots[i, 1])
                    x2 = min(cols_zero+crop_cols, annots[i, 2])
                    y2 = min(rows_zero+crop_rows, annots[i, 3])
                    label = annots[i, 4]
                    if x1+10 < x2 and y1+10 < y2:
                        new_annots = np.append(new_annots, np.array([[x1,y1,x2,y2,label]]), axis=0)
                
                if not new_annots.shape[0]:
                    continue

                return {'img': new_image, 'annot': new_annots,'scale': scales}


class Color(object):

    def __call__(self, sample):
        image, annots, scales = sample['img'], sample['annot'], sample['scale']
        image = Image.fromarray(image)

        ratio = [0.5, 0.8, 1.2, 1.5]

        if random.choice([0, 1]):
            enh_bri = ImageEnhance.Brightness(image)
            brightness = random.choice(ratio)
            image = enh_bri.enhance(brightness)
        if random.choice([0, 1]):
            enh_col = ImageEnhance.Color(image)
            color = random.choice(ratio)
            image = enh_col.enhance(color)
        if random.choice([0, 1]):
            enh_con = ImageEnhance.Contrast(image)
            contrast = random.choice(ratio)
            image = enh_con.enhance(contrast)
        if random.choice([0, 1]):
            enh_sha = ImageEnhance.Sharpness(image)
            sharpness = random.choice(ratio)
            image = enh_sha.enhance(sharpness)
        if random.choice([0, 1]):
            image = image.filter(ImageFilter.BLUR)

        image = np.asarray(image)
        return {'img': image, 'annot': annots, 'scale': scales}


class Normalizer(object):

    def __init__(self):
        self.mean = np.array([[[0.485, 0.456, 0.406]]])
        self.std = np.array([[[0.229, 0.224, 0.225]]])

    def __call__(self, sample):

        image, annots, scales = sample['img'], sample['annot'], sample['scale']

        image = (image.astype(np.float32)-self.mean)/self.std
        
        sample = {'img': torch.from_numpy(image), 'annot': torch.from_numpy(annots), 'scale': scales}
        return sample

class UnNormalizer(object):
    def __init__(self, mean=None, std=None):
        if mean == None:
            self.mean = [0.485, 0.456, 0.406]
        else:
            self.mean = mean
        if std == None:
            self.std = [0.229, 0.224, 0.225]
        else:
            self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


class AspectRatioBasedSampler(Sampler):

    def __init__(self, data_source, batch_size, drop_last):
        self.data_source = data_source
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.groups = self.group_images()

    def __iter__(self):
        random.shuffle(self.groups)
        for group in self.groups:
            yield group

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size

    def group_images(self):
        # determine the order of the images
        order = list(range(len(self.data_source)))
        order.sort(key=lambda x: self.data_source.image_aspect_ratio(x))

        # divide into groups, one group = one batch
        return [[order[x % len(order)] for x in range(i, i + self.batch_size)] for i in range(0, len(order), self.batch_size)]


