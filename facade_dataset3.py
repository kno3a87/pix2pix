# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 16:21:52 2019
参考
https://spjai.com/pix2pix-image-generation/
@author: KNO3
"""
import os

import numpy
from PIL import Image
import six

import numpy as np

from io import BytesIO
import os
import pickle
import json
import numpy as np

import skimage.io as io

from chainer.dataset import dataset_mixin

# download `BASE` dataset from http://cmp.felk.cvut.cz/~tylecr1/facade/
class FacadeDataset(dataset_mixin.DatasetMixin):
    def __init__(self, kyoushiTpPfDir='./Tpin/',kyoushiPfDir='./Tpout/'):
        self.dataset = []
        #labelが入力画像、baseが教師出力画像
        files = os.listdir(kyoushiTpPfDir)
        for file in files:
            img = Image.open(kyoushiTpPfDir+file)
            label = Image.open(kyoushiPfDir+file)
            label = label.convert(mode="RGB")
            # Chainerの形に合わせる
            # https://qiita.com/ysasaki6023/items/fa2fe9c2336677821583
            # 128なのはNumpy
            # https://qiita.com/DogFortune/items/b6e71ba8aa5b358f01af
            img = np.asarray(img).astype("f").transpose(2,0,1)/128.0-1.0
            label = np.asarray(label).astype("f").transpose(2,0,1)/128.0-1.0

            self.dataset.append((img, label))

        print("load dataset done")
        
    def __len__(self):
        return len(self.dataset)

    def get_example(self, i, crop_width=320):
        return self.dataset[i][1], self.dataset[i][0]
    
    
class FacadeTestDataset(dataset_mixin.DatasetMixin):
    def __init__(self, testTpPfDir='./TpTestin/',testPfDir='./TpTestout/'):
        self.testdataset = []
        files = os.listdir(testTpPfDir)
        for file in files:
            img = Image.open(testTpPfDir+file)
            label = Image.open(testPfDir+file)
            label = label.convert(mode="RGB")
            img = np.asarray(img).astype("f").transpose(2,0,1)/128.0-1.0
            label = np.asarray(label).astype("f").transpose(2,0,1)/128.0-1.0

            self.testdataset.append((img, label))

        print("load testdataset done")
        
    def __len__(self):
        return len(self.testdataset)

    def get_example(self, i, crop_width=65):
        return self.testdataset[i][1], self.testdataset[i][0]
    
