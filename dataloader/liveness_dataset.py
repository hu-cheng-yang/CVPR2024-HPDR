import os
import random
import cv2
import json
import numpy as np
import lmdb
from functools import partial
import albumentations as alb
from albumentations.pytorch.transforms import ToTensorV2
import torch
from torch.utils.data import Dataset, DataLoader
from dataloader.type import getType

class LivenessDataset(Dataset):
    def __init__(self, setting = "CASIA", split='train', category=None, transform=None, img_mode='rgb_hsv', depth_size=32):

        self.setting = setting
        self.split = split
        self.category = category
        self.transform1 = transform[0]
        self.transform2 = alb.Compose([
                           alb.Resize(256, 256),
                           alb.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
                           ToTensorV2(),
                           ])
        self.img_mode = img_mode
        self.depth_size = depth_size
        self.margin = 0.7
        root = "ROOT_PATH"
        LMDB_root = root + "/Dataset"
        self.env = lmdb.open(LMDB_root, readonly=True, max_readers=512)
        self.data = self.env.begin(write=False)
        self.use_LMDB = True
        self.setting = setting

        datasetname = None
        if setting == "CASIA":
            datasetname = "CASIA_database"
        elif setting == "idiap":
            datasetname = "replayattack"
        elif setting == "OULU":
            datasetname = "Oulu_NPU"
        elif setting == "MSU":
            datasetname = "MSU-MFSD"
        else:
            raise Exception("Dataset name is not right!")
        train_pos_list_path = root + "/Dataset/{}/lists/train_real_5points.list".format(datasetname)
        train_neg_list_path = root + "/Dataset/{}/lists/train_fake_5points.list".format(datasetname)
        test_pos_list_path = root + "/Dataset/{}/lists/test_real_5points.list".format(datasetname)
        test_neg_list_path = root + "/Dataset/{}/lists/test_fake_5points.list".format(datasetname)
        test_list_path = root + "/Dataset/{}/lists/test_5points.list".format(datasetname)
        if self.split == 'train' and self.category == 'pos':
            self.items = open(train_pos_list_path).read().splitlines() + open(test_pos_list_path).read().splitlines()
        elif self.split == 'train' and self.category == 'neg':
            self.items = open(train_neg_list_path).read().splitlines() + open(test_neg_list_path).read().splitlines()
        elif self.split == 'test' and self.category == 'pos':
            self.items = open(test_pos_list_path).read().splitlines()
        elif self.split == 'test' and self.category == 'neg':
            self.items = open(test_neg_list_path).read().splitlines()
        elif self.split == 'test' and self.category == None:
            self.items = open(test_list_path).read().splitlines()
        else:
            self.items = []

        self._display_infos()

    def _display_infos(self):
        print(f'=> Dataset {self.setting} loaded')
        print(f'=> Split {self.split}')
        print(f'=> category {self.category}')
        print(f'=> Total number of items: {len(self.items)}')
        print(f'=> Image mode: {self.img_mode}')
        print(f'===========================================')

    def _add_face_margin(self, x, y, w, h, margin=2.0):
        x_marign = int(w * margin / 2)
        y_marign = int(h * margin / 2)

        x1 = x - x_marign
        x2 = x + w + x_marign
        y1 = y - y_marign
        y2 = y + h + y_marign

        return x1, x2, y1, y2

    def _get_item_index(self,index=0):
        item = self.items[index]
        res = item.split(' ')
        img_path = res[0]
        label = int(res[1])

        if self.use_LMDB:
            img_bin = self.data.get(img_path.encode())
            try:
                img_buf = np.frombuffer(img_bin, dtype=np.uint8)
                img = cv2.imdecode(img_buf, cv2.IMREAD_COLOR)
            except:
                print('load img_buf error')
                print(img_path)
                img_path, label, img, res = self._get_item_index(index+1)
        else:
            img = cv2.imread(img_path)

        return img_path, label, img, res

    def _reset_lmdb(self):
        self.env.close()
        self.env.close()
        self.env = lmdb.open(self.LMDB_root, readonly=True, max_readers=1024)
        self.data = self.env.begin(write=False)
        print(self.data.id())

    def __getitem__(self, index):
        item = self.items[index]
        res = item.split(' ')
        img_path = res[0]
        label = int(res[1])
        img_bin = self.data.get(img_path.encode())
        try:
            img_buf = np.frombuffer(img_bin, dtype=np.uint8)
            img = cv2.imdecode(img_buf, cv2.IMREAD_COLOR)
        except:
            raise Exception("dataloader Error!")


        x_list = [int(float(res[6])), int(float(res[8])), int(float(res[10])), int(float(res[12])), int(float(res[14]))]
        y_list = [int(float(res[7])), int(float(res[9])), int(float(res[11])), int(float(res[13])), int(float(res[15]))]

        x, y = min(x_list), min(y_list)
        w, h = max(x_list) - x, max(y_list) - y

        side = w if w > h else h

        x1, x2, y1, y2 = self._add_face_margin(x, y, side, side, margin=self.margin)
        max_h, max_w = img.shape[:2]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(max_w, x2)
        y2 = min(max_h, y2)
        if x1>=x2 or y1>=y2:
            return self.__getitem__(0)
        img = img[y1:y2, x1:x2]


        if self.img_mode == 'rgb':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif self.img_mode == 'hsv':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        elif self.img_mode == 'ycrcb':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        elif self.img_mode == 'rgb_hsv':
            img_ori = img
            img = cv2.cvtColor(img_ori, cv2.COLOR_BGR2RGB)
            img_hsv = cv2.cvtColor(img_ori, cv2.COLOR_BGR2HSV)

        if self.split=='test':
            label = 0 if label == 0 else 1
        if self.transform1 is not None and self.transform2 is not None:
            img_q = self.transform1(image=img)['image']
            img_k = self.transform2(image=img)['image']
        if self.img_mode == 'rgb_hsv':
            print(self.img_mode)
            transHSV = alb.Compose([
                           alb.Resize(256, 256),
                           alb.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                           ToTensorV2(),
                           ])
            img_hsv = transHSV(image=img_hsv)['image']
            img_q = torch.cat([img_q, img_hsv], 0)
            img_k = torch.cat([img_k, img_hsv], 0)

        if label == 1:
            domain = getType(self.setting, img_path)
        else:
            domain = 0

        if self.split == "test":
           return img_q, label, img_path[:-9], domain

        size = 32

        if label == 0:
            binary_map = torch.ones((1,size,size)).float()
        else:
            binary_map = torch.zeros((1,size,size)).float()
        return img_q, img_k, binary_map, label, domain, img_path


    def __len__(self):
        return len(self.items)

def set_seed(SEED):
    if SEED:
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
