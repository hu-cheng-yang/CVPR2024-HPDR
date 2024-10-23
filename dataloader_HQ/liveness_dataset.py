import os
import random
import cv2
import json
import numpy as np
import lmdb
from functools import partial
from skimage.feature import local_binary_pattern
from torchvision.transforms import transforms
import albumentations as alb
from albumentations.pytorch.transforms import ToTensorV2
import torch
from torch.utils.data import Dataset, DataLoader
from dataloader.type import getType
from tqdm import tqdm

class LivenessDataset(Dataset):
    def __init__(self, setting = "makeup", split='train', category=None, transform=None, img_mode='rgb_hsv', depth_size=32):

        self.setting = setting
        self.split = split
        self.category = category
        self.transform1 = transform[0]
        self.transform2 = alb.Compose([
                           alb.Resize(256, 256),
                           ToTensorV2(),
                           ])
        self.img_mode = img_mode
        self.depth_size = depth_size
        self.margin = 0.7

        root = "PATH/HQ_WMCA"
        LMDB_root = root
        self.env = lmdb.open(LMDB_root, readonly=True, max_readers=10000)
        self.data = self.env.begin(write=False)

        img_files = open("HQ_list.txt").read().splitlines()
        
        self.use_LMDB = True
        self.setting = setting

        datasetname = setting
        list_path = f"{root}/PROTOCOL-LOO_{datasetname}.csv"
        self.items = open(list_path).read().splitlines()

        ref_label = 0 if category == "pos" else 1
        split = "eval" if split == "test" else split

        item_ = []
        videos = []
        _item_ = []
        for i in self.items:
            s = i.split(",")
            label = int(s[1])
            split_our = s[3]
            if (ref_label == label or category == None) and split == split_our:
                item_.append(i)

        for i in item_:
            s = i.split(",")
            video = s[0]
            videos.append(video.split("/")[1])

        for v in img_files:
            if v.split("/")[1] in videos:
                index = videos.index(v.split("/")[1])
                s = item_[index].split(",")
                _item_.append(f"{v},{s[1]},{s[2]},{s[3]}")

        self.items = _item_
            
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
        res = item.split(',')
        img_path = res[0]
        label = int(res[1])
        if self.use_LMDB:
            img_bin = self.data.get(img_path.encode())
            try:
                img_buf = np.frombuffer(img_bin, dtype=np.uint8)
                img = cv2.imdecode(img_buf, cv2.IMREAD_COLOR)
            except:
                raise Exception("dataloader Error!")
        else:
            img = cv2.imread(img_path)

        #try:
        if True:
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

        domain = res[2]
        img_path = img_path.split("/")[1]

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
