from __future__ import division

import os
import random
from PIL import Image
import torch
import numpy as np
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.nn import init
import scipy.io as scio
import math
from torch.nn import DataParallel
from torch import nn
from copy import deepcopy

from pdb import set_trace as st
import torch.nn.functional as F


def make_variable(tensor, volatile=False):
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return tensor


def tensor2im(image_tensor, imtype=np.uint8):
    image_numpy = image_tensor.cpu().detach().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0
    return image_numpy.astype(imtype)


def denormalize(x, std, mean):
    """Invert normalization, and then convert array into image."""
    out = x * std + mean
    return out.clamp(0, 1)

def weights_init_const(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.constant_(m.weight.data, 1)
        if m.bias is not None:
            init.constant_(m.bias.data, 0.0)

def weights_init_xavier(m):
    classname = m.__class__.__name__
    print('classname', classname)
    if classname.find('Conv2d') != -1:
        init.xavier_normal_(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.xavier_normal_(m.weight.data, gain=1)
        if m.bias is not None:
            init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm2d') != -1:
        init.uniform(m.weight.data, 0.02, 1.0)
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchInstanceNorm2d') != -1:
        init.uniform(m.weight.data, 0.02, 1.0)
        init.constant_(m.bias.data, 0.0)
    elif classname.find('InstanceNorm2d') != -1:
        init.uniform(m.weight.data, 0.02, 1.0)
        init.constant_(m.bias.data, 0.0)
    else:
        pass

def weights_init_normal(m):
    classname = m.__class__.__name__
    print(classname)
    if classname.find('Conv') != -1:
        init.uniform(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.uniform(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm2d') != -1:
        init.uniform(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
        if m.bias is not None:
            init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm2d') != -1:
        init.uniform(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    print(classname)
    if classname.find('Conv') != -1:
        init.orthogonal_(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.orthogonal_(m.weight.data, gain=1)
        if m.bias is not None:
            init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm2d') != -1:
        init.uniform(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

def weights_init_orthogonal_rnn(m):
    classname = m.__class__.__name__
    if classname.find('LSTM') != -1:
        init.orthogonal_(m.all_weights[0][0], gain=1)
        init.orthogonal_(m.all_weights[0][1], gain=1)
        init.constant_(m.all_weights[0][2], 1)
        init.constant_(m.all_weights[0][3], 1)
    elif classname.find('Linear') != -1:
        init.xavier_normal_(m.weight.data, gain=1)
        init.constant_(m.bias.data, 0.0)


def init_weights(net, init_type='normal'):
    print('initialization method [%s]' % init_type)
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'xavier':
        net.apply(weights_init_xavier)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    elif init_type == 'orthogonal_rnn':
        net.apply(weights_init_orthogonal_rnn)
    elif init_type == 'const':
        net.apply(weights_init_const)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)                


def init_random_seed(manual_seed):
    """Init random seed."""
    seed = None
    if manual_seed is None:
        seed = random.randint(1, 10000)
    else:
        seed = manual_seed
    print("use random seed: {}".format(seed))
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    return seed


def init_model(net, restore, init_type, init=True):
    """Init models with cuda and weights."""
    # init weights of model
    if init:
        init_weights(net, init_type)
    
    # restore model weights
    if restore is not None:
        if os.path.exists(restore):

            # original saved file with DataParallel
            state_dict = torch.load(restore)
            # create new OrderedDict that does not contain `module.`
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if 'module' in k:
                    name = k[7:] # remove `module.`
                else:
                    name = k
                new_state_dict[name] = v
            # load params
            net.load_state_dict(new_state_dict)
            
            net.restored = True
            print("*************Restore model from: {}".format(os.path.abspath(restore)))
        else:
            # raise ValueError('the path ' + restore +' does not exist')
            print('the path ' + restore +' does not exist')
    print('init model')

    if torch.cuda.is_available():
        cudnn.benchmark = True
        net.cuda()

    return net


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def get_sampled_data_loader(dataset, candidates_num, shuffle=True):
    """Get data loader for sampled dataset."""
    # get indices
    indices = torch.arange(0, len(dataset))
    if shuffle:
        indices = torch.randperm(len(dataset))
    # slice indices
    candidates_num = min(len(dataset), candidates_num)
    excerpt = indices.narrow(0, 0, candidates_num).long()
    sampler = torch.utils.data.sampler.SubsetRandomSampler(excerpt)
    return make_data_loader(dataset, sampler=sampler, shuffle=False)

 
    

def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil = image_pil.resize((128, 128), resample=Image.BICUBIC)
    image_pil.save(image_path)
    


def get_inf_iterator(data_loader, depth=True, return_path=False):
    """Inf data iterator."""
    while True:
        if depth:
            for catimages, k, depthimages, labels, d, index in data_loader:
                yield (catimages, k, depthimages, labels, d, index)
        else:
            if return_path:
                for catimages, k, depthimages, labels, img_paths in data_loader:
                    yield (catimages, k, labels, img_paths)
            else:
                for catimages, k, depthimages, labels, img_paths in data_loader:
                    yield (catimages, k, labels)

def get_inf_iterator_CM(data_loader, depth=True, return_path=False):
    """Inf data iterator."""
    while True:
        if depth:
            for catimages, labels in data_loader:
                yield (catimages, labels)
        else:
            if return_path:
                for catimages, labels, img_paths in data_loader:
                    yield (catimages, labels, img_paths)
            else:
                for catimages, labels, img_paths in data_loader:
                    yield (catimages, labels)

def get_inf_iterator_tst(data_loader):
    """Inf data iterator."""
    while True:
        for catimages, labels in data_loader:
            yield (catimages, labels)
   


def copy_weights(from_net, to_net):
    ''' Set this module's weights to be the same as those of 'net' '''
    # TODO: breaks if nets are not identical
    # TODO: won't copy buffers, e.g. for batch norm
    for m_from, m_to in zip(from_net.modules(), to_net.modules()):
        if isinstance(m_to, nn.Linear) or isinstance(m_to, nn.Conv2d) or isinstance(m_to, nn.BatchNorm2d):
            m_to.weight.data = m_from.weight.data.clone()
            if m_to.bias is not None:
                m_to.bias.data = m_from.bias.data.clone()

class AverageMeter(object):
	"""Computes and stores the average and current value"""
	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val):
		self.val = val
		self.sum += val
		self.count += 1
		self.avg = self.sum / self.count





