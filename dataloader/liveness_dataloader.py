try:
    from .liveness_dataset import LivenessDataset
    from .transforms import create_data_transforms
except Exception:
    from dataloader.liveness_dataset import LivenessDataset
    from dataloader.transforms import create_data_transforms

import torch.utils.data as data

def create_dataloader(name, getreal, batch_size, args, spilt='train', in_channels=3):

    split = spilt
    setting = name
    if getreal:
       category = "pos"
    else:
       category = "neg"

    if getreal is None:
       category = None

    if in_channels == 3:
       img_mode='rgb'
    elif in_channels == 6:
       img_mode='rgb_hsv'
    else:
       raise Exception("img channels should be 3 or 6!") 

    transform1 = create_data_transforms(args, split)
    transform2 = create_data_transforms(args, split)

    dataset = LivenessDataset(setting=setting, split=split, transform=[transform1, transform2], category=category, img_mode=img_mode)

    sampler = None

    shuffle = True if sampler is None and split == 'train' else False
    batch_size = batch_size

    dataloader = data.DataLoader(dataset,
                                 batch_size=batch_size,
                                 shuffle=True,
                                 sampler=sampler,
                                 num_workers=3,
                                 drop_last=shuffle,
                                 pin_memory=False)
    return dataloader


def get_dataset_loader(name, getreal, batch_size, args, in_channels=3):
    return create_dataloader(name, getreal, batch_size, args, spilt='train', in_channels=3)

def get_dataset_loader_both(name, getreal, batch_size, args, in_channels=3):
    return create_dataloader(name, getreal, batch_size, args, spilt='both', in_channels=3)

def get_tgt_dataset_loader(name, getreal, batch_size, args, in_channels=3):
    return create_dataloader(name, getreal, batch_size, args, spilt='test', in_channels=3) 
