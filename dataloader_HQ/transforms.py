import cv2
import albumentations as alb
from albumentations.pytorch.transforms import ToTensorV2

from PIL import Image
from torchvision.transforms import transforms

def create_data_transforms_alb(args, split='train'):
    if split == 'train':
        return alb.Compose([
            alb.Resize(args.image_size, args.image_size),
            alb.HorizontalFlip(p=0.5),
            alb.VerticalFlip(p=0.5),
            alb.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
            ToTensorV2(),
        ])
    elif split == 'eval':
        return alb.Compose([
            alb.Resize(args.image_size, args.image_size),
            alb.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
            ToTensorV2(),
        ])
    elif split == 'test':
        return alb.Compose([
            alb.Resize(args.image_size, args.image_size),
            alb.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
            ToTensorV2(),
        ])

create_data_transforms = create_data_transforms_alb
