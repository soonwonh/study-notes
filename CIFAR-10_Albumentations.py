from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2, ToTensor

train_transform = A.Compose([

    A.RandomResizedCrop(always_apply=False, p=1, height=32, width=32, scale=(0.08,1.0),ratio=(0.75,1.33333333), interpolation=0),
    A.ColorJitter(0.4,0.4,0.4,0.1,False,0.8),
    A.HorizontalFlip(p=0.5),
    A.ToGray(p=0.2),
    A.Normalize(mean=(0.4914, 0.4822, 0.4465),std=(0.2023, 0.1994, 0.2010),max_pixel_value=255.0,always_apply=True,p=1.0),
    ToTensorV2()
])


class CIFAR10Pair(CIFAR10):

    def __getitem__(self, index):

        img = self.data[index]

        if self.transform is not None:
            im_1 = self.transform(image=img)["image"]
            im_2 = self.transform(image=img)["image"]

        return im_1, im_2

train_data = CIFAR10Pair(root='data', train=True, transform=train_transform, download=True)
train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=16, pin_memory=True, drop_last=True)


