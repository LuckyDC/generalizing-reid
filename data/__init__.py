import os

import torch
import torch.distributed as dist
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler
from torch.utils.data import RandomSampler

from data.dataset import CrossDataset
from data.dataset import ImageFolder
from data.dataset import ImageListFile
from data.sampler import CrossDatasetDistributedSampler
from data.sampler import CrossDatasetRandomSampler
from data.sampler import RandomIdentitySampler


def collate_fn(batch):  # img, label, cam_id, img_path, img_id
    samples = list(zip(*batch))

    data = [torch.stack(x, 0) for i, x in enumerate(samples) if i != 3]
    data.insert(3, samples[3])
    return data


def get_train_loader(root, batch_size, image_size, random_flip=False, random_crop=False, random_erase=False,
                     color_jitter=False, padding=0, num_workers=4):
    # data pre-processing
    t = [T.Resize(image_size)]

    if random_flip:
        t.append(T.RandomHorizontalFlip())

    if color_jitter:
        t.append(T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0))

    if random_crop:
        t.extend([T.Pad(padding, fill=127), T.RandomCrop(image_size)])

    t.extend([T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    if random_erase:
        t.append(T.RandomErasing(scale=(0.02, 0.25)))

    transform = T.Compose(t)

    # dataset
    train_dataset = ImageFolder(root, transform=transform, recursive=True, label_organize=True)

    if dist.is_initialized():
        rand_sampler = DistributedSampler(train_dataset)
    else:
        rand_sampler = RandomSampler(train_dataset)

    # loader
    train_loader = DataLoader(train_dataset, batch_size, drop_last=True, pin_memory=True,
                              collate_fn=collate_fn, num_workers=num_workers, sampler=rand_sampler)

    return train_loader


def get_cross_domain_train_loader(source_root, target_root, batch_size, image_size, random_flip=False,
                                  random_crop=False, random_erase=False, color_jitter=False, padding=0, num_workers=4):
    if isinstance(random_crop, bool):
        random_crop = (random_crop, random_crop)
    if isinstance(random_flip, bool):
        random_flip = (random_flip, random_flip)
    if isinstance(random_erase, bool):
        random_erase = (random_erase, random_erase)
    if isinstance(color_jitter, bool):
        color_jitter = (color_jitter, color_jitter)

    # data pre-processing
    source_transform = [T.Resize(image_size)]
    target_transform = [T.Resize(image_size)]

    if random_flip[0]:
        source_transform.append(T.RandomHorizontalFlip())
    if random_flip[1]:
        target_transform.append(T.RandomHorizontalFlip())

    if color_jitter[0]:
        source_transform.append(T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0))
    if color_jitter[1]:
        target_transform.append(T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0))

    if random_crop[0]:
        source_transform.extend([T.Pad(padding, fill=127), T.RandomCrop(image_size)])
    if random_crop[1]:
        target_transform.extend([T.Pad(padding, fill=127), T.RandomCrop(image_size)])

    source_transform.extend([T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    target_transform.extend([T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    if random_erase[0]:
        source_transform.append(T.RandomErasing(scale=(0.02, 0.25)))
    if random_erase[1]:
        target_transform.append(T.RandomErasing(scale=(0.02, 0.25)))

    source_transform = T.Compose(source_transform)
    target_transform = T.Compose(target_transform)

    # dataset
    source_dataset = ImageFolder(source_root, transform=source_transform, recursive=True, label_organize=True)
    target_dataset = ImageFolder(target_root, transform=target_transform, recursive=True, label_organize=True)

    concat_dataset = CrossDataset(source_dataset, target_dataset)

    # sampler
    if dist.is_initialized():
        cross_sampler = CrossDatasetDistributedSampler(source_dataset, target_dataset, batch_size)
    else:
        cross_sampler = CrossDatasetRandomSampler(source_dataset, target_dataset, batch_size)

    # data loader
    train_loader = DataLoader(concat_dataset, batch_size, sampler=cross_sampler, drop_last=True, pin_memory=True,
                              collate_fn=collate_fn, num_workers=num_workers)

    return train_loader


def get_test_loader(root, batch_size, image_size, num_workers=4):
    # transform
    transform = T.Compose([
        T.Resize(image_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # dataset
    if root.lower().find("msmt") != -1:
        prefix = os.path.join(os.path.dirname(root), "test")
        test_dataset = ImageListFile(root, prefix=prefix, transform=transform)
    else:
        test_dataset = ImageFolder(root, transform=transform)

    # dataloader
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             pin_memory=True,
                             drop_last=False,
                             collate_fn=collate_fn,
                             num_workers=num_workers)

    return test_loader
