import os
from glob import glob

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

'''
    Specific dataset classes for person re-identification dataset. 
'''


class ImageFolder(Dataset):
    def __init__(self, root, transform=None, recursive=False, label_organize=False):
        if recursive:
            image_list = glob(os.path.join(root, "**", "*.jpg"), recursive=recursive) + \
                         glob(os.path.join(root, "**", "*.png"), recursive=recursive)
        else:
            image_list = glob(os.path.join(root, "*.jpg")) + glob(os.path.join(root, "*.png"))

        self.image_list = list(filter(lambda x: int(os.path.basename(x).split("_")[0]) != -1, image_list))
        self.image_list.sort()

        ids = []
        cam_ids = []
        for img_path in self.image_list:
            splits = os.path.basename(img_path).split("_")
            ids.append(int(splits[0]))

            if root.lower().find("msmt") != -1:
                cam_id = int(splits[2])
            else:
                cam_id = int(splits[1][1])

            cam_ids.append(cam_id - 1)

        if label_organize:
            # organize identity label
            unique_ids = set(ids)
            label_map = dict(zip(unique_ids, range(len(unique_ids))))

            ids = map(lambda x: label_map[x], ids)
            ids = list(ids)

        self.ids = ids
        self.cam_ids = cam_ids
        self.num_id = len(set(ids))

        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, item):
        img_path = self.image_list[item]

        img = Image.open(img_path)

        if self.transform is not None:
            img = self.transform(img)

        label = torch.tensor(self.ids[item], dtype=torch.long)
        cam = torch.tensor(self.cam_ids[item], dtype=torch.long)
        item = torch.tensor(item, dtype=torch.long)

        return img, label, cam, img_path, item


class ImageListFile(Dataset):
    def __init__(self, path, prefix=None, transform=None, label_organize=False):
        if not os.path.isfile(path):
            raise ValueError("The file %s does not exist." % path)

        image_list = list(np.loadtxt(path, delimiter=" ", dtype=np.str)[:, 0])

        if prefix is not None:
            image_list = map(lambda x: os.path.join(prefix, x), image_list)

        self.image_list = list(filter(lambda x: int(os.path.basename(x).split("_")[0]) != -1, image_list))
        self.image_list.sort()

        ids = []
        cam_ids = []
        for img_path in self.image_list:
            splits = os.path.basename(img_path).split("_")
            ids.append(int(splits[0]))

            if path.lower().find("msmt") != -1:
                cam_id = int(splits[2])
            else:
                cam_id = int(splits[1][1])

            cam_ids.append(cam_id - 1)

        if label_organize:
            # organize identity label
            unique_ids = set(ids)
            label_map = dict(zip(unique_ids, range(len(unique_ids))))

            ids = map(lambda x: label_map[x], ids)
            ids = list(ids)

        self.cam_ids = cam_ids
        self.ids = ids
        self.num_id = len(set(ids))

        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, item):
        img_path = self.image_list[item]
        img = Image.open(img_path)

        if self.transform is not None:
            img = self.transform(img)

        label = torch.tensor(self.ids[item], dtype=torch.long)
        cam = torch.tensor(self.cam_ids[item], dtype=torch.long)
        item = torch.tensor(item, dtype=torch.long)

        return img, label, cam, img_path, item


class CrossDataset(Dataset):
    def __init__(self, source_dataset, target_dataset):
        super(CrossDataset, self).__init__()

        self.source_dataset = source_dataset
        self.target_dataset = target_dataset

        self.source_size = len(self.source_dataset)
        self.target_size = len(target_dataset)

        self.num_source_cams = len(set(source_dataset.cam_ids))
        self.num_target_cams = len(set(target_dataset.cam_ids))

    def __len__(self):
        return self.source_size + self.target_size

    def __getitem__(self, idx):
        # from source dataset
        if idx < self.source_size:
            sample = self.source_dataset[idx]
            sample[2].add_(self.num_target_cams)

            return sample
        # from target dataset
        else:
            idx = idx - self.source_size
            sample = list(self.target_dataset[idx])
            sample[1].fill_(-1)  # set target label to -1

            return sample
