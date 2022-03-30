import os.path as osp
import torch.utils.data as data
import numpy as np
from .dataset import IncrementalSegmentationDataset
from PIL import Image

ignore_labels = [12, 26, 29, 30, 45, 66, 68, 69, 71, 83, 91]  # starting from 1=person


class COCO(data.Dataset):

    def __init__(self,
                 root,
                 train=True,
                 transform=None,
                 indices=None):

        root = osp.expanduser(root)
        base_dir = "coco"
        ds_root = osp.join(root, base_dir)
        splits_dir = osp.join(ds_root, 'split')

        if train:
            self.image_set = "train"
            split_f = osp.join(splits_dir, 'train.txt')
            folder = 'train2017'
        else:
            self.image_set = "val"
            split_f = osp.join(splits_dir, 'val.txt')
            folder = 'val2017'

        ann_folder = "annotations"

        with open(osp.join(split_f), "r") as f:
            files = f.readlines()

        self.images = [(osp.join(ds_root, "images", folder, x[:-1] + ".jpg"),
                        osp.join(ds_root, ann_folder, folder, x[:-1] + ".png")) for x in files]

        self.img_lvl_labels = np.load(osp.join(ds_root, f"1h_labels_{self.image_set}.npy"))

        self.transform = transform
        self.indices = indices if indices is not None else np.arange(len(self.images))
        # self.img_lvl_only = False

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        # if self.img_lvl_only:  # 4 performance reason
        #     img = Image.open(self.images[self.indices[index]][0]).convert('RGB')
        #     img_lvl_lbls = self.img_lvl_labels[self.indices[index]]
        #
        #     if self.transform is not None:
        #         img = self.transform(img)
        #
        #     return img, None, img_lvl_lbls
        # else:
        img = Image.open(self.images[self.indices[index]][0]).convert('RGB')
        target = Image.open(self.images[self.indices[index]][1])
        img_lvl_lbls = self.img_lvl_labels[self.indices[index]]

        if self.transform is not None:
            img, target = self.transform(img, target)

        return img, target, img_lvl_lbls

    def __len__(self):
        return len(self.indices)


class COCOIncremental(IncrementalSegmentationDataset):
    def make_dataset(self, root, train, indices, saliency=False, pseudo=None):
        full_voc = COCO(root, train, transform=None, indices=indices)
        return full_voc

