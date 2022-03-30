import torch
import numpy as np
import bisect


def image_labels(dataset):
    images = []
    for i in range(len(dataset)):
        cls = np.unique(np.array(dataset[i][1]))
        images.append(cls)
    return images


def group_images(dataset, labels):
    # Group images based on the label in LABELS (using labels not reordered)
    idxs = {lab: [] for lab in labels}

    labels_cum = labels + [0, 255]
    for i in range(len(dataset)):
        cls = np.unique(np.array(dataset[i][1]))
        if all(x in labels_cum for x in cls):
            for x in cls:
                if x in labels:
                    idxs[x].append(i)
    return idxs


def group_images_bkg(dataset, labels):
    # Group images based on the label in LABELS (using labels not reordered)
    idxs = {lab: [] for lab in labels}

    labels_cum = labels + [0, 255]
    for i in range(len(dataset)):
        cls, count_classes = np.unique(np.array(dataset[i][1]), return_counts=True)
        count = 0
        if all(x in labels_cum for x in cls):
            for j, cl in enumerate(cls):
                if cl == 0 or cl == 255:
                    count += count_classes[j]
            for x in cls:
                if x in labels:
                    idxs[x].append((i, count))
    return idxs


class Subset(torch.utils.data.Dataset):
    """
    Subset of a dataset at specified indices.
    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
        transform (callable): way to transform the images and the targets
        target_transform(callable): way to transform the target labels
    """

    def __init__(self, dataset, indices, transform=None, target_transform=None):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, idx):
        sample, target = self.dataset[self.indices[idx]]

        if self.transform is not None:
            sample, target = self.transform(sample, target)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.indices)


class ConcatDataset(torch.utils.data.Dataset):
    r"""Dataset as a concatenation of multiple datasets.

    This class is useful to assemble different existing datasets.

    Arguments:
        datasets (sequence): List of datasets to be concatenated
    """

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def __init__(self, datasets):
        super(ConcatDataset, self).__init__()
        assert len(datasets) > 0, 'datasets should not be an empty iterable'
        self.datasets = list(datasets)
        self.cumulative_sizes = self.cumsum(self.datasets)

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]


class MaskLabels:
    """
    Use this class to mask labels that you don't want in your dataset.
    Arguments:
    labels_to_keep (list): The list of labels to keep in the target images
    mask_value (int): The value to replace ignored values (def: 0)
    """
    def __init__(self, labels_to_keep, mask_value=0):
        self.labels = labels_to_keep
        self.value = torch.tensor(mask_value, dtype=torch.uint8)

    def __call__(self, sample):
        # sample must be a tensor
        assert isinstance(sample, torch.Tensor), "Sample must be a tensor"

        sample.apply_(lambda t: t.apply_(lambda x: x if x in self.labels else self.value))

        return sample