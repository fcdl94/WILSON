from torch import nn
import torch
from torch.nn import functional as F
import numpy as np
from torch.nn import Parameter


class IncrementalClassifier(nn.Module):
    def __init__(self, classes, norm_feat=False, channels=256):
        super().__init__()
        self.channels = channels
        self.classes = classes
        self.tot_classes = 0
        for lcl in classes:
            self.tot_classes += lcl
        self.cls = nn.ModuleList(
            [nn.Conv2d(channels, c, 1) for c in classes])
        self.norm_feat = norm_feat

    def forward(self, x):
        if self.norm_feat:
            x = F.normalize(x, p=2, dim=1)
        out = []
        for mod in self.cls:
            out.append(mod(x))
        return torch.cat(out, dim=1)

    def imprint_weights_step(self, features, step):
        self.cls[step].weight.data = features.view_as(self.cls[step].weight.data)

    def imprint_weights_class(self, features, cl, alpha=1):
        step = 0
        while cl >= self.classes[step]:
            cl -= self.classes[step]
            step += 1
        if step == len(self.classes) - 1:  # last step! alpha = 1
            alpha = 0
        self.cls[step].weight.data[cl] = alpha * self.cls[step].weight.data[cl] + \
                                         (1 - alpha) * features.view_as(self.cls[step].weight.data[cl])
        self.cls[step].bias.data[cl] = 0.


class CosineClassifier(nn.Module):
    def __init__(self, classes, channels=256):
        super().__init__()
        self.channels = channels
        self.cls = nn.ModuleList(
            [nn.Conv2d(channels, c, 1, bias=False) for c in classes])
        self.scaler = 10.
        self.classes = classes
        self.tot_classes = 0
        for lcl in classes:
            self.tot_classes += lcl

    def forward(self, x):
        x = F.normalize(x, p=2, dim=1)
        out = []
        for i, mod in enumerate(self.cls):
            out.append(self.scaler * F.conv2d(x, F.normalize(mod.weight, dim=1, p=2)))
        return torch.cat(out, dim=1)

    def imprint_weights_step(self, features, step):
        self.cls[step].weight.data = features.view_as(self.cls[step].weight.data)

    def imprint_weights_class(self, features, cl, alpha=1):
        step = 0
        while cl >= self.classes[step]:
            cl -= self.classes[step]
            step += 1
        if step == len(self.classes) - 1:  # last step! alpha = 1
            alpha = 0
        self.cls[step].weight.data[cl] = alpha * self.cls[step].weight.data[cl] + \
                                         (1 - alpha) * features.view_as(self.cls[step].weight.data[cl])