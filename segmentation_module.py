import torch
import torch.nn as nn
import torch.nn.functional as functional

import inplace_abn
from inplace_abn import InPlaceABNSync, InPlaceABN, ABN

from functools import partial, reduce

import models
from modules import DeeplabV3, custom_bn
import torch.distributed as distributed


def get_norm(opts):
    if opts.norm_act == 'iabn_sync':
        norm = partial(InPlaceABNSync, activation="leaky_relu", activation_param=.01, group=distributed.group.WORLD)
    elif opts.norm_act == 'iabn':
        norm = partial(InPlaceABN, activation="leaky_relu", activation_param=.01)
    elif opts.norm_act == 'abr':
        norm = partial(custom_bn.ABR, activation="leaky_relu", activation_param=.01)
    elif opts.norm_act == 'iabr':
        norm = partial(custom_bn.InPlaceABR, activation="leaky_relu", activation_param=.01)
    else:  # std bn + leaky RELU -> NO INPLACE here
        norm = partial(ABN, activation="leaky_relu", activation_param=.01)

    return norm


def get_body(opts, norm):
    body = models.__dict__[f'net_{opts.backbone}'](norm_act=norm, output_stride=opts.output_stride)
    if not opts.no_pretrained:
        if opts.backbone == "wider_resnet38_a2":
            pretrained_path = f'pretrained/wide_resnet38_ipabn_lr_256.pth.tar'
        else:
            pretrained_path = f'pretrained/{opts.backbone}_iabn_sync.pth.tar'
        pre_dict = torch.load(pretrained_path, map_location='cpu')

        new_state = {}
        for k, v in pre_dict['state_dict'].items():
            if "module" in k:
                new_state[k[7:]] = v
            else:
                new_state[k] = v

        if 'classifier.fc.weight' in new_state:
            del new_state['classifier.fc.weight']
            del new_state['classifier.fc.bias']

        body.load_state_dict(new_state)
        del pre_dict  # free memory
        del new_state
    return body


def make_model(opts, classes=None):
    norm = get_norm(opts)
    body = get_body(opts, norm)

    head_channels = 256
    head = DeeplabV3(body.out_channels, head_channels, 256, norm_act=norm,
                     out_stride=opts.output_stride, pooling_size=opts.pooling)

    if classes is not None:
        model = IncrementalSegmentationModule(body, head, head_channels, classes=classes)
    else:
        model = SegmentationModule(body, head, head_channels, opts.num_classes)

    return model


def flip(x, dim):
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1,
                                dtype=torch.long, device=x.device)
    return x[tuple(indices)]


class IncrementalClassifier(nn.ModuleList):
    def forward(self, input):
        out = []
        for mod in self:
            out.append(mod(input))
        sem_logits = torch.cat(out, dim=1)
        return sem_logits


class IncrementalSegmentationModule(nn.Module):

    def __init__(self, body, head, head_channels, classes):
        super(IncrementalSegmentationModule, self).__init__()
        self.body = body
        self.head = head
        # classes must be a list where [n_class_task[i] for i in tasks]
        assert isinstance(classes, list), \
            "Classes must be a list where to every index correspond the num of classes for that task"
        self.cls = IncrementalClassifier(
            [nn.Conv2d(head_channels, c, 1) for c in classes]
        )
        self.classes = classes
        self.head_channels = head_channels
        self.tot_classes = reduce(lambda a, b: a + b, self.classes)

    def init_new_classifier(self, device):
        cls = self.cls[-1]
        imprinting_w = self.cls[0].weight[0]
        bkg_bias = self.cls[0].bias[0]

        bias_diff = torch.log(torch.FloatTensor([self.classes[-1] + 1])).to(device)

        new_bias = (bkg_bias - bias_diff)

        cls.weight.data.copy_(imprinting_w)
        cls.bias.data.copy_(new_bias)

        self.cls[0].bias[0].data.copy_(new_bias.squeeze(0))

    def forward(self, x, as_feature_extractor=False, interpolate=True, scales=None, do_flip=False):
        out_size = x.shape[-2:]

        x_b, x_b3 = self.body(x, ret_int=True)
        if not as_feature_extractor:
            x_pl = self.head(x_b)

            sem_logits = self.cls(x_pl)

            if interpolate:
                sem_logits = functional.interpolate(sem_logits, size=out_size, mode="bilinear", align_corners=False)

            return sem_logits, {"body": x_b, "pre_logits": x_pl, 'b3': x_b3}
        else:
            return {"body": x_b, 'b3': x_b3}

    def freeze(self):
        for par in self.parameters():
            par.requires_grad = False

    def fix_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, inplace_abn.ABN):
                m.eval()
                m.weight.requires_grad = False
                m.bias.requires_grad = False


class _MeanFusion:
    def __init__(self, x, classes):
        self.buffer = x.new_zeros(x.size(0), classes, x.size(2), x.size(3))
        self.counter = 0

    def update(self, sem_logits):
        # probs = F.softmax(sem_logits, dim=1)
        self.counter += 1
        self.buffer.add_((sem_logits - self.buffer) / self.counter)

    def output(self):
        _, cls = self.buffer.max(1)
        return self.buffer, cls


class _SumFusion:
    def __init__(self, x, classes):
        self.buffer = x.new_zeros(x.size(0), classes, x.size(2), x.size(3))
        self.counter = 0

    def update(self, sem_logits):
        self.counter += 1
        self.buffer.add_(sem_logits)

    def output(self):
        _, cls = self.buffer.max(1)
        return self.buffer, cls


class TestAugmentation:
    def __init__(self, classes, scales=None, do_flip=True, fusion='mean'):
        self.scales = scales if scales is not None else [1.]
        self.do_flip = do_flip
        self.fusion_cls = _MeanFusion if fusion == "mean" else _SumFusion
        self.classes = classes

    def __call__(self, func, x):

        fusion = self.fusion_cls(x, self.classes)
        out_size = x.shape[-2:]

        for scale in self.scales:
            # Main orientation
            if scale != 1:
                scaled_size = [round(s * scale) for s in x.shape[-2:]]
                x_up = nn.functional.interpolate(x, size=scaled_size, mode="bilinear", align_corners=False)
            else:
                x_up = x
            # Flipped orientation
            if self.do_flip:
                x_up = torch.cat((x_up, flip(x_up, -1)), dim=0)

            sem_logits = func(x_up)
            sem_logits = nn.functional.interpolate(sem_logits, size=out_size, mode="bilinear", align_corners=False)

            if self.do_flip:
                fusion.update(flip(sem_logits[1].unsqueeze(0), -1))
                sem_logits = sem_logits[0].unsqueeze(0)

            fusion.update(sem_logits)

        return fusion.output()


class SegmentationModule(nn.Module):

    def __init__(self, body, head, head_channels, classifier):
        super(SegmentationModule, self).__init__()
        self.body = body
        self.head = head
        self.head_channels = head_channels
        self.cls = classifier

    def forward(self, x, use_classifier=True, return_feat=False, return_body=False,
                only_classifier=False, only_head=False):

        if only_classifier:
            return self.cls(x)
        elif only_head:
            return self.cls(self.head(x))
        else:
            x_b = self.body(x)
            if isinstance(x_b, dict):
                x_b = x_b["out"]
            out = self.head(x_b)

            out_size = x.shape[-2:]

            if use_classifier:
                sem_logits = self.cls(out)
                sem_logits = functional.interpolate(sem_logits, size=out_size, mode="bilinear", align_corners=False)
            else:
                sem_logits = out

            if return_feat:
                if return_body:
                    return sem_logits, out, x_b
                return sem_logits, out

            return sem_logits

    def freeze(self):
        for par in self.parameters():
            par.requires_grad = False

    def fix_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, inplace_abn.ABN):
                m.eval()
                m.weight.requires_grad = False
                m.bias.requires_grad = False
