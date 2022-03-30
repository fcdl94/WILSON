import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn.functional as F
import torch.nn as nn

from functools import partial


#
# Helper modules
#
class LocalAffinity(nn.Module):

    def __init__(self, dilations=[1]):
        super(LocalAffinity, self).__init__()
        self.dilations = dilations
        weight = self._init_aff()
        self.register_buffer('kernel', weight)

    def _init_aff(self):
        # initialising the shift kernel
        weight = torch.zeros(8, 1, 3, 3)

        for i in range(weight.size(0)):
            weight[i, 0, 1, 1] = 1

        weight[0, 0, 0, 0] = -1
        weight[1, 0, 0, 1] = -1
        weight[2, 0, 0, 2] = -1

        weight[3, 0, 1, 0] = -1
        weight[4, 0, 1, 2] = -1

        weight[5, 0, 2, 0] = -1
        weight[6, 0, 2, 1] = -1
        weight[7, 0, 2, 2] = -1

        self.weight_check = weight.clone()

        return weight

    def forward(self, x):

        self.weight_check = self.weight_check.type_as(x)
        assert torch.all(self.weight_check.eq(self.kernel))

        B, K, H, W = x.size()
        x = x.view(B * K, 1, H, W)

        x_affs = []
        for d in self.dilations:
            x_pad = F.pad(x, [d] * 4, mode='replicate')
            x_aff = F.conv2d(x_pad, self.kernel, dilation=d)
            x_affs.append(x_aff)

        x_aff = torch.cat(x_affs, 1)
        return x_aff.view(B, K, -1, H, W)


class LocalAffinityCopy(LocalAffinity):

    def _init_aff(self):
        # initialising the shift kernel
        weight = torch.zeros(8, 1, 3, 3)

        weight[0, 0, 0, 0] = 1
        weight[1, 0, 0, 1] = 1
        weight[2, 0, 0, 2] = 1

        weight[3, 0, 1, 0] = 1
        weight[4, 0, 1, 2] = 1

        weight[5, 0, 2, 0] = 1
        weight[6, 0, 2, 1] = 1
        weight[7, 0, 2, 2] = 1

        self.weight_check = weight.clone()
        return weight


class LocalStDev(LocalAffinity):

    def _init_aff(self):
        weight = torch.zeros(9, 1, 3, 3)
        weight.zero_()

        weight[0, 0, 0, 0] = 1
        weight[1, 0, 0, 1] = 1
        weight[2, 0, 0, 2] = 1

        weight[3, 0, 1, 0] = 1
        weight[4, 0, 1, 1] = 1
        weight[5, 0, 1, 2] = 1

        weight[6, 0, 2, 0] = 1
        weight[7, 0, 2, 1] = 1
        weight[8, 0, 2, 2] = 1

        self.weight_check = weight.clone()
        return weight

    def forward(self, x):
        # returns (B,K,P,H,W), where P is the number
        # of locations
        x = super(LocalStDev, self).forward(x)

        return x.std(2, keepdim=True)


class LocalAffinityAbs(LocalAffinity):

    def forward(self, x):
        x = super(LocalAffinityAbs, self).forward(x)
        return torch.abs(x)


# PAMR module
class PAMR(nn.Module):

    def __init__(self, num_iter=10, dilations=[1, 2, 4, 8, 12, 24]):
        super(PAMR, self).__init__()

        self.num_iter = num_iter
        self.aff_x = LocalAffinityAbs(dilations)
        self.aff_m = LocalAffinityCopy(dilations)
        self.aff_std = LocalStDev(dilations)

    def forward(self, x, mask):
        mask = F.interpolate(mask, size=x.size()[-2:], mode="bilinear", align_corners=True)

        # x: [BxKxHxW]
        # mask: [BxCxHxW]
        B, K, H, W = x.size()
        _, C, _, _ = mask.size()

        x_std = self.aff_std(x)

        x = -self.aff_x(x) / (1e-8 + 0.1 * x_std)
        x = x.mean(1, keepdim=True)
        x = F.softmax(x, 2)

        for _ in range(self.num_iter):
            m = self.aff_m(mask)  # [BxCxPxHxW]
            mask = (m * x).sum(2)

        # xvals: [BxCxHxW]
        return mask

class GCI(nn.Module):
    """Global Cue Injection
    Takes shallow features with low receptive
    field and augments it with global info via
    adaptive instance normalisation"""

    def __init__(self, ch_deep, ch_shallow, norm=nn.BatchNorm2d):
        super(GCI, self).__init__()

        self.fc_deep = nn.Sequential(nn.Conv2d(ch_deep, 512, 1, bias=False),
                                     norm(512), nn.ReLU())

        # affine=False means no learnable parameters (weight and bias) but still running mean and variance
        self.fc_skip = nn.Sequential(nn.Conv2d(ch_shallow, 256, 1, bias=False),
                                     norm(256, affine=False))

        self.fc_cls = nn.Sequential(nn.Conv2d(256, 256, 1, bias=False),
                                    norm(256), nn.ReLU())

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight.data)
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.SyncBatchNorm):
                if m.weight is not None:  # affine=True
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

    def forward(self, x_shallow, x_deep):
        """Forward pass
        Args:
            x_shallow: shallow features
            x_deep: deep features
        """

        # extract global attributes (GMP)
        x_deep = self.fc_deep(x_deep)
        attrs, _ = x_deep.view(x_deep.size(0), x_deep.size(1), -1).max(-1)

        # pre-process shallow features
        x_shallow = self.fc_skip(x_shallow)
        x_shallow = F.relu(self._adin_conv(x_shallow, attrs))

        return self.fc_cls(x_shallow)

    def _adin_conv(self, x_shallow, attrs):
        bs, num_c, _, _ = x_shallow.size()
        assert 2 * num_c == attrs.size(1), "AdIN: dimension mismatch"

        attrs = attrs.view(bs, 2, num_c)
        gamma, beta = attrs[:, 0], attrs[:, 1]

        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)

        return x_shallow * (gamma + 1) + beta


class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation, BatchNorm):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                            stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn = BatchNorm(planes)

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return x


class ASPP(nn.Module):
    def __init__(self, inplanes, output_stride, norm):
        super(ASPP, self).__init__()

        if output_stride == 16:
            dilations = [1, 6, 12, 18]
        elif output_stride == 8:
            dilations = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        self.aspp1 = _ASPPModule(inplanes, 256, 1, padding=0, dilation=dilations[0], BatchNorm=norm)
        self.aspp2 = _ASPPModule(inplanes, 256, 3, padding=dilations[1], dilation=dilations[1], BatchNorm=norm)
        self.aspp3 = _ASPPModule(inplanes, 256, 3, padding=dilations[2], dilation=dilations[2], BatchNorm=norm)
        self.aspp4 = _ASPPModule(inplanes, 256, 3, padding=dilations[3], dilation=dilations[3], BatchNorm=norm)

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(inplanes, 256, 1, stride=1, bias=False),
                                             norm(256))

        self.conv1 = nn.Conv2d(1280, 256, 1, bias=False)
        self.bn1 = norm(256)

        self._init_weight()

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)

        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                if not m.weight is None:
                    m.weight.data.fill_(1)
                else:
                    print("ASPP has not weight: ", m)

                if not m.bias is None:
                    m.bias.data.zero_()
                else:
                    print("ASPP has not bias: ", m)

class StochasticGate(nn.Module):
    """Stochastically merges features from two levels
    with varying size of the receptive field
    """

    def __init__(self):
        super(StochasticGate, self).__init__()
        self._mask_drop = None

    def forward(self, x1, x2, alpha_rate=0.3):
        """Stochastic Gate (SG)
        SG stochastically mixes deep and shallow features
        at training time and deterministically combines
        them at test time with a hyperparam. alpha
        """
        # for each nn.Module
        # .train() set self.training = True
        # .eval() set self.training = False
        # SG has a different behaviour depending on the stage (training or inference)

        # training time
        if self.training:
            # dropout: selecting either x1 or x2
            if self._mask_drop is None:
                bs, c, h, w = x1.size()
                assert c == x2.size(1), "Number of features is different"
                self._mask_drop = torch.ones_like(x1)

            # a mask of {0,1}
            mask_drop = (1 - alpha_rate) * F.dropout(self._mask_drop, alpha_rate)

            # shift and scale deep features
            # at train time: E[x] = x1
            x1 = (x1 - alpha_rate * x2) / max(1e-8, 1 - alpha_rate)

            # combine the features
            x = mask_drop * x1 + (1 - mask_drop) * x2
        # inference time: deterministic
        else:
            x = (1 - alpha_rate) * x1 + alpha_rate * x2

        return x
