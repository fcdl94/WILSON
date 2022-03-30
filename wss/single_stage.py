import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules import PAMR, GCI, StochasticGate, ASPP
from segmentation_module import get_norm
from utils.utils import denorm
from modules import DeeplabV3


def _rescale_and_clean(masks, image, labels):
    """Rescale to fit the image size and remove any masks
    of labels that are not present"""
    masks = F.interpolate(masks, size=image.size()[-2:], mode='bilinear', align_corners=True)
    masks[:, 1:] *= labels[:, :, None, None].type_as(masks)
    return masks


def pseudo_gtmask(mask, ambiguous=True, cutoff_top=0.6, cutoff_bkg=0.6, cutoff_low=0.2, eps=1e-8, old_classes=16):
    """Convert continuous mask into binary mask"""
    bs, c, h, w = mask.size()
    mask = mask.view(bs, c, -1)

    # for each class extract the max confidence
    mask_max, _ = mask.max(-1, keepdim=True)
    mask_max[:, :1] *= cutoff_bkg
    mask_max[:, 1:] *= cutoff_top
    # mask_max *= cutoff_top

    # if the top score is too low, ignore it
    lowest = torch.Tensor([cutoff_low]).type_as(mask_max)
    mask_max = mask_max.max(lowest)

    pseudo_gt = (mask > mask_max).type_as(mask)

    # remove ambiguous pixels
    if ambiguous:
        ambiguous = (pseudo_gt.sum(1, keepdim=True) > 1).type_as(mask)
        pseudo_gt = (1 - ambiguous) * pseudo_gt

    return pseudo_gt.view(bs, c, h, w)


def balanced_mask_loss_ce(mask, pseudo_gt, gt_labels, ignore_index=255, old_classes=16):
    """Class-balanced CE loss
    - cancel loss if only one class in pseudo_gt
    - weight loss equally between classes
    """

    # b,c,h,w = pseudo_gt.shape
    mask = F.interpolate(mask, size=pseudo_gt.size()[-2:], mode="bilinear", align_corners=True)

    # indices of the max classes
    mask_gt = torch.argmax(pseudo_gt, 1)

    # for each pixel there should be at least one 1
    # otherwise, ignore
    ignore_mask = pseudo_gt.sum(1) < 1.
    mask_gt[ignore_mask] = ignore_index

    # class weight balances the loss w.r.t. number of pixels
    # because we are equally interested in all classes
    bs, c, h, w = pseudo_gt.size()
    num_pixels_per_class = pseudo_gt.view(bs, c, -1).sum(-1)  # BS, C -> pixel per class
    num_pixels_total = num_pixels_per_class.sum(-1, keepdim=True)  # BS -> pixel per image
    class_weight = (num_pixels_total - num_pixels_per_class) / (1 + num_pixels_total)  # BS, C
    class_weight = (pseudo_gt * class_weight[:, :, None, None]).sum(1).view(bs, -1)  # BS, H, W

    # BCE loss
    loss = F.cross_entropy(mask, mask_gt, ignore_index=ignore_index, reduction="none")
    loss = loss.view(bs, -1)

    # we will have the loss only for batch indices
    # which have all classes in pseudo mask
    gt_num_labels = gt_labels.sum(-1).type_as(loss) + 1  # + BG
    ps_num_labels = (num_pixels_per_class > 0).type_as(loss).sum(-1)
    batch_weight = (gt_num_labels == ps_num_labels).type_as(loss)

    loss = batch_weight * (class_weight * loss).mean(-1)
    return loss.mean()


def balanced_mask_loss_unce(mask, pseudo_gt, gt_labels, old_cl, ignore_index=255):
    """Class-balanced CE loss
    - cancel loss if only one class in pseudo_gt
    - weight loss equally between classes
    """

    mask = F.interpolate(mask, size=pseudo_gt.size()[-2:], mode="bilinear", align_corners=True)
    mask_gt = torch.argmax(pseudo_gt, 1)

    # for each pixel there should be at least one 1
    # otherwise, ignore
    ignore_mask = pseudo_gt.sum(1) < 1.
    mask_gt[ignore_mask] = ignore_index

    # class weight balances the loss w.r.t. number of pixels
    # because we are equally interested in all classes
    bs, c, h, w = pseudo_gt.size()
    num_pixels_per_class = pseudo_gt.view(bs, c, -1).sum(-1)  # BS, C -> pixel per class
    num_pixels_total = num_pixels_per_class.sum(-1, keepdim=True)  # BS -> pixel per image
    class_weight = (num_pixels_total - num_pixels_per_class) / (1 + num_pixels_total)  # BS, C
    class_weight = (pseudo_gt * class_weight[:, :, None, None]).sum(1).view(bs, -1)  # BS, H, W

    # BCE loss
    # den = torch.logsumexp(mask, dim=1)  # softmax denominator
    # num = torch.logsumexp(mask*pseudo_gt, dim=1)  # use all the positive classes for optimization
    # loss = F.nll_loss((num - den).unsqueeze(1), mask_gt, reduction='none', ignore_index=ignore_index)
    # loss = loss.view(bs, -1)

    outputs = torch.zeros_like(mask)  # B, C (1+V+N), H, W
    den = torch.logsumexp(mask, dim=1)  # B, H, W       den of softmax
    outputs[:, 0] = torch.logsumexp(mask[:, 0:old_cl], dim=1) - den  # B, H, W       p(O)
    outputs[:, old_cl:] = mask[:, old_cl:] - den.unsqueeze(dim=1)  # B, N, H, W    p(N_i)

    loss = F.nll_loss(outputs, mask_gt, ignore_index=ignore_index, reduction="none").view(bs, -1)

    gt_num_labels = gt_labels.sum(-1).type_as(loss) + 1  # + BG
    ps_num_labels = (num_pixels_per_class > 0).type_as(loss).sum(-1)
    batch_weight = (gt_num_labels == ps_num_labels).type_as(loss)

    loss = batch_weight * (class_weight * loss).mean(-1)
    return loss.mean()


class Single_Stage(nn.Module):

    def __init__(self, channels, classes, opts):
        super(Single_Stage, self).__init__()

        self.affinity = PAMR()
        norm = get_norm(opts)

        self.head = ASPP(channels, opts.output_stride, norm)
        channels = 256

        ch_skip = 48
        ch_sh = 256 if "wide" in opts.backbone else 512
        self.fc8_skip = nn.Sequential(nn.Conv2d(ch_sh, ch_skip, 1, bias=False), norm(ch_skip))
        self.fc8_x = nn.Sequential(nn.Conv2d(ch_skip+channels, 256, kernel_size=3, stride=1, padding=1, bias=False), norm(256))

        self.gci = GCI(256, ch_sh, torch.nn.SyncBatchNorm)
        self.sg = StochasticGate()
        channels = 256

        self.cls = nn.Conv2d(256, classes, kernel_size=1, stride=1)
        self.last_conv = nn.Sequential(nn.Conv2d(channels, 256, kernel_size=3, stride=1, padding=1, bias=False), norm(256),
                                       nn.Dropout(0.5),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False), norm(256),
                                       nn.Dropout(0.1))
        self.init_pars()

        self.SG_PSI = 0.3
        self.pretrain_epoch = 5
        self.segm_weight = 1.0
        self.cur_epoch = 0
        self.bkg_disc = 3

        self.use_labels_val = True
        self.classes = classes

        pos_weight = 1
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.full([classes], fill_value=pos_weight))
        self.fusion = 'mean'

    def set_epoch(self, epoch):
        self.cur_epoch = epoch

    def adjust_mask(self, mask):
        mask[:, 0] = mask[:, 0].pow(self.bkg_disc)
        return mask

    def init_pars(self):
        modules = [self.fc8_skip, self.fc8_x, self.head, self.last_conv, self.cls]
        # modules = [self.last_conv, self.cls]
        for module in modules:
            for m in module.modules():
                if isinstance(m, nn.Conv2d):
                    torch.nn.init.kaiming_normal_(m.weight)

    def setup_optimizer(self, opts):
        return [{"params": filter(lambda p: p.requires_grad, self.last_conv.parameters()),
                   'lr': opts.lr, 'weight_decay': opts.weight_decay},
                  {"params": filter(lambda p: p.requires_grad, self.cls.parameters()),
                   'lr': opts.lr, 'weight_decay': opts.weight_decay},
                  {"params": filter(lambda p: p.requires_grad, self.gci.parameters()),
                   'lr': opts.lr, 'weight_decay': opts.weight_decay},
                  {"params": filter(lambda p: p.requires_grad, self.head.parameters()),
                   'lr': opts.lr, 'weight_decay': opts.weight_decay},
                  {"params": filter(lambda p: p.requires_grad, self.fc8_skip.parameters()),
                   'lr': opts.lr, 'weight_decay': opts.weight_decay},
                  {"params": filter(lambda p: p.requires_grad, self.fc8_x.parameters()),
                   'lr': opts.lr, 'weight_decay': opts.weight_decay}]

    def compute_masks(self, features):
        x_shallow, x_deep = features['b3'], features['body']
        x_deep = self.head(x_deep)

        # merging deep and shallow features
        # a) skip connection for deep features
        x2_shallow = self.fc8_skip(x_shallow)
        x_up = self.rescale_as(x_deep, x2_shallow)
        x_deep = self.fc8_x(torch.cat([x_up, x2_shallow], 1))
        # b) deep feature context for shallow features
        x_shallow = self.gci(x_shallow, x_deep)
        # # c) stochastically merging the masks
        x_deep = self.sg(x_deep, x_shallow, alpha_rate=self.SG_PSI)

        # final convs to get the masks
        x_deep = self.last_conv(x_deep)
        x = self.cls(x_deep)

        return x

    def forward_train(self, images, features, labels):
        # == MASKS PREDICTION
        logits = self.compute_masks(features)  # only #class as channels (no background predicted)

        # refining masks with background
        bg = torch.ones_like(logits[:, :1])
        logits = torch.cat([bg, logits], 1)

        bs, c, h, w = logits.size()
        features = logits.view(bs, c, -1)
        masks = F.softmax(logits, dim=1)
        masks_ = masks.view(bs, c, -1)

        # classification loss (normalized Global Weighted Pooling nGWP)
        y_ngwp = (features * masks_).sum(-1) / (1.0 + masks_.sum(-1))

        # focal penalty loss
        y_focal = torch.pow(1 - masks_.mean(-1), 3) * torch.log(0.01 + masks_.mean(-1))

        # adding the losses together
        y = y_ngwp[:, 1:] + y_focal[:, 1:]  # background excluded from the loss (not trained)

        # == MASKS REFINEMENT (AFFINITY)
        total_loss = self.criterion(y, labels.float())

        if self.cur_epoch >= self.pretrain_epoch:
            # mask refinement with PAMR
            image_raw = denorm(images.clone())
            im = F.interpolate(image_raw, masks.detach().size()[-2:], mode="bilinear", align_corners=True)
            masks_dec = self.affinity(im, masks.detach())

            # == SEGMENTATION
            # upscale the masks & clean
            # masks = self._rescale_and_clean(masks, x, labels)
            masks_dec = _rescale_and_clean(masks_dec, images, labels)

            # create pseudo GT (removing activations under threshold)
            pseudo_gt = pseudo_gtmask(masks_dec).detach()
            # segmentation loss
            loss_mask = balanced_mask_loss_ce(logits, pseudo_gt, labels)

            total_loss += loss_mask.mean() * self.segm_weight

        return logits, total_loss

    def forward_val(self, images, features, labels, out_size):
        # == MASKS PREDICTION
        logits = self.compute_masks(features)  # emette solo 20 classi

        # refining masks with background
        bg = torch.ones_like(logits[:, :1])
        masks = torch.cat([bg, logits], 1)
        if out_size is not None:
            masks = F.interpolate(masks, size=out_size, mode="bilinear", align_corners=False)

        # this is super useful, since we'll filter bad predictions on train.
        masks = F.softmax(masks, dim=1)
        masks[:, 1:] *= labels.view(-1, self.classes, 1, 1)

        # note: the masks contain the background as the first channel
        return masks

    def forward_inference(self, images, features, labels=None):
        # == MASKS PREDICTION
        logits = self.compute_masks(features)  # emette solo 20 classi

        if labels is not None:
            logits *= labels.view(-1, self.classes, 1, 1)

        # refining masks with background
        bg = torch.ones_like(logits[:, :1])
        masks = torch.cat([bg, logits], 1)
        # masks = nn.functional.interpolate(masks, size=(x.shape[2], x.shape[3]), mode="bilinear")
        masks = F.softmax(masks, dim=1)
        if labels is not None:
            masks *= labels.view(-1, self.classes, 1, 1)

        # note: the masks contain the background as the first channel
        return masks

    def freeze(self):
        for par in self.parameters():
            par.requires_grad = False

    def forward(self, images, features, labels, out_size=None, train=True):
        if train:
            return self.forward_train(images, features, labels)
        else:
            return self.forward_val(images, features, labels, out_size)

    def rescale_as(self, x, y, mode="bilinear", align_corners=True):
        h, w = y.size()[2:]
        x = F.interpolate(x, size=[h, w], mode=mode, align_corners=align_corners)
        return x