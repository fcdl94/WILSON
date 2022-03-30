import os
import numpy as np
from dataset.voc import VOCSegmentation
from PIL import Image
from tqdm import tqdm
import shutil

coco_map = {
    0: 0,
    1: 5,
    2: 2,
    3: 16,
    4: 9,
    5: 44,
    6: 6,
    7: 3,
    8: 17,
    9: 62,
    10: 21,
    11: 67,
    12: 18,
    13: 19,
    14: 4,
    15: 1,
    16: 64,
    17: 20,
    18: 63,
    19: 7,
    20: 72,
    255: 255
}


class LabelTransform:
    def __init__(self, mapping):
        self.mapping = mapping

    def __call__(self, x):
        return Image.fromarray(self.mapping[x])


mapping = np.zeros((256,), dtype=np.uint8)
for k,v in coco_map.items():
    mapping[k] = v
remap = LabelTransform(mapping)

data = VOCSegmentation('data', train=True)
os.makedirs(f"data/voc/SegmentationClassAugAsCoco", exist_ok=True)

split_txt = []
for ip, lp in tqdm(data.images):
    lbl = Image.open(lp)
    idx = lp.split("/")[-1][:-4]
    new_path = f"data/voc/SegmentationClassAugAsCoco/{idx}.png"
    new_lbl = remap(lbl)
    new_lbl.save(new_path, "PNG")
