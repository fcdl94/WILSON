from PIL import Image
import numpy as np
import os
import os.path as osp
from tqdm import tqdm

orig_path = '.'  # insert here the source path where original COCO annotations are
dest_path = '.'  # the destination folder, which should be this one'

for split in ["train2017", "val2017"]:
    annotations = f"{orig_path}/annotations/{split}"
    nov_ann = f"{dest_path}/annotations_my/{split}"

    # clear folder if exists
    if osp.exists(nov_ann):
        print("Removing existing")
        os.rmdir(nov_ann)
    os.makedirs(nov_ann)

    # remap labels in the novel interval (+1 for Stuff, +1 and stuff on 0 for objects)
    mapping = np.zeros((256,), dtype=np.int8)
    for i, cl in enumerate(range(91)):
        mapping[cl] = i + 1
    mapping[255] = 255
    target_transform = lambda x: Image.fromarray(mapping[x])

    for f in tqdm(os.listdir(annotations)):
        lbl = Image.open(osp.join(annotations, f))
        lbl = target_transform(lbl)
        lbl.save(osp.join(nov_ann, f))
