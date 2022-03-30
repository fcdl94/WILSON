#!/usr/bin/env bash

# use this script in the destination folder.
# such as add cd /home/datasets

# Use this structure
# images/
#   train2017
#   val2017
# annotations/ (COCO format -> 0 for person, not bkg)
#   train2017
#   val2017
# after downloading, convert to object or stuff using the python scripts provided in data/{dataset}/make_annotation.py


mkdir images
mkdir annotations
wget http://images.cocodataset.org/zips/train2017.zip
unzip train2017.zip -d images
wget http://images.cocodataset.org/zips/val2017.zip
unzip val2017.zip -d images
wget http://calvin.inf.ed.ac.uk/wp-content/uploads/data/cocostuffdataset/stuffthingmaps_trainval2017.zip
unzip stuffthingmaps_trainval2017.zip -d annotations