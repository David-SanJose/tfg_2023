import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

import cv2

import numpy as np
from PIL import Image

seqRot90 = iaa.Sequential([
    iaa.Rot90(1, keep_size=False)
])

seqFlipH = iaa.Sequential([
    iaa.Fliplr(1)
])

seqFlipV = iaa.Sequential([
    iaa.Flipud(1)
])

def aug_test(image, all_boxes):

    lista_boxes = []
    for c, boxes_tmp in enumerate(all_boxes):
        for box in boxes_tmp:
            obj = BoundingBox(x1=box[0], y1=box[1],x2=box[0]+box[2], y2=box[1]+box[3], label=c)
            lista_boxes.append(obj)
    
    bbs = BoundingBoxesOnImage(lista_boxes, shape=image.shape)
    
    lista_aumentaciones = [seqRot90, seqFlipH, seqFlipV]
    images_with_boxes = []
    for aument in lista_aumentaciones:
        image_aug, bbs_aug = aument(image=image, bounding_boxes=bbs)
    #image_flipH, bbs_flipH = seqFlipH(image=image, bounding_boxes=bbs)
        images_with_boxes.append((image_aug, bbs_aug))
    #images_with_boxes.append((image_flipH, bbs_flipH))
    return images_with_boxes
