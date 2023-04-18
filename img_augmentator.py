import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

import cv2

import numpy as np
from PIL import Image

seqRot90 = iaa.Sequential([
    iaa.Rotate(90)
])

seqFlipH = iaa.Sequential([
    iaa.Fliplr(1)
])

seqFlipV = iaa.Sequential([
    iaa.Flipud(1)
])

seqRot180 = iaa.Sequential([
    iaa.Rotate(180)
])

seqRot270 = iaa.Sequential([
    iaa.Rotate(270)
])

def aug_test(image, all_boxes):

    lista_boxes = []
    for c, boxes_tmp in enumerate(all_boxes):
        for box in boxes_tmp:
            obj = BoundingBox(x1=box[0], y1=box[1],x2=box[0]+box[2], y2=box[1]+box[3], label=c)
            lista_boxes.append(obj)
    
    bbs = BoundingBoxesOnImage(lista_boxes, shape=image.shape)
    
    lista_aumentaciones = [seqRot90, seqFlipH, seqFlipV, seqRot180, seqRot270]
    images_with_boxes = [(image, bbs)]
    for aument in lista_aumentaciones:
        image_aug, bbs_aug = aument(image=image, bounding_boxes=bbs)
        #Se eliminan boxes exteriores
        bbs_aug = bbs_aug.remove_out_of_image().clip_out_of_image()

        images_with_boxes.append((image_aug, bbs_aug))
    
    return images_with_boxes
