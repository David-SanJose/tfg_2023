
def imgaug_to_yolo(box, img):
        x, y, w, h = box
        img_h, img_w = img.shape[:2]

        x_centre = ((x + w) / 2) / img_w
        y_centre = ((y + h) / 2) / img_h
        new_width = ((w-x) * 1.1) / img_w
        new_height = ((h-y) * 1.1) / img_h

        return (x_centre, y_centre, new_width, new_height)