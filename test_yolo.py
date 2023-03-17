
from ultralytics import YOLO

import cv2

model = YOLO('yolov8n.pt') # pass any model type
#results = model.train(data='coco128.yaml', epochs=3)
#results = model.val()  # evaluate model performance on the validation set
results = model('https://ultralytics.com/images/bus.jpg')

res_plotted = results[0].plot()
cv2.imshow("result", res_plotted)
cv2.waitKey(0)
