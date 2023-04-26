
from ultralytics import YOLO

import cv2
import os


ruta = "..\collab\content\\runs\detect\\train2\weights\\best.pt"
imagen = "..\imagenes\img2\Aumentadas_check\\"
model = YOLO(ruta)


files_list = os.listdir(imagen)


for file in files_list:
    if file.endswith(".png"):
        results = model(imagen+file)

        res_plotted = results[0].plot()

        res_plotted = cv2.resize(res_plotted, (960, 540))
        cv2.imshow("result", res_plotted)
        cv2.waitKey(0)

'''
model = YOLO('yolov8n.pt')


results = model.train(data='custom.yaml', epochs=3, device=0)  # train the model
results = model.val()  # evaluate model performance on the validation set

for r in results:
    ploted = r.plot()
    cv2.imshow("result", ploted)
    cv2.waitKey(0)

model = YOLO('yolov8n.pt') # pass any model type
#results = model.train(data='coco128.yaml', epochs=3)
#results = model.val()  # evaluate model performance on the validation set
results = model('https://ultralytics.com/images/bus.jpg')

res_plotted = results[0].plot()
cv2.imshow("result", res_plotted)
cv2.waitKey(0)
'''
