
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

