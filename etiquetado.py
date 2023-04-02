'''
ACCIONES A REALIZAR

Leer el archivo de lista.csv
Por cada fila:
    Extraer nombres RGB y SEG
    Cargar RGB y SEG
    Detectar en la segmentación coches y almacenar en lista rectangulos
    Mostrar la RGB con los rectangulos de deteccion
    Almacenar en un fichero los limites de cada caja
'''
import cv2
import numpy as np
from deteccion_elementos import Deteccion
import img_augmentator

UMBRAL_AREA_COCHE = 750

#LEER ARCHIVO DE LISTA
rutaBase= "..\\imagenes\\img2"
lista_f = open(f"{rutaBase}\\lista.csv")

mostrar = True

de = Deteccion()

lista_txt = lista_f.readlines()

for c, row in enumerate(lista_txt):
    #Se extraen los nombres de las imagenes
    (img_RGB_name, img_SEG_name) = row.split(";")[1:3]

    #Se cargan las imagenes
    img_RGB = cv2.imread(f'{rutaBase}\\RGB\\{img_RGB_name}.png')
    img_SEG = cv2.imread(f'{rutaBase}\\SEG\\{img_SEG_name}.tiff')
    
    #Fase inicial
    if c == 0:
        de.setObstacles(img_RGB)
        de.setPedestrianLines(img_RGB)
    print(img_SEG_name)

    de.clear_listas()
    de.getRectCar(img_SEG)
    de.getRectBikesAndPedestrians(img_SEG)
    de.simplificar_motos()
    de.eliminar_peatones_en_obstaculos()

    all_boxes = [de.lista_box_coches,
        de.lista_box_motos,
        de.lista_box_peatones,
        de.lista_pedestrian_lines]
    
    images_aug_with_boxes = img_augmentator.aug_test(img_RGB, all_boxes)
    for img_aug, boxes in images_aug_with_boxes:
        img_aug = boxes.draw_on_image(img_aug, size=2, color=[0, 0, 255])

        print("SHAPE",img_aug.shape, img_RGB.shape)
        img_aug_h = img_aug.shape[0]//2
        img_aug_w = img_aug.shape[1]//2
        img_aug = cv2.resize(img_aug, (img_aug_w, img_aug_h))
        
        cv2.imshow("AUG",img_aug)
        cv2.waitKey(1000)
    
    
    
    de.save_to_yolo(f"{rutaBase}\\RGB",img_RGB_name,img_SEG)
    de.mostrar(img_RGB)
    print(f"{c} / {len(lista_txt)}")
    
    #aumentation software



    