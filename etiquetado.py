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
import yolo_box_transformer as ybt

UMBRAL_AREA_COCHE = 750

#LEER ARCHIVO DE LISTA
rutaBase= "..\\imagenes\\img2"
lista_f = open(f"{rutaBase}\\lista.csv")

mostrar = True
interfaz_g = False

de = Deteccion()

def save_img_and_boxes(image, boxes, name, path):
    full_name = f"{path}\\{name}"

    cv2.imwrite(f"{full_name}.png", image)
    #array_boxes = boxes.to_xyxy_array()

    f = open(f"{full_name}.txt", "w")
    for box in boxes:
        tmp_array = [box.x1_int, box.y1_int, box.x2_int, box.y2_int]
        yolobox = ybt.imgaug_to_yolo(tmp_array, image)

        f_box = [ '%.6f' % elem for elem in yolobox ]
        f_text = " ".join(f_box)
        f.write(f"{box.label} {f_text}\n")


lista_txt = lista_f.readlines()

for c, row in enumerate(lista_txt):
    #Se extraen los nombres de las imagenes
    (img_RGB_name, img_SEG_name) = row.split(";")[1:3]

    #Se cargan las imagenes
    img_RGB = cv2.imread(f'{rutaBase}\\RGB\\{img_RGB_name}.png')
    img_SEG = cv2.imread(f'{rutaBase}\\SEG\\{img_SEG_name}.tiff')
    
    #Fase inicial
    if c == 0:
        if interfaz_g:
            de.setObstacles(img_RGB)
            de.setPedestrianLines(img_RGB)
        else:
            de.readObstaclesFromFile()
            de.readPedestrianLinesFromFile()
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
    for c2 , (img_aug, boxes) in enumerate(images_aug_with_boxes):
        
        nombre_aug = f"{img_RGB_name}_{c2}"
        print(nombre_aug)
        ruta_aug = "..\\imagenes\\img2\\Aumentadas"
        if c2 == 1:
            ruta_aug = "..\\imagenes\\img2\\Aumentadas_check"
        save_img_and_boxes(img_aug, boxes, nombre_aug, ruta_aug)
        if mostrar:
            img_aug_tmp = boxes.draw_on_image(img_aug, size=2, color=[0, 0, 255])
            
            img_aug_h = img_aug_tmp.shape[0]//2
            img_aug_w = img_aug_tmp.shape[1]//2
            img_aug_tmp = cv2.resize(img_aug_tmp, (img_aug_w, img_aug_h))
            
            cv2.imshow("AUG",img_aug_tmp)
            key_pressed = cv2.waitKey()
            if key_pressed == -1: exit(0)


    de.save_to_yolo(f"{rutaBase}\\RGB",img_RGB_name,img_SEG)
    if mostrar:
        de.mostrar(img_RGB)
    print(f"{c} / {len(lista_txt)}")
    
    #aumentation software



    