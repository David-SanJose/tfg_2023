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
    de.save_to_yolo(f"{rutaBase}\\RGB",img_RGB_name,img_SEG)
    de.mostrar(img_RGB)
    print(f"{c} / {len(lista_txt)}")
    



    