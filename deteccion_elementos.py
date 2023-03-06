import cv2
import numpy as np



UMBRAL_AREA_COCHE = 750
UMBRAL_AREA_MOTO = 10
UMBRAL_AREA_PEATON = 20
umbral_color_coche = (np.array([0, 0, 10]) , np.array([255, 255, 10]))
umbral_color_peaton = (np.array([0, 0, 4]) , np.array([255, 255, 4]))
mostrar = True

class Deteccion:
    def __init__(self) -> None:
        
        self.lista_box_coches = []
        self.lista_box_peatones = []
        self.lista_box_motos = []
        self.lista_posibles_motos = []
        self.lista_tmp_motos = []


    def getRectCar(self, img_SEG):

        mask_coche = cv2.inRange(img_SEG, umbral_color_coche[0], umbral_color_coche[1])
        contornos_coche, _= cv2.findContours(mask_coche, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contornos_coche:
            box = cv2.boundingRect(cnt)
            box_area = box[2] * box[3]
            if box_area > UMBRAL_AREA_COCHE:
                self.lista_box_coches.append(box)
            elif box_area > UMBRAL_AREA_MOTO:
                self.lista_posibles_motos.append(box)
                    
    def getRectBikesAndPedestrians(self, img_SEG):

        mask_peaton = cv2.inRange(img_SEG, umbral_color_peaton[0], umbral_color_peaton[1])
        contornos_peaton, _= cv2.findContours(mask_peaton, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        #Se obtienen los posibles peatones
        lista_posibles_peatones = []
        for cnt in contornos_peaton:
            box_peaton = cv2.boundingRect(cnt)
            box_area_peaton = box_peaton[2] * box_peaton[3]
            if box_area_peaton > UMBRAL_AREA_PEATON:
                lista_posibles_peatones.append(box_peaton)
        
        for pos_peaton in lista_posibles_peatones:
            isAPeaton = True
            for pos_moto in self.lista_posibles_motos:
                res = self.box_intersection(pos_moto, pos_peaton)
                if res:
                    box_moto = self.box_union(pos_moto, pos_peaton)
                    self.lista_tmp_motos.append(box_moto)
                    isAPeaton = False
            
            if isAPeaton:
                self.lista_box_peatones.append(pos_peaton)

    def simplificar_motos(self):
        motos_size = len(self.lista_tmp_motos)
        if motos_size > 0:
            lista_tmp = []
            i = 0
            j = 1

            while i < motos_size:
                a = self.lista_tmp_motos[i]
                
                if j < motos_size:
                    b = self.lista_tmp_motos[j]
                    if self.box_intersection(a, b):
                        self.lista_tmp_motos[i] = self.box_union(a,b)
                        j += 1
                    else:
                        lista_tmp.append(a)
                        i = j
                        j += 1
                else:
                    lista_tmp.append(a)
                    i = motos_size
            self.lista_box_motos = lista_tmp
        
    def mostrar(self, img_RGB):
        img_RGB_copy = img_RGB.copy()

        for box in self.lista_box_coches:
            img_RGB_copy = cv2.rectangle(img_RGB_copy, (box[0], box[1]), 
                                    (box[0]+ box[2], box[1]+ box[3]), (255, 0, 0), 2)
            
        for box in self.lista_box_peatones:
            img_RGB_copy = cv2.rectangle(img_RGB_copy, (box[0], box[1]), 
                                (box[0]+ box[2], box[1]+ box[3]), (0, 255, 0), 2)

        for box in self.lista_box_motos:
            img_RGB_copy = cv2.rectangle(img_RGB_copy, (box[0], box[1]), 
                                (box[0]+ box[2], box[1]+ box[3]), (0, 100, 255), 2)
            

            
        cv2.imshow("IMAGEN", img_RGB_copy)

        key_pressed = cv2.waitKey(0)
        if key_pressed == -1: exit(0)

    def clear_listas(self):
        self.lista_box_peatones.clear()
        self.lista_box_motos.clear()
        self.lista_box_coches.clear()
        self.lista_posibles_motos.clear()

        self.lista_tmp_motos.clear()


    def box_union(self, a, b):
        x = min(a[0], b[0])
        y = min(a[1], b[1])
        w = max(a[0]+a[2], b[0]+b[2]) - x
        h = max(a[1]+a[3], b[1]+b[3]) - y
        return (x, y, w, h)

    def box_intersection(self,a,b):
        x = max(a[0], b[0])
        y = max(a[1], b[1])
        w = min(a[0]+a[2], b[0]+b[2]) - x
        h = min(a[1]+a[3], b[1]+b[3]) - y
        if w<0 or h<0: return () # or (0,0,0,0) ?
        return (x, y, w, h)

'''
Tomamos ambas imagenes
Se localizan los vehiculos y peatones, almacenandose en listas
Se recorre la lista de vehiculos, 
    si es coche se procesa
    si no, se añade a lista de posibles motos
Se recorre la lista de peatones
    Se recorre la lista de posibles motos:
        si colisiona con box de moto:
            Se añade a la lista de motos
            Se elimina la moto de la lista de posibles
            Se deja de recorrer la lista de posibles motos
        si no:
            Se procesa el peaton




'''
