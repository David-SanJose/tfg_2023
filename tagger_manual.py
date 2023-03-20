import cv2

class TaggerManual():
    def __init__(self) -> None:
        self.listaTmp = []
        self.listaDef = []


    def on_click(self, event,x,y,flags,param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(x,y)
            tmp_len = len(self.listaTmp)

            if tmp_len <= 1:
                self.listaTmp.append((x,y))
            else:
                self.listaTmp[0] = self.listaTmp[1]
                self.listaTmp[1] = (x,y)
            
            print("LISTA TMP", self.listaTmp)
    
    def validate_coordinate(self):
        if len(self.listaTmp) == 2:
            print("Coordinada validada:", self.listaTmp)
            self.listaDef.append(self.listaTmp.copy())
            self.clean_last_coordinates()

        else:
            print("ERROR: no hay coordenadas:", self.listaTmp)

    def clean_last_coordinates(self):
        self.listaTmp.clear()

    def get_last_coordinates(self):
        return self.listaTmp
    def get_def_coordinates(self):
        return self.listaDef
