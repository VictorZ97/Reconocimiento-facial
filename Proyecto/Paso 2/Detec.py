# Librerias requeridas
import cv2
import os
import numpy as np

# Importaremos las fotos creadas anteriormente
direccion = 'D:\Proyectos\Reconocimiento facial Con IA\RecFac\Rostros' # <-- En esta parte pondras la carpeta donde lo estes guardando
lista = os.listdir(direccion)

etiquetas = []
rostros = []
con = 0

for nameDir in lista:
    nombre = direccion + '/' + nameDir                          #Leeremos las fotos tomadas de los rostros

    for filename in os.listdir(nombre):                         #Se asignaran las etiquetas a las fotos
        etiquetas.append(con)                                   #Valor de la etiqueta
        rostros.append(cv2.imread(nombre + '/' + filename,0))   #Añadimos las imagenes en EDG

    con = con + 1

#Creamos el modelo
reconocimiento = cv2.face.LBPHFaceRecognizer_create()

#Se entrena con las fotos tomadas previamente
reconocimiento.train(rostros,np.array(etiquetas))

#Se guarda el modelo
reconocimiento.write('modeloLBP.xml')
print("Modelo Creado")