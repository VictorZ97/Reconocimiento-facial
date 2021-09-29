# Librerias requeridas
import cv2
import matplotlib
import matplotlib.pyplot
import imutils
from mtcnn.mtcnn import MTCNN
import os

# Carpeta de almacenamiento
nombre = 'Victor_Tapabocas' #Nombre de la carpeta que se creará
direccion = 'D:\Proyectos\Reconocimiento facial Con IA\RecFac\Rostros' # <-- En esta parte pondras la carpeta donde lo estes guardando
carpeta = direccion + '/' + nombre

# Creamos la carpeta en el caso de que no exista
if not os.path.exists(carpeta):
    os.makedirs(carpeta)

# Creamos el video en tiempo real
director =  MTCNN()   #Creamos el objeto a detectar
cap = cv2.VideoCapture(0)
count = 0

while True:
    ret, frame = cap.read()
    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    copia = frame.copy()

    caras = director.detect_faces(copia)
    for i in range(len(caras)):
        x1, y1, ancho, alto = caras[i]['box']
        x2, y2 = x1 + ancho, y1 + alto
        cara_reg = frame[y1:y2, x1:x2]
        cara_reg = cv2.resize(cara_reg, (150,200), interpolation = cv2.INTER_CUBIC)
        cv2.imwrite(carpeta + "/rostro_{}.jpg".format(count) ,cara_reg)
        count = count + 1
    cv2.imshow("Entrenamiento", frame)

    t = cv2.waitKey(1)
    if t == 27 or count >= 300:
        break
cap.release()
cv2.destoyAllWindows()
