# Librerias requeridas
import cv2
from mtcnn.mtcnn import MTCNN
import os

# Importaremos los nombres
direccion = 'D:\Proyectos\Reconocimiento facial Con IA\RecFac\Rostros'  # <-- En esta parte pondras la carpeta donde lo estes guardando
dire_img = os.listdir(direccion)
print ('Nombres: ', dire_img)

#Llamamos al modelo de reconocimiento
reconocimiento = cv2.face.LBPHFaceRecognizer_create()

#Leemos el modelo
reconocimiento.read('modeloLBP.xml')

#Capturamos el video en tiempo real
detector = MTCNN()          #Creamos el objeto a detectar
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if ret == False: break
    gris = cv2.cvtColor(frame , cv2.COLOR_BGR2GRAY)
    copia = frame.copy()
    copia2 = gris.copy()
    caras = detector.detect_faces(copia)

    for i in range(len(caras)) :
        x1, y1, ancho, alto = caras[i]['box']
        x2, y2 = x1 + ancho, y1 + alto
        cara_reg = copia2[y1:y2, x1:x2]
        cara_rec = cv2.resize(cara_reg, (150,200), interpolation = cv2.INTER_CUBIC)     #Ajustamos la imagen con un tamaño de 150 x 200
        resultado = reconocimiento.predict(cara_rec)

    #Se mostraran los resultados en pantalla
    if resultado[0] == 0:
        cv2.putText(frame, '{}'.format(dire_img(0)),(x1,y1-5),1,1.3,(0,0,255),1,cv2.LINE_AA)
        cv2.rectangle(frame,(x1,y1),(x1 + ancho, y1 + alto),(0,0,255),2)
    if resultado[0] == 1:
        cv2.putText(frame, '{}'.format(dire_img(1)),(x1,y1-5),1,1.3,(0,255,0),1,cv2.LINE_AA)
        cv2.rectangle(frame,(x1,y1),(x1 + ancho, y1 + alto),(0,255,0),2)
    cv2.imshow('Reconocimiento',frame)
    t = cv2.waitKey(1)
    if t == 27:
        break
cap.release()
cv2.destroyAllWindows()