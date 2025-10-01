import cv2 
import os
import numpy as np

def entrenar_reconocedor_facil():
    dataPath = 'C:/Users/GuidoUib/Desktop/TESIS OFICIAL/APP-PYTHON/REC_FACIAL_UIB/PRIMCIPAL/Data'
    peopleList = os.listdir(dataPath)
    print('Lista de personas', peopleList)
    labels= []
    facesDate = []
    label = 0
    for nameDir in peopleList:
        personPath = dataPath + '/' + nameDir
        print('Leyendo las imagenes')

        for fileName in os.listdir(personPath):
            print('Rostros: ', nameDir +'/'+ fileName)
            labels.append(label)
            facesDate.append(cv2.imread(personPath + '/' + fileName,0))
            image = cv2.imread(personPath + '/' + fileName,0)
        label = label + 1
    face_recognizer = cv2.face.EigenFaceRecognizer_create()
    print("Entrenando...")
    face_recognizer.train(facesDate,np.array(labels))
    face_recognizer.write('modeloEigenFace1.xml')
    print("Modelo almacenado...")