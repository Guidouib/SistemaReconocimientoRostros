import cv2
import os

dataPath = 'C:/Users/GuidoUib/Desktop/TESIS OFICIAL/APP-PYTHON/REC_FACIAL_UIB/PRIMCIPAL/Data'
imagePaths = os.listdir(dataPath)

#face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer = cv2.face.EigenFaceRecognizer_create()
#face_recognizer.read('modeloLBPHFace.xml')
face_recognizer.read('modeloEigenFace1.xml')
def ReconocimiendoFacial(frame,x, y, w, h):

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    auxFrame = gray.copy()
    rostro = auxFrame[y:y+h,x:x+w]
    try:
        rostro = cv2.resize(rostro,(150,150),interpolation=cv2.INTER_CUBIC)
        result = face_recognizer.predict(rostro)
        if result[1] < 5700:
            nombre = imagePaths[result[0]]
            color = (0,255,0)
        else:
            nombre = "Desconocido"
            color = (0, 0, 255)
        return nombre, color
    except Exception as e:
        print("Error en reconocimiento: ", e)
        return "Error", (0,0,255)

