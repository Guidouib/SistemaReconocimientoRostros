import cv2 
import os
import numpy as np
import pickle
from deepface import DeepFace

# Definir rutas relativas para portabilidad
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, 'Data')
MODEL_FILE = os.path.join(SCRIPT_DIR, 'face_embeddings.pkl')

def entrenar_reconocedor_facil():
    print(f"Buscando datos en: {DATA_PATH}")
    if not os.path.exists(DATA_PATH):
        print("Error: No existe la carpeta Data")
        return

    peopleList = os.listdir(DATA_PATH)
    print('Lista de personas:', peopleList)
    
    known_embeddings = []
    known_names = []

    for nameDir in peopleList:
        personPath = os.path.join(DATA_PATH, nameDir)
        print(f'Procesando: {nameDir}')

        if not os.path.isdir(personPath):
            continue

        for fileName in os.listdir(personPath):
            if not fileName.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
                
            imagePath = os.path.join(personPath, fileName)
            try:
                # Generar embedding con ArcFace
                # enforce_detection=False porque ya son recortes de rostros
                embedding_objs = DeepFace.represent(
                    img_path=imagePath, 
                    model_name="ArcFace", 
                    enforce_detection=False
                )
                
                # DeepFace.represent devuelve una lista, tomamos el primer (y unico) rostro
                if embedding_objs:
                    embedding = embedding_objs[0]["embedding"]
                    known_embeddings.append(embedding)
                    known_names.append(nameDir)
                    
            except Exception as e:
                print(f"Error procesando {fileName}: {e}")

    # Guardar los embeddings y nombres en un archivo pickle
    data = {
        "embeddings": known_embeddings,
        "names": known_names
    }
    
    try:
        with open(MODEL_FILE, "wb") as f:
            pickle.dump(data, f)
        print(f"Entrenamiento finalizado. Modelo guardado en {MODEL_FILE}")
        print(f"Total rostros procesados: {len(known_embeddings)}")
    except Exception as e:
        print(f"Error guardando el modelo: {e}")
