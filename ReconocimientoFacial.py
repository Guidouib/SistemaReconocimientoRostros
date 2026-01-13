import cv2
import os
import pickle
import numpy as np
from deepface import DeepFace

# Configuración de rutas
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(SCRIPT_DIR, 'face_embeddings.pkl')

# Variables globales para el modelo en memoria
known_embeddings = []
known_names = []
model_loaded = False

def cargar_modelo():
    global known_embeddings, known_names, model_loaded
    if os.path.exists(MODEL_FILE):
        try:
            with open(MODEL_FILE, "rb") as f:
                data = pickle.load(f)
                known_embeddings = np.array(data["embeddings"])
                known_names = data["names"]
                model_loaded = True
                print(f"Modelo cargado: {len(known_embeddings)} rostros.")
        except Exception as e:
            print(f"Error cargando modelo: {e}")
            model_loaded = False
    else:
        print("Advertencia: No se encontró el archivo de modelo 'face_embeddings.pkl'.")
        model_loaded = False

# Cargar al inicio (si existe)
cargar_modelo()

def calcular_distancia_coseno(source_representation, test_representation):
    a = np.matmul(np.transpose(source_representation), test_representation)
    b = np.sum(np.multiply(source_representation, source_representation))
    c = np.sum(np.multiply(test_representation, test_representation))
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))

def ReconocimiendoFacial(frame, x, y, w, h):
    global model_loaded
    
    # Si el modelo no cargó al inicio, intentar cargarlo de nuevo (por si se entrenó recién)
    if not model_loaded:
        cargar_modelo()
        if not model_loaded:
            return "Sin Modelo", (0, 0, 255)

    # Recorte del rostro
    rostro = frame[y:y+h, x:x+w]
    
    try:
        # Obtener embedding del rostro actual
        # enforce_detection=False ya que estamos pasando un recorte
        results = DeepFace.represent(
            img_path=rostro, 
            model_name="ArcFace", 
            enforce_detection=False
        )
        
        if not results:
             return "Desconocido", (0, 0, 255)

        target_embedding = results[0]["embedding"]
        target_embedding = np.array(target_embedding)

        min_dist = float("inf")
        best_match_index = -1

        # Comparar con todos los embeddings guardados
        # Nota: Esto se puede optimizar con operaciones matriciales vectorizadas
        # pero con <1000 rostros es suficientemente rápido iterando o broadcasting simple.
        
        # Versión vectorizada simple
        # distance = 1 - cosine_similarity
        
        # Normalizar para facilitar coseno
        # DeepFace embeddings no siempre estan normalizados, asi que usamos la formula completa
        
        # Calculo matricial de distancias coseno
        # d(A, B) = 1 - (A . B) / (||A|| * ||B||)
        
        # Pre-calculos necesarios si se quiere optimizar, pero hagamoslo simple primero
        for i, source_emb in enumerate(known_embeddings):
            dist = calcular_distancia_coseno(source_emb, target_embedding)
            if dist < min_dist:
                min_dist = dist
                best_match_index = i
        
        # Threshold para ArcFace
        # DeepFace default para ArcFace es 0.68 (Cosine)
        umbral = 0.7
        
        if min_dist < umbral and best_match_index != -1:
            nombre = known_names[best_match_index]
            color = (0, 255, 0)
        else:
            nombre = "Desconocido"
            color = (0, 0, 255)
            
        return nombre, color

    except Exception as e:
        print("Error en reconocimiento:", e)
        return "Error", (0, 0, 255)


