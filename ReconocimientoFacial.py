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
    """
    Realiza reconocimiento facial y clasifica el resultado.
    
    Retorna: (nombre, color, distancia, clasificacion)
      - clasificacion: "TP", "FP", "FN", "TN", o "FILTRADO"
    """
    global model_loaded
    
    # Si el modelo no cargó al inicio, intentar cargarlo de nuevo (por si se entrenó recién)
    if not model_loaded:
        cargar_modelo()
        if not model_loaded:
            return "Sin Modelo", (0, 0, 255), -1.0, "FILTRADO"

    # =============================
    # 1️⃣ Recorte del rostro
    # =============================
    rostro = frame[y:y+h, x:x+w]

    # Validación defensiva
    if rostro.size == 0:
        return "Desconocido", (0, 0, 255), -1.0, "FILTRADO"

    # =============================
    # 2️⃣ FILTRO POR TAMAÑO MÍNIMO
    # =============================
    if w < 80 or h < 80:
        return "Desconocido", (0, 0, 255), -1.0, "FILTRADO"

    # =============================
    # 3️⃣ FILTRO DE NITIDEZ (BLUR)
    # =============================
    gray = cv2.cvtColor(rostro, cv2.COLOR_BGR2GRAY)
    blur = cv2.Laplacian(gray, cv2.CV_64F).var()

    if blur < 50:
        return "Desconocido", (0, 0, 255), -1.0, "FILTRADO"

    # =============================
    # 4️⃣ OBTENER EMBEDDING (ArcFace)
    # =============================
    
    # Umbrales de clasificación (distancia coseno)
    UMBRAL_TP = 0.35       # dist < 0.35 → Reconocimiento seguro (TP)
    UMBRAL_RECONOCER = 0.45 # dist < 0.45 → Se reconoce (pero si 0.35-0.45 es FP potencial)
    UMBRAL_FN = 0.55        # dist < 0.55 → FN potencial (muy cerca del umbral, posible persona conocida)

    try:
        results = DeepFace.represent(
            img_path=rostro, 
            model_name="ArcFace", 
            enforce_detection=False
        )
        
        if not results:
             return "Desconocido", (0, 0, 255), -1.0, "FILTRADO"

        target_embedding = results[0]["embedding"]
        target_embedding = np.array(target_embedding)

        min_dist = float("inf")
        best_match_index = -1

        for i, source_emb in enumerate(known_embeddings):
            dist = calcular_distancia_coseno(source_emb, target_embedding)
            if dist < min_dist:
                min_dist = dist
                best_match_index = i
        
        # =============================
        # 5️⃣ CLASIFICACIÓN AUTOMÁTICA
        # =============================
        candidato = known_names[best_match_index] if best_match_index != -1 else "Nadie"
        
        if min_dist < UMBRAL_TP and best_match_index != -1:
            # ✅ Reconocimiento seguro → Verdadero Positivo
            nombre = known_names[best_match_index]
            color = (0, 255, 0)  # Verde
            clasificacion = "TP"
        elif min_dist < UMBRAL_RECONOCER and best_match_index != -1:
            # ⚠️ Reconocido pero con baja confianza → Falso Positivo potencial
            nombre = known_names[best_match_index]
            color = (0, 165, 255)  # Naranja
            clasificacion = "FP"
        elif min_dist < UMBRAL_FN:
            # ⚠️ No reconocido pero cerca del umbral → Falso Negativo potencial
            nombre = "Desconocido"
            color = (0, 100, 255)  # Rojo-naranja
            clasificacion = "FN"
        else:
            # ❌ Definitivamente desconocido → Verdadero Negativo
            nombre = "Desconocido"
            color = (0, 0, 255)  # Rojo
            clasificacion = "TN"
        
        print(f"--> [DEBUG] Distancia: {min_dist:.4f} | Candidato: {candidato} | Resultado: {nombre} | Clasificación: {clasificacion}")
            
        return nombre, color, min_dist, clasificacion

    except Exception as e:
        print("Error en reconocimiento:", e)
        return "Error", (0, 0, 255), -1.0, "FILTRADO"


