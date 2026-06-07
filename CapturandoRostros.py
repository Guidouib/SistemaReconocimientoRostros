from tkinter import *
from tkinter import filedialog
from PIL import Image
from PIL import ImageTk
from ultralytics import YOLO 
import cv2
import os
import entrenandoRF
import ReconocimientoFacial
import Conexion
import datetime
import pyodbc
import RegistroTrabajador
from tkinter import ttk
import torch
import torchvision
from torchvision import transforms as T
import threading
import queue
import metricas

frame_count = 0
limite_imagenes = 300
modo_reconocerFacial = False
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "yolo11n-face.pt")

# Modelos
yolo_model = YOLO(model_path)
ssd_model = None
faster_rcnn_model = None

conn = Conexion.get_db_connection()

detecciones_totales = 0
video_actual_path = ""
tiempo_inicio_simulacion = None
after_id = None
frames_procesados = 0  # Contador de frames para cálculo de tiempo/frame

# Contadores automáticos de clasificación (modo con reconocimiento)
conteo_tp = 0
conteo_fp = 0
conteo_fn = 0
conteo_tn = 0

# === MODO SIN RECONOCIMIENTO (DETECCIÓN PURA) ===
# Solo mide velocidad y volumen de detección, sin identificar a la persona.
# Responde a: ¿qué tan rápido y cuánto detecta el modelo sin importar quién es?
modo_deteccion_pura = False

# Cache para reconocimiento facial
face_data_cache = []
import glob

recognition_queue = queue.Queue()
cola_videos = []
modo_batch_seleccionado = "reconocimiento"  # Por defecto: "reconocimiento" o "deteccion_pura"

# Optimización para modelos pesados (SSD, Faster R-CNN)
ssd_transform = T.Compose([T.ToTensor()])
_heavy_skip_interval = 3  # Solo ejecutar inferencia cada N frames
_heavy_frame_counter = 0
_cached_heavy_detections = []  # Cache de detecciones (x, y, w, h)

def worker_reconocimiento():
    """Hilo en segundo plano para procesar reconocimiento facial sin bloquear UI"""
    global conteo_tp, conteo_fp, conteo_fn, conteo_tn
    while True:
        try:
            # Obtener tarea (bloqueante, pero en hilo aparte)
            task = recognition_queue.get()
            if task is None: break # Señal de parada
            
            rostro_img, face_data = task
            
            # Ejecutar reconocimiento (retorna 4 valores)
            h, w, _ = rostro_img.shape
            nombre, color, distancia, clasificacion = ReconocimientoFacial.ReconocimiendoFacial(rostro_img, 0, 0, w, h)
            
            # Actualizar diccionario compartido
            face_data['name'] = nombre
            face_data['color'] = color
            face_data['distancia'] = distancia
            face_data['clasificacion'] = clasificacion
            face_data['skip'] = 10
            face_data['pending'] = False
            
            recognition_queue.task_done()
        except Exception as e:
            print(f"Error en hilo de reconocimiento: {e}")

# Iniciar hilo demonio (se cierra al cerrar la app)
t = threading.Thread(target=worker_reconocimiento, daemon=True)
t.start()

def guardar_frame(frame, x, y, w, h):
    global frame_count, limite_imagenes

    if frame_count < limite_imagenes:
        rostro = frame[y:y+h, x:x+w]
        rostro = cv2.resize(rostro,(150,150),interpolation=cv2.INTER_CUBIC)
        frame_count += 1
        filename = os.path.join(personPath, f"frame_{frame_count:05d}.jpg")
        cv2.imwrite(filename, rostro)
        lblContador.config(text=f"Imágenes capturadas: {frame_count}/{limite_imagenes}")

def procesar_deteccion(frame, x, y, w, h):
    global detecciones_totales
    global conteo_tp, conteo_fp, conteo_fn, conteo_tn
    
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    detecciones_totales += 1
    
    # === MODO DETECCIÓN PURA (sin reconocimiento) ===
    # Solo marca la detección visualmente y cuenta. No ejecuta ArcFace.
    if modo_deteccion_pura:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 165, 0), 2)
        cv2.putText(frame, "Detectado", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, (255, 165, 0), 2, cv2.LINE_AA)
        return frame
    
    if modo_reconocerFacial:
        # Lógica de Caché para optimización
        best_match_index = -1
        center_x = x + w / 2
        center_y = y + h / 2
        
        # Buscar similitud con rostros cacheados (por distancia)
        for i, face in enumerate(face_data_cache):
            fx, fy, fw, fh = face['x'], face['y'], face['w'], face['h']
            fc_x = fx + fw / 2
            fc_y = fy + fh / 2
            
            # Distancia euclidiana entre centros
            dist = ((center_x - fc_x)**2 + (center_y - fc_y)**2)**0.5
            
            # Umbral de movimiento (píxeles). Si está cerca, es la misma persona.
            if dist < 50: 
                best_match_index = i
                break
        
        # Variables para mostrar (mientras se procesa o si ya existe)
        nombre_mostrar = "Procesando..."
        color_mostrar = (128, 128, 128) # Gris
        
        if best_match_index != -1:
            item = face_data_cache[best_match_index]
            
            if item['skip'] <= 0 and not item.get('pending', False):
                 item['pending'] = True
                 rostro_recorte = frame[y:y+h, x:x+w].copy()
                 recognition_queue.put((rostro_recorte, item))
            else:
                 if not item.get('pending', False):
                     item['skip'] -= 1

            item['x'], item['y'], item['w'], item['h'] = x, y, w, h
            item['seen'] = True
            
            nombre_mostrar = item['name']
            color_mostrar = item['color']
            
        else:
            new_item = {
                'x': x, 'y': y, 'w': w, 'h': h,
                'name': "Procesando...",
                'color': (128, 128, 128),
                'skip': 0,
                'seen': True,
                'pending': True
            }
            face_data_cache.append(new_item)
            
            # Enviar a reconocer inmediatamente
            rostro_recorte = frame[y:y+h, x:x+w].copy()
            recognition_queue.put((rostro_recorte, new_item))
            
            nombre_mostrar = new_item['name']
            color_mostrar = new_item['color']

        cv2.putText(frame, nombre_mostrar, (x, y - 25), 2, 1.1, color_mostrar, 1, cv2.LINE_AA)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color_mostrar, 2)
        
        if best_match_index != -1:
            cls = face_data_cache[best_match_index].get('clasificacion')
        else:
            cls = new_item.get('clasificacion')
            
        if cls == "TP": conteo_tp += 1
        elif cls == "FP": conteo_fp += 1
        elif cls == "FN": conteo_fn += 1
        elif cls == "TN": conteo_tn += 1
    else:
        pass 
        if 'guardar' in globals() and guardar: 
             guardar_frame(frame.copy(), x, y, w, h)

    return frame

def detectar_rostro_yolo(frame):
    results = yolo_model.predict(frame, stream=False, verbose=False, conf=0.6)[0]
    boxes = results.boxes.xyxy.cpu().numpy()
    
    for (x1, y1, x2, y2) in boxes:
        x, y = int(x1), int(y1)
        w, h = int(x2-x1), int(y2 - y1)
        procesar_deteccion(frame, x, y, w, h)
    return frame

cargando_modelo_estado = False
modelo_cargando_nombre = ""

def cargar_modelo_en_background(nombre_modelo):
    global ssd_model, faster_rcnn_model, cargando_modelo_estado
    try:
        if nombre_modelo == "SSD":
            print("Descargando/Cargando modelo SSD (PyTorch) en segundo plano...")
            ssd_model = torchvision.models.detection.ssd300_vgg16(weights=torchvision.models.detection.SSD300_VGG16_Weights.DEFAULT)
            ssd_model.eval()
            if torch.cuda.is_available():
                ssd_model.to('cuda')
            print("SSD Listo!")
            
        elif nombre_modelo == "FASTER R-CNN":
            print("Descargando/Cargando modelo Faster R-CNN (PyTorch) en segundo plano...")
            faster_rcnn_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
            faster_rcnn_model.eval()
            if torch.cuda.is_available():
                faster_rcnn_model.to('cuda')
            print("Faster R-CNN Listo!")
    except Exception as e:
        print(f"Error cargando el modelo {nombre_modelo}: {e}")
    finally:
        cargando_modelo_estado = False

def detectar_rostro_ssd(frame):
    global ssd_model, _heavy_frame_counter, _cached_heavy_detections
    if ssd_model is None:
        return frame
    
    _heavy_frame_counter += 1
    
    if _heavy_frame_counter % _heavy_skip_interval != 0 and len(_cached_heavy_detections) > 0:
        for (cx, cy, cw, ch) in _cached_heavy_detections:
            procesar_deteccion(frame, cx, cy, cw, ch)
        return frame
    
    img_tensor = ssd_transform(frame)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    img_tensor = img_tensor.to(device)
    
    with torch.no_grad():
        if device == 'cuda':
            with torch.amp.autocast('cuda'):
                detections = ssd_model([img_tensor])[0]
        else:
            detections = ssd_model([img_tensor])[0]
    
    new_detections = []
    confidence_threshold = 0.5
    for i in range(len(detections['boxes'])):
        score = detections['scores'][i].item()
        label = detections['labels'][i].item()
        
        if score > confidence_threshold and label == 1:  # 1 es Persona
            box = detections['boxes'][i].cpu().numpy()
            x1, y1, x2, y2 = box.astype(int)
            x, y = x1, y1
            w, h = x2 - x1, y2 - y1
            new_detections.append((x, y, w, h))
            procesar_deteccion(frame, x, y, w, h)
    
    _cached_heavy_detections = new_detections
    return frame

def detectar_rostro_faster_rcnn(frame):
    global faster_rcnn_model, _heavy_frame_counter, _cached_heavy_detections
    if faster_rcnn_model is None:
        return frame

    _heavy_frame_counter += 1
    
    if _heavy_frame_counter % _heavy_skip_interval != 0 and len(_cached_heavy_detections) > 0:
        for (cx, cy, cw, ch) in _cached_heavy_detections:
            procesar_deteccion(frame, cx, cy, cw, ch)
        return frame

    img_tensor = ssd_transform(frame)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    img_tensor = img_tensor.to(device)
    
    with torch.no_grad():
        if device == 'cuda':
            with torch.amp.autocast('cuda'):
                detections = faster_rcnn_model([img_tensor])[0]
        else:
            detections = faster_rcnn_model([img_tensor])[0]
    
    new_detections = []
    confidence_threshold = 0.5
    for i in range(len(detections['boxes'])):
        score = detections['scores'][i].item()
        label = detections['labels'][i].item()
        
        if score > confidence_threshold and label == 1:  # 1 es Persona
            box = detections['boxes'][i].cpu().numpy()
            x1, y1, x2, y2 = box.astype(int)
            x, y = x1, y1
            w, h = x2 - x1, y2 - y1
            new_detections.append((x, y, w, h))
            procesar_deteccion(frame, x, y, w, h)
    
    _cached_heavy_detections = new_detections
    return frame

def deteccion_facial(frame):
    global detecciones_totales, MODELO_ACTIVO, face_data_cache, cargando_modelo_estado, modelo_cargando_nombre
    
    if cargando_modelo_estado:
        h, w = frame.shape[:2]
        texto = f"Cargando {modelo_cargando_nombre}... Espere por favor."
        cv2.putText(frame, texto, (20, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        return frame

    for face in face_data_cache:
        face['seen'] = False

    result_frame = frame
    if MODELO_ACTIVO == "YOLO":
        result_frame = detecting_rostro_yolo_wrapper(frame)
    elif MODELO_ACTIVO == "SSD":
        result_frame = detectar_rostro_ssd(frame)
    elif MODELO_ACTIVO == "FASTER R-CNN":
        result_frame = detectar_rostro_faster_rcnn(frame)
    else:
        result_frame = detecting_rostro_yolo_wrapper(frame)
        
    face_data_cache[:] = [f for f in face_data_cache if f['seen']]
    
    return result_frame

def detecting_rostro_yolo_wrapper(frame):
    return detectar_rostro_yolo(frame)

def visualizar():
    global cap, personPath, tiempo_inicio_simulacion, detecciones_totales, after_id, guardar, frames_procesados
    ret, frame = cap.read()

    if not cap or not cap.isOpened():
        return

    if ret == True:
        frames_procesados += 1
        if tiempo_inicio_simulacion is None:
            tiempo_inicio_simulacion = datetime.datetime.now()
        
        label_width = lblVideo.winfo_width()
        label_height = lblVideo.winfo_height()

        if label_width < 50 or label_height < 50:
            after_id = lblVideo.after(10, visualizar)
            return

        h, w = frame.shape[:2]
        scale = min(label_width / w, label_height / h)

        new_w = int(w * scale)
        new_h = int(h * scale)

        frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Logica de preparacion de directorios
        if guardar == True and btnGuardar['state'] == DISABLED or modo_reconocerFacial or modo_deteccion_pura:
             # Solo crear directorios si vamos a guardar realmente
             if guardar:
                personName = textNombre.get()
                output_dir = 'C:/Users/ASUS TUF/Desktop/Tesis Of/SistemaReconocimientoRostros/Data'
                personPath = output_dir + '/' + personName
                if not os.path.exists(personPath):
                    os.makedirs(personPath)

             frame = deteccion_facial(frame) 
             lblDetecciones.config(text=f"Detecciones: {detecciones_totales}")
             if modo_deteccion_pura:
                 lblClasificacion.config(text=f"Modo: DETECCIÓN PURA | Frames: {frames_procesados}")
             else:
                 lblClasificacion.config(text=f"TP: {conteo_tp} | FP: {conteo_fp} | FN: {conteo_fn} | TN: {conteo_tn}")

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(frame)
        img = ImageTk.PhotoImage(image = im)

        lblVideo.configure(image = img)
        lblVideo.image = img
        after_id = lblVideo.after(10, visualizar)
    else:
        if modo_reconocerFacial:
           # Guardar resultados del video actual en la BD
           finalizar_guardar_resultado()
           
           # Revisar si hay un siguiente video en la cola
           global cola_videos
           if len(cola_videos) > 0:
               # Pequeña pausa para no encolar eventos de Tkinter de forma conflictiva
               lblVideo.after(500, iniciar_siguiente_video_de_cola)
           else:
               # No hay más videos en la cola, limpiar UI completamente
               finalizar_limpiar()
        elif modo_deteccion_pura:
           # Guardar resultados del modo SIN reconocimiento en la BD
           finalizar_guardar_resultado_puro()
           
           # Revisar si hay un siguiente video en la cola
           if len(cola_videos) > 0:
               lblVideo.after(500, iniciar_siguiente_video_de_cola)
           else:
               finalizar_limpiar()
        else:
           finalizar_limpiar()
           
def activar_reconocimiento():
    global modo_reconocerFacial, modo_deteccion_pura
    modo_reconocerFacial = True
    modo_deteccion_pura = False  # Mutuamente excluyentes
    btnReconocerFacial.configure(state="disabled", text="✓ Reconocimiento Activo")
    btnDeteccionPura.configure(state="disabled")
    lblEstado.config(text="Estado: Reconociendo rostros", fg="green")

def activar_deteccion_pura():
    """Activa el modo de detección pura (sin reconocimiento facial).
    Solo cuenta detecciones, mide tiempo y FPS. No identifica personas."""
    global modo_deteccion_pura, modo_reconocerFacial
    modo_deteccion_pura = True
    modo_reconocerFacial = False  # Mutuamente excluyentes
    btnDeteccionPura.configure(state="disabled", text="✓ Detección Pura Activa")
    btnReconocerFacial.configure(state="disabled")
    lblEstado.config(text="Estado: Detección pura (sin reconocimiento)", fg="orange")

def video_de_entrada(opcion):
    global cap, video_actual_path, detecciones_totales, tiempo_inicio_simulacion, frame_count
    global conteo_tp, conteo_fp, conteo_fn, conteo_tn, frames_procesados, limite_imagenes
    
    if opcion == 1:
        path_video = filedialog.askopenfilename(
            filetypes=[("Video files", "*.mp4 *.avi")])
        
        if len(path_video) > 0:
            video_actual_path = path_video
            lblInfoVideoPath.configure(text=os.path.basename(path_video))
            cap = cv2.VideoCapture(path_video)
            
            # Adaptar límite de imágenes según la duración del video
            total_frames_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames_video > 0:
                limite_imagenes = total_frames_video
            else:
                limite_imagenes = 300  # Fallback si no se puede leer
            print(f"Límite de captura adaptado al video: {limite_imagenes} frames")
        else:
            return  # Usuario canceló, no hacer nada
    elif opcion == 2:
        video_actual_path = "Camara" 
        lblInfoVideoPath.configure(text="Cámara en Directo")
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        limite_imagenes = 1000  # Cámara: límite fijo de 1000 rostros
        print(f"Límite de captura para cámara: {limite_imagenes} frames")

    btnVIdeo.configure(state="disabled")
    btnCarpeta.configure(state="disabled")
    btnCamara.configure(state="disabled")
    btnEnd.configure(state="normal")
    btnGuardar.configure(state="normal")
    textNombre.config(state="normal")
    btnEntrenar.configure(state="normal")
    btnReconocerFacial.configure(state="normal")
    btnDeteccionPura.configure(state="normal")
    
    # Resetear TODOS los contadores para un inicio limpio
    detecciones_totales = 0
    frame_count = 0
    frames_procesados = 0
    conteo_tp = 0
    conteo_fp = 0
    conteo_fn = 0
    conteo_tn = 0
    tiempo_inicio_simulacion = None
    face_data_cache.clear()
    _cached_heavy_detections.clear()
    
    lblEstado.config(text="Estado: Procesando...", fg="blue")
    lblDetecciones.config(text="Detecciones: 0")
    lblClasificacion.config(text="TP: 0 | FP: 0 | FN: 0 | TN: 0")
    lblContador.config(text=f"Imágenes capturadas: 0/{limite_imagenes}")
    visualizar()

def cargar_carpeta_videos():
    global cola_videos, cap, modo_reconocerFacial, modo_deteccion_pura
    from tkinter import messagebox
    
    # Preguntar al usuario qué modo usar para el lote
    respuesta = messagebox.askyesnocancel(
        "Modo de procesamiento por lotes",
        "¿En qué modo desea procesar los videos?\n\n"
        "• SÍ → Con Reconocimiento Facial (TP/FP/FN/TN, ArcFace)\n"
        "• NO → Detección Pura (solo velocidad y volumen, sin identidad)\n"
        "• CANCELAR → Volver"
    )
    
    if respuesta is None:  # Canceló
        return
    
    # Configurar el modo según la respuesta
    modo_batch_reconocer = respuesta  # True = reconocimiento, False = detección pura
    
    # El usuario selecciona directamente un video de la carpeta
    archivo_inicio = filedialog.askopenfilename(
        title="Seleccionar video desde donde iniciar el lote",
        filetypes=[("Video files", "*.mp4 *.avi")]
    )
    if not archivo_inicio: return
    
    # Obtener la carpeta del video seleccionado
    carpeta = os.path.dirname(archivo_inicio)
    
    # Buscar todos los videos en esa carpeta
    videos = glob.glob(os.path.join(carpeta, "*.mp4")) + glob.glob(os.path.join(carpeta, "*.avi"))
    if len(videos) == 0:
        print("No se encontraron videos en la carpeta seleccionada")
        return
    
    # Normalizar rutas para comparar correctamente
    videos = [os.path.normpath(v) for v in videos]
    archivo_inicio = os.path.normpath(archivo_inicio)
    
    # Ordenar videos alfabéticamente para asegurar orden correcto
    videos.sort()
    
    # Buscar el índice del video seleccionado
    indice_inicio = -1
    for i, v in enumerate(videos):
        if v == archivo_inicio:
            indice_inicio = i
            break
    
    if indice_inicio == -1:
        # Fallback: procesar todos
        messagebox.showwarning(
            "Video no encontrado en lista",
            f"No se pudo ubicar el video seleccionado en la lista.\n"
            f"Se procesarán todos los {len(videos)} videos."
        )
        videos_a_procesar = videos
    else:
        videos_a_procesar = videos[indice_inicio:]
    
    cola_videos.extend(videos_a_procesar)
    print(f"Se agregaron {len(videos_a_procesar)} videos a la cola (iniciando desde: {os.path.basename(videos_a_procesar[0])}).")
    
    # Guardar el modo elegido para usarlo en cada video de la cola
    global modo_batch_seleccionado
    modo_batch_seleccionado = "reconocimiento" if modo_batch_reconocer else "deteccion_pura"
    print(f"Modo seleccionado para el lote: {modo_batch_seleccionado.upper()}")
    
    if cap is None or not cap.isOpened():
        iniciar_siguiente_video_de_cola()

def iniciar_siguiente_video_de_cola():
    global cola_videos, video_actual_path, cap, detecciones_totales, tiempo_inicio_simulacion, frame_count
    global conteo_tp, conteo_fp, conteo_fn, conteo_tn, frames_procesados
    
    if len(cola_videos) == 0:
        print("Todos los videos de la cola han sido procesados.")
        lblInfoVideoPath.configure(text="Procesamiento por lotes finalizado")
        finalizar_limpiar()
        return
        
    video_path = cola_videos.pop(0)
    video_actual_path = video_path
    
    # Preparar el UI para el siguiente video
    lblInfoVideoPath.configure(text=f"Lote [{MODELO_ACTIVO}]: {os.path.basename(video_path)} ({len(cola_videos)} restantes)")
    cap = cv2.VideoCapture(video_path)
    
    btnVIdeo.configure(state="disabled")
    btnCarpeta.configure(state="disabled")
    btnCamara.configure(state="disabled")
    btnEnd.configure(state="normal")
    btnGuardar.configure(state="disabled")
    textNombre.config(state="disabled")
    btnEntrenar.configure(state="disabled")
    
    # Resetear TODOS los contadores para el siguiente video
    detecciones_totales = 0
    frame_count = 0
    frames_procesados = 0
    conteo_tp = 0
    conteo_fp = 0
    conteo_fn = 0
    conteo_tn = 0
    tiempo_inicio_simulacion = None
    face_data_cache.clear()
    _cached_heavy_detections.clear()
    
    # Actualizar labels de la UI
    lblDetecciones.config(text="Detecciones: 0")
    lblClasificacion.config(text="TP: 0 | FP: 0 | FN: 0 | TN: 0")
    
    # Activar el modo correspondiente según lo elegido en el lote
    if modo_batch_seleccionado == "deteccion_pura":
        activar_deteccion_pura()
    else:
        activar_reconocimiento()
    
    # Iniciar la visualización
    visualizar()

def finalizar_limpiar():
    global cap, after_id, modo_reconocerFacial, guardar, modo_deteccion_pura
    global conteo_tp, conteo_fp, conteo_fn, conteo_tn, cola_videos
    
    if after_id is not None:
        lblVideo.after_cancel(after_id)
        after_id = None

    if cap and cap.isOpened():
        cap.release()
        cap = None
    
    modo_reconocerFacial = False
    modo_deteccion_pura = False
    guardar = False
    
    # Resetear contadores de clasificación
    conteo_tp = 0
    conteo_fp = 0
    conteo_fn = 0
    conteo_tn = 0
    
    # Limpiar caché y cola
    face_data_cache.clear()
    _cached_heavy_detections.clear()
    cola_videos.clear()
    
    # Vaciar la cola de reconocimiento pendiente (para que no procese rostros del video anterior)
    while not recognition_queue.empty():
        try:
            recognition_queue.get_nowait()
        except queue.Empty:
            break
    
    lblVideo.image = ""
    lblInfoVideoPath.configure(text="Ningún video seleccionado")
    btnVIdeo.configure(state="normal")
    btnCarpeta.configure(state="normal")
    btnCamara.configure(state="normal")
    btnGuardar.configure(state="disabled")
    textNombre.config(state="normal")
    textNombre.delete(0, END)
    textNombre.config(state="disabled")
    btnEntrenar.configure(state="normal")
    btnEnd.configure(state="disabled")
    lblEstado.config(text="Estado: Inactivo", fg="gray")
    lblDetecciones.config(text="Detecciones: 0")
    lblContador.config(text="Imágenes capturadas: 0/300")
    lblClasificacion.config(text="TP: 0 | FP: 0 | FN: 0 | TN: 0")
    btnReconocerFacial.configure(state="disabled", text="🔍 Reconocer Persona")
    btnDeteccionPura.configure(state="disabled", text="📊 Detección Pura")

def guardar_nombre(textNombre):
    global personName, guardar, frame_count
    personName = textNombre.get().strip()
    
    if personName == "" or not personName.isdigit() or len(personName) != 8:
        lblEstado.config(text="Estado: ⚠ Ingrese DNI valido de 8 digitos.", fg="red")
        return
    
    guardar = True
    frame_count = 0
    btnGuardar.configure(state="disabled")
    textNombre.config(state="disabled")
    textNombre.delete(0, END)
    btnEntrenar.configure(state="disabled")
    btnReconocerFacial.configure(state="disabled")
    lblEstado.config(text=f"Estado: Guardando rostros para '{personName}'", fg="green")

def finalizar_guardar_resultado():
    global cap, detecciones_totales, tiempo_inicio_simulacion, video_actual_path
    global conteo_tp, conteo_fp, conteo_fn, conteo_tn, face_data_cache, frames_procesados

    tiempo_fin_simulacion = datetime.datetime.now()
    tiempo_total_ms = round((tiempo_fin_simulacion - tiempo_inicio_simulacion).total_seconds() * 1000, 3) if tiempo_inicio_simulacion else 0

    # Generar reporte completo usando el módulo de métricas
    video_nombre = os.path.basename(video_actual_path) if video_actual_path else "Camara en Directo"
    reporte = metricas.generar_reporte(
        tp=conteo_tp, fp=conteo_fp, fn=conteo_fn, tn=conteo_tn,
        tiempo_total_ms=tiempo_total_ms, total_frames=frames_procesados,
        modelo=MODELO_ACTIVO, video=video_nombre
    )
    
    # Imprimir reporte completo en consola
    print(reporte["reporte_texto"])

    fecha_simulacion_str = tiempo_fin_simulacion.strftime('%Y-%m-%d %H:%M:%S')
    insertar_Resultado_Deteccion(
        algoritmo=MODELO_ACTIVO, 
        video_prueba=video_nombre,
        detecciones_correctas=reporte["tp"],
        total_rostros_video=reporte["total_evaluados"],
        tiempo_respuesta_ms=tiempo_total_ms,
        falsos_positivos=reporte["fp"],
        falsos_negativos=reporte["fn"],
        configuracion="Video grabado con cámara de video vigilancia.",
        fecha_simulacion=fecha_simulacion_str,
        recall=reporte["recall"],
        f1_score=reporte["f1_score"],
        especificidad=reporte["especificidad"],
        tiempo_promedio_frame=reporte["tiempo_promedio_frame"]
    )
    
    # Resetear contadores para el siguiente video (sin limpiar UI ni modo_reconocerFacial)
    conteo_tp = 0
    conteo_fp = 0
    conteo_fn = 0
    conteo_tn = 0
    detecciones_totales = 0
    frames_procesados = 0
    tiempo_inicio_simulacion = None
    face_data_cache.clear()
    _cached_heavy_detections.clear()
    
    # Liberar el video actual
    if cap and cap.isOpened():
        cap.release()

def finalizar_guardar_resultado_puro():
    """
    Calcula y guarda en BD las métricas del modo DETECCIÓN PURA (sin reconocimiento).
    Solo registra: total detecciones, tiempo, FPS y tiempo/frame.
    Tabla: SI_FinRendimientoDeteccionPura
    SP:    DET_InsertDataDeteccionPura_SP
    """
    global cap, detecciones_totales, tiempo_inicio_simulacion, video_actual_path
    global frames_procesados

    tiempo_fin_simulacion = datetime.datetime.now()
    tiempo_total_ms = round((tiempo_fin_simulacion - tiempo_inicio_simulacion).total_seconds() * 1000, 3) if tiempo_inicio_simulacion else 0
    tiempo_total_segundos = tiempo_total_ms / 1000.0 if tiempo_total_ms > 0 else 0

    # Calcular métricas
    fps_promedio = round(frames_procesados / tiempo_total_segundos, 2) if tiempo_total_segundos > 0 else 0
    tiempo_promedio_frame = round(tiempo_total_ms / frames_procesados, 3) if frames_procesados > 0 else 0

    video_nombre = os.path.basename(video_actual_path) if video_actual_path else "Camara en Directo"
    
    # Imprimir reporte en consola
    print("")
    print("=" * 55)
    print("  RESUMEN DE DETECCIÓN PURA (SIN RECONOCIMIENTO)")
    print("=" * 55)
    print(f"  Video:                {video_nombre}")
    print(f"  Modelo:               {MODELO_ACTIVO}")
    print("-" * 55)
    print(f"  Total detecciones:    {detecciones_totales}")
    print(f"  Frames procesados:    {frames_procesados}")
    print(f"  Tiempo total:         {tiempo_total_ms:.2f} ms")
    print(f"  FPS promedio:         {fps_promedio:.2f}")
    print(f"  Tiempo/Frame:         {tiempo_promedio_frame:.2f} ms")
    print("=" * 55)
    print("")

    fecha_simulacion_str = tiempo_fin_simulacion.strftime('%Y-%m-%d %H:%M:%S')
    insertar_Resultado_Deteccion_Pura(
        algoritmo=MODELO_ACTIVO,
        video_prueba=video_nombre,
        total_detecciones=detecciones_totales,
        total_frames_procesados=frames_procesados,
        tiempo_respuesta_ms=tiempo_total_ms,
        fps_promedio=fps_promedio,
        tiempo_promedio_frame_ms=tiempo_promedio_frame,
        fecha_simulacion=fecha_simulacion_str,
        configuracion="Video grabado con cámara de video vigilancia."
    )

    # Resetear contadores para el siguiente video
    detecciones_totales = 0
    frames_procesados = 0
    tiempo_inicio_simulacion = None
    face_data_cache.clear()
    _cached_heavy_detections.clear()

    if cap and cap.isOpened():
        cap.release()


def insertar_Resultado_Deteccion_Pura(algoritmo, video_prueba, total_detecciones,
                                       total_frames_procesados, tiempo_respuesta_ms,
                                       fps_promedio, tiempo_promedio_frame_ms,
                                       fecha_simulacion, configuracion):
    """
    Inserta un nuevo registro en la tabla SI_FinRendimientoDeteccionPura
    llamando al stored procedure DET_InsertDataDeteccionPura_SP.
    """
    global conn

    # Asegurar que la conexión esté activa (reconectar si es necesario)
    if not _asegurar_conexion():
        print(f"ERROR CRÍTICO: No se pudo guardar el resultado puro del video '{video_prueba}'. Sin conexión a BD.")
        return

    cursor = conn.cursor()

    # Llamada al stored procedure DET_InsertDataDeteccionPura_SP con 9 parámetros
    sp_call = "EXEC DET_InsertDataDeteccionPura_SP ?, ?, ?, ?, ?, ?, ?, ?, ?"

    params = (
        algoritmo,
        video_prueba,
        total_detecciones,
        total_frames_procesados,
        tiempo_respuesta_ms,
        fps_promedio,
        tiempo_promedio_frame_ms,
        fecha_simulacion,
        configuracion
    )

    try:
        cursor.execute(sp_call, params)
        conn.commit()
        print(f"[DETECCIÓN PURA] Datos guardados en BD. (Modelo: {algoritmo}, Video: {video_prueba})")
    except pyodbc.Error as ex:
        print(f"Error al insertar datos de detección pura: {ex}")
        # Intentar reconectar y reintentar UNA vez
        print("Reintentando con nueva conexión...")
        try:
            conn = Conexion.get_db_connection()
            if conn:
                cursor2 = conn.cursor()
                cursor2.execute(sp_call, params)
                conn.commit()
                cursor2.close()
                print(f"Reintento exitoso. Datos guardados. (Modelo: {algoritmo}, Video: {video_prueba})")
            else:
                print(f"FALLO TOTAL: No se pudieron guardar los datos puros del video '{video_prueba}'.")
        except Exception as ex2:
            print(f"FALLO en reintento: {ex2}")
    finally:
        cursor.close()


def _asegurar_conexion():
    """Verifica que la conexión a la BD esté activa. Si no, reconecta."""
    global conn
    try:
        if conn is not None:
            # Prueba rápida para verificar si la conexión sigue viva
            conn.cursor().execute("SELECT 1").close()
            return True
    except Exception:
        print("Conexión a BD perdida. Reconectando...")
        conn = None
    
    # Intentar reconectar
    try:
        conn = Conexion.get_db_connection()
        if conn is not None:
            print("Reconexión exitosa a la base de datos.")
            return True
    except Exception as e:
        print(f"Error al reconectar: {e}")
    
    return False

def insertar_Resultado_Deteccion(algoritmo, video_prueba, detecciones_correctas, total_rostros_video, tiempo_respuesta_ms, falsos_positivos, falsos_negativos, configuracion, fecha_simulacion, recall=0, f1_score=0, especificidad=0, tiempo_promedio_frame=0):
    """
    Inserta un nuevo registro en la tabla SI_FinRendimientoModelos llamando a un procedimiento almacenado.
    Incluye métricas completas: Precisión, Exactitud, Recall, F1-Score, Especificidad y Tiempo/Frame.
    """
    global conn
    
    # Asegurar que la conexión esté activa (reconectar si es necesario)
    if not _asegurar_conexion():
        print(f"ERROR CRÍTICO: No se pudo guardar el resultado del video '{video_prueba}'. Sin conexión a BD.")
        return

    cursor = conn.cursor()
    
    # Calcular métricas usando el módulo centralizado
    precision = metricas.calcular_precision(detecciones_correctas, falsos_positivos)
    
    # Calcular Exactitud usando el módulo centralizado
    verdaderos_negativos = total_rostros_video - (detecciones_correctas + falsos_positivos + falsos_negativos)
    if verdaderos_negativos < 0: verdaderos_negativos = 0 # Protección
    exactitud = metricas.calcular_exactitud(detecciones_correctas, verdaderos_negativos, total_rostros_video)
    
    # Prepara la llamada al Stored Procedure (los 11 parámetros originales)
    sp_call = "EXEC DET_InsertDataDeteccionModelo_SP ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?"
    
    params = (
        algoritmo,
        video_prueba,
        detecciones_correctas,
        total_rostros_video,
        precision,
        tiempo_respuesta_ms,
        falsos_positivos,
        falsos_negativos,
        fecha_simulacion,
        configuracion,
        exactitud
    )
    
    # Imprimir métricas adicionales calculadas por el módulo de métricas
    print(f"  [MÉTRICAS ADICIONALES] Recall: {recall:.2f}% | F1-Score: {f1_score:.2f}% | Especificidad: {especificidad:.2f}% | Tiempo/Frame: {tiempo_promedio_frame:.2f} ms")
    
    try:
        # Pasa los parámetros a la ejecución.
        cursor.execute(sp_call, params)
        conn.commit()
        print(f"Datos insertados correctamente en la base de datos. (Modelo: {algoritmo}, Video: {video_prueba})")
    except pyodbc.Error as ex:
        print(f"Error al insertar datos: {ex}")
        # Intentar reconectar y reintentar UNA vez
        print("Reintentando con nueva conexión...")
        try:
            conn = Conexion.get_db_connection()
            if conn:
                cursor2 = conn.cursor()
                cursor2.execute(sp_call, params)
                conn.commit()
                cursor2.close()
                print(f"Reintento exitoso. Datos guardados. (Modelo: {algoritmo}, Video: {video_prueba})")
            else:
                print(f"FALLO TOTAL: No se pudieron guardar los datos del video '{video_prueba}'.")
        except Exception as ex2:
            print(f"FALLO en reintento: {ex2}")
    finally:
        cursor.close()

def limpiar():
    finalizar_limpiar()

def cargar_formulario():
    global root, lblInfoVideoPath, lblVideo, btnVIdeo, btnCamara, btnCarpeta, btnEnd
    global btnGuardar, textNombre, guardar, btnEntrenar, btnReconocerFacial, btnDeteccionPura
    global lblEstado, lblDetecciones, lblContador, lblClasificacion, cap, MODELO_ACTIVO
    
    cap = None
    
    root = Tk()
    root.title("Sistema de Reconocimiento Facial con YOLO")
    root.state("zoomed")
    root.configure(bg="#f0f0f0")

    # ========== FRAME PRINCIPAL ==========s
    main_frame = Frame(root, bg="#f0f0f0")
    main_frame.pack(fill=BOTH, expand=True, padx=10, pady=10)

    # ========== TÍTULO ==========
    titulo_frame = Frame(main_frame, bg="#2c3e50", height=60)
    titulo_frame.pack(fill=X, pady=(0, 10))
    titulo_frame.pack_propagate(False)
    
    Label(titulo_frame, text="🎯 SISTEMA DE RECONOCIMIENTO FACIAL", 
          font=("Arial", 20, "bold"), bg="#2c3e50", fg="white").pack(pady=15)

    # ========== CONTENEDOR PRINCIPAL ==========
    contenedor = Frame(main_frame, bg="#f0f0f0")
    contenedor.pack(fill=BOTH, expand=True)

    # ========== PANEL IZQUIERDO (Video) ==========
    panel_video = Frame(contenedor, bg="white", relief=RIDGE, bd=2)
    panel_video.pack(side=LEFT, fill=BOTH, expand=True, padx=(0, 10))

    Label(panel_video, text="📹 VISUALIZACIÓN", font=("Arial", 14, "bold"), 
          bg="white", fg="#2c3e50").pack(pady=10)

    lblVideo = Label(panel_video, bg="black")
    lblVideo.pack(pady=10, padx=10, fill=BOTH, expand=True)

    # Info del video actual
    info_video_frame = Frame(panel_video, bg="white")
    info_video_frame.pack(fill=X, padx=10, pady=5)
    Label(info_video_frame, text="📁 Fuente:", font=("Arial", 10, "bold"), 
          bg="white").pack(side=LEFT)
    lblInfoVideoPath = Label(info_video_frame, text="Ningún video seleccionado", 
                             font=("Arial", 10), bg="white", fg="#7f8c8d")
    lblInfoVideoPath.pack(side=LEFT, padx=5)

    # ========== PANEL DERECHO (Controles) ==========
    panel_controles = Frame(contenedor, bg="white", relief=RIDGE, bd=2, width=320)
    panel_controles.pack(side=RIGHT, fill=Y)
    panel_controles.pack_propagate(False)

    Label(panel_controles, text="⚙️ CONTROLES", font=("Arial", 12, "bold"), 
          bg="white", fg="#2c3e50").pack(pady=8)

    # --- Sección: Entrada de Video ---
    seccion_entrada = LabelFrame(panel_controles, text="  Entrada de Video  ", 
                                 font=("Arial", 9, "bold"), bg="white", fg="#34495e", padx=10, pady=5)
    seccion_entrada.pack(fill=X, padx=10, pady=5)

    btnVIdeo = Button(seccion_entrada, text="📂 Elegir Video único", font=("Arial", 9),
                      bg="#3498db", fg="white", relief=FLAT, cursor="hand2",
                      command=lambda: video_de_entrada(1))
    btnVIdeo.pack(fill=X, pady=3, ipady=3)

    btnCarpeta = Button(seccion_entrada, text="📁 Procesar Lote (Carpeta)", font=("Arial", 9),
                      bg="#9b59b6", fg="white", relief=FLAT, cursor="hand2",
                      command=cargar_carpeta_videos)
    btnCarpeta.pack(fill=X, pady=3, ipady=3)

    btnCamara = Button(seccion_entrada, text="📷 Cámara en Directo", font=("Arial", 9),
                       bg="#2ecc71", fg="white", relief=FLAT, cursor="hand2",
                       command=lambda: video_de_entrada(2))
    btnCamara.pack(fill=X, pady=3, ipady=3)

    # --- Sección: Captura de Rostros ---
    seccion_captura = LabelFrame(panel_controles, text="  Captura de Rostros  ", 
                                 font=("Arial", 9, "bold"), bg="white", fg="#34495e", padx=10, pady=5)
    seccion_captura.pack(fill=X, padx=10, pady=5)

        # --- Selección de Modelo ---
    Label(seccion_captura, text="Modelo:", font=("Arial", 8),
        bg="white").pack(anchor=W, pady=(0, 3))

    modelos_disponibles = [
        "YOLO",
        "SSD",
        "FASTER R-CNN",
    ]

    modelo_seleccionado = StringVar()
    modelo_seleccionado.set(modelos_disponibles[0])

    cmbModelos = ttk.Combobox(
        seccion_captura,
        textvariable=modelo_seleccionado,
        values=modelos_disponibles,
        state="readonly",
        font=("Arial", 9)
    )
    cmbModelos.pack(fill=X, pady=(0, 8))

    def on_model_change(event=None):
        global MODELO_ACTIVO, cargando_modelo_estado, modelo_cargando_nombre, ssd_model, faster_rcnn_model
        MODELO_ACTIVO = modelo_seleccionado.get()
        lblEstado.config(text=f"Modelo activo: {MODELO_ACTIVO}", fg="blue")
        print(MODELO_ACTIVO)
        
        # Iniciar carga asíncrona si el modelo es pesado y no está en memoria
        if MODELO_ACTIVO == "SSD" and ssd_model is None:
            cargando_modelo_estado = True
            modelo_cargando_nombre = "SSD"
            threading.Thread(target=cargar_modelo_en_background, args=(MODELO_ACTIVO,), daemon=True).start()
        elif MODELO_ACTIVO == "FASTER R-CNN" and faster_rcnn_model is None:
            cargando_modelo_estado = True
            modelo_cargando_nombre = "FASTER R-CNN"
            threading.Thread(target=cargar_modelo_en_background, args=(MODELO_ACTIVO,), daemon=True).start()
            
    cmbModelos.bind("<<ComboboxSelected>>", on_model_change)
    # Inicializar
    MODELO_ACTIVO = modelo_seleccionado.get()

    Label(seccion_captura, text="Nombre:", font=("Arial", 8), 
          bg="white").pack(anchor=W, pady=(0, 3))
    
    textNombre = Entry(seccion_captura, font=("Arial", 10), relief=SOLID, bd=1, state="disabled")
    textNombre.pack(fill=X, pady=3, ipady=3)

    guardar = False
    btnGuardar = Button(seccion_captura, text="💾 Iniciar Captura", font=("Arial", 9),
                        bg="#e67e22", fg="white", relief=FLAT, cursor="hand2",
                        command=lambda: guardar_nombre(textNombre), state="disabled")
    btnGuardar.pack(fill=X, pady=3, ipady=3)

    lblContador = Label(seccion_captura, text="Imágenes capturadas: 0/300", 
                        font=("Arial", 8), bg="white", fg="#7f8c8d")
    lblContador.pack(pady=(3, 0))

    # --- Sección: Reconocimiento ---
    seccion_reconocimiento = LabelFrame(panel_controles, text="  Reconocimiento  ", 
                                       font=("Arial", 9, "bold"), bg="white", fg="#34495e", padx=10, pady=5)
    seccion_reconocimiento.pack(fill=X, padx=10, pady=5)

    btnEntrenar = Button(seccion_reconocimiento, text="🎓 Entrenar Modelo", font=("Arial", 9),
                        bg="#9b59b6", fg="white", relief=FLAT, cursor="hand2", state="disable",
                        command=lambda: entrenandoRF.entrenar_reconocedor_facil())
    btnEntrenar.pack(fill=X, pady=3, ipady=3)

    btnReconocerFacial = Button(seccion_reconocimiento, text="🔍 Reconocer Persona", 
                               font=("Arial", 9), bg="#1abc9c", fg="white", 
                               relief=FLAT, cursor="hand2", state="disabled",
                               command=activar_reconocimiento)
    btnReconocerFacial.pack(fill=X, pady=3, ipady=3)

    # === NUEVO BOTÓN: DETECCIÓN PURA (SIN RECONOCIMIENTO) ===
    btnDeteccionPura = Button(seccion_reconocimiento, text="📊 Detección Pura",
                              font=("Arial", 9), bg="#e67e22", fg="white",
                              relief=FLAT, cursor="hand2", state="disabled",
                              command=activar_deteccion_pura)
    btnDeteccionPura.pack(fill=X, pady=3, ipady=3)

    # --- Sección: Estado ---
    seccion_estado = LabelFrame(panel_controles, text="  Información  ", 
                               font=("Arial", 9, "bold"), bg="white", fg="#34495e", padx=10, pady=5)
    seccion_estado.pack(fill=X, padx=10, pady=5)

    lblEstado = Label(seccion_estado, text="Estado: Inactivo", font=("Arial", 8), 
                     bg="white", fg="gray", anchor=W)
    lblEstado.pack(fill=X, pady=1)

    lblDetecciones = Label(seccion_estado, text="Detecciones: 0", font=("Arial", 8), 
                          bg="white", fg="#34495e", anchor=W)
    lblDetecciones.pack(fill=X, pady=1)

    lblClasificacion = Label(seccion_estado, text="TP: 0 | FP: 0 | FN: 0 | TN: 0", 
                             font=("Arial", 8, "bold"), bg="white", fg="#8e44ad", anchor=W)
    lblClasificacion.pack(fill=X, pady=1)
    btnRegistro = Button(panel_controles, text="📝 Registro de Asistencia", font=("Arial", 10, "bold"),
                   bg="#16a085", fg="white", relief=FLAT, cursor="hand2",
                   state="normal", command=lambda: RegistroTrabajador.RegistroPersona(conn))
    btnRegistro.pack(fill=X, padx=10, pady=8, ipady=5)

    # --- Botón Finalizar ---
    btnEnd = Button(panel_controles, text="Finalizar", font=("Arial", 10, "bold"),
                   bg="#e74c3c", fg="white", relief=FLAT, cursor="hand2",
                   state="disabled", command=finalizar_limpiar)
    btnEnd.pack(fill=X, padx=10, pady=8, ipady=5)

    root.mainloop()


if __name__ == "__main__":
    cargar_formulario()