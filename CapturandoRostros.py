from tkinter import *
from tkinter import filedialog
from PIL import Image
from PIL import ImageTk
from ultralytics import YOLO 
import cv2
import imutils
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

faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
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
modelo_activo = None

# Cache para reconocimiento facial
face_data_cache = []
recognition_queue = queue.Queue()

def worker_reconocimiento():
    """Hilo en segundo plano para procesar reconocimiento facial sin bloquear UI"""
    while True:
        try:
            # Obtener tarea (bloqueante, pero en hilo aparte)
            task = recognition_queue.get()
            if task is None: break # Señal de parada
            
            rostro_img, face_data = task
            
            # Ejecutar reconocimiento (esto es lo que demoraba)
            # Pasamos x,y,w,h como 0 porque ya es un recorte
            h, w, _ = rostro_img.shape
            nombre, color = ReconocimientoFacial.ReconocimiendoFacial(rostro_img, 0, 0, w, h)
            
            # Actualizar diccionario compartido
            face_data['name'] = nombre
            face_data['color'] = color
            face_data['skip'] = 10  # Aumentamos skip ya que es asincrono
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
    
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    detecciones_totales += 1
    
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
            # === ROSTRO CONOCIDO (EN CACHE) ===
            item = face_data_cache[best_match_index]
            
            # Verificar si necesita actualización
            # Si skip <= 0 Y no se está procesando ya
            if item['skip'] <= 0 and not item.get('pending', False):
                 item['pending'] = True
                 # Enviar copia del recorte a la cola
                 rostro_recorte = frame[y:y+h, x:x+w].copy()
                 recognition_queue.put((rostro_recorte, item))
            else:
                 # Decrementar contador si no está pendiente
                 if not item.get('pending', False):
                     item['skip'] -= 1

            # Actualizar posición y visto
            item['x'], item['y'], item['w'], item['h'] = x, y, w, h
            item['seen'] = True
            
            # Usar valores actuales del cache (pueden ser "Procesando" o el nombre anterior)
            nombre_mostrar = item['name']
            color_mostrar = item['color']
            
        else:
            # === ROSTRO NUEVO ===
            # Crear entrada inicial
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
    else:
        # Solo guardar si estamos en modo captura (no reconocimiento)
        # y si se ha definido 'guardar' en el scope global (que se usa en 'visualizar')
        # Dado que esta funcion solo detecta, la logica de guardado original estaba acoplada.
        # Para mantener compatibilidad con 'visualizar' que llama a 'deteccion_facial':
        pass 
        # NOTA: La lógica de guardar_frame se movió fuera del bucle de visualización en el código original?
        # Revisando codigo anterior: guardar_frame se llamaba DENTRO del loop de detecciones.
        # Restoramos comportamiento original.
        if 'guardar' in globals() and guardar: # Chequeo defensivo
             guardar_frame(frame.copy(), x, y, w, h)

    return frame

def detectar_rostro_yolo(frame):
    # Aumentamos confianza a 0.6 para evitar detectar objetos random como caras
    results = yolo_model.predict(frame, stream=False, verbose=False, conf=0.6)[0]
    boxes = results.boxes.xyxy.cpu().numpy()
    
    for (x1, y1, x2, y2) in boxes:
        x, y = int(x1), int(y1)
        w, h = int(x2-x1), int(y2 - y1)
        procesar_deteccion(frame, x, y, w, h)
    return frame

def detectar_rostro_ssd(frame):
    global ssd_model
    if ssd_model is None:
        print("Cargando modelo SSD (PyTorch)...")
        # Usamos SSD300 VGG16 preentrenado. Detecta 80 clases COCO. Persona = 1.
        ssd_model = torchvision.models.detection.ssd300_vgg16(weights=torchvision.models.detection.SSD300_VGG16_Weights.DEFAULT)
        ssd_model.eval()
        if torch.cuda.is_available():
            ssd_model.to('cuda')
            
    # Preprocesamiento
    transform = T.Compose([T.ToTensor()])
    img_tensor = transform(frame).to('cuda' if torch.cuda.is_available() else 'cpu')
    
    with torch.no_grad():
        detections = ssd_model([img_tensor])[0]
    
    # Filtrar detecciones (Clase 1 = Person en COCO)
    confidence_threshold = 0.5
    for i in range(len(detections['boxes'])):
        score = detections['scores'][i].item()
        label = detections['labels'][i].item()
        
        if score > confidence_threshold and label == 1: # 1 es Persona
            box = detections['boxes'][i].cpu().numpy()
            x1, y1, x2, y2 = box.astype(int)
            x, y = x1, y1
            w, h = x2 - x1, y2 - y1
            procesar_deteccion(frame, x, y, w, h)
            
    return frame

def detectar_rostro_faster_rcnn(frame):
    global faster_rcnn_model
    if faster_rcnn_model is None:
        print("Cargando modelo Faster R-CNN (PyTorch)...")
        # Faster R-CNN ResNet50
        faster_rcnn_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
        faster_rcnn_model.eval()
        if torch.cuda.is_available():
            faster_rcnn_model.to('cuda')

    # Preprocesamiento
    transform = T.Compose([T.ToTensor()])
    img_tensor = transform(frame).to('cuda' if torch.cuda.is_available() else 'cpu')
    
    with torch.no_grad():
        detections = faster_rcnn_model([img_tensor])[0]
    
    # Filtrar
    confidence_threshold = 0.5
    for i in range(len(detections['boxes'])):
        score = detections['scores'][i].item()
        label = detections['labels'][i].item()
        
        if score > confidence_threshold and label == 1: # 1 es Persona
            box = detections['boxes'][i].cpu().numpy()
            x1, y1, x2, y2 = box.astype(int)
            x, y = x1, y1
            w, h = x2 - x1, y2 - y1
            procesar_deteccion(frame, x, y, w, h)
    
    return frame

def deteccion_facial(frame):
    global detecciones_totales, MODELO_ACTIVO, face_data_cache
    
    # 1. Marcar todos como no vistos al inicio del frame
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
        # Default
        result_frame = detecting_rostro_yolo_wrapper(frame)
        
    # 2. Limpiar caché de caras que ya no están visibles
    # Filtramos la lista conservando solo los que seen == True
    face_data_cache[:] = [f for f in face_data_cache if f['seen']]
    
    return result_frame

# Wrapper para mantener nombre consistente si es necesario, 
# aunque en el codigo original se llamaba detectar_rostro_yolo directamente.
# Haremos un alias temporal para no romper la logica del if/else si modificamos nombres,
# pero aqui simplemente llamamos al existente.
def detecting_rostro_yolo_wrapper(frame):
    return detectar_rostro_yolo(frame)

def visualizar():
    global cap, personPath, tiempo_inicio_simulacion, detecciones_totales, after_id, guardar
    ret, frame = cap.read()

    if not cap or not cap.isOpened():
        return

    if ret == True:
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
        if guardar == True and btnGuardar['state'] == DISABLED or modo_reconocerFacial:
             # Solo crear directorios si vamos a guardar realmente
             if guardar:
                personName = textNombre.get()
                output_dir = 'C:/Users/ASUS TUF/Desktop/Tesis Of/SistemaReconocimientoRostros/Data'
                personPath = output_dir + '/' + personName
                if not os.path.exists(personPath):
                    os.makedirs(personPath)

             frame = deteccion_facial(frame) 
             lblDetecciones.config(text=f"Detecciones: {detecciones_totales}")

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(frame)
        img = ImageTk.PhotoImage(image = im)

        lblVideo.configure(image = img)
        lblVideo.image = img
        after_id = lblVideo.after(10, visualizar)
    else:
        if modo_reconocerFacial:
           finalizar_guardar_resultado()
        else:
           finalizar_limpiar()
           
def activar_reconocimiento():
    global modo_reconocerFacial
    modo_reconocerFacial = True
    btnReconocerFacial.configure(state="disabled", text="✓ Reconocimiento Activo")
    lblEstado.config(text="Estado: Reconociendo rostros", fg="green")

def video_de_entrada(opcion):
    global cap, video_actual_path, detecciones_totales, tiempo_inicio_simulacion, frame_count
    if opcion == 1:
        path_video = filedialog.askopenfilename(
            filetypes=[("Video files", "*.mp4 *.avi")])
        
        if len(path_video) > 0:
            video_actual_path = path_video
            lblInfoVideoPath.configure(text=os.path.basename(path_video))
            cap = cv2.VideoCapture(path_video)
    if opcion == 2:
        # Codigo original tenia video_actual_path fijo para camara
        video_actual_path= "Camara" 
        lblInfoVideoPath.configure(text="Cámara en Directo")
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    btnVIdeo.configure(state="disabled")
    btnCamara.configure(state="disabled")
    btnEnd.configure(state="normal")
    btnGuardar.configure(state="normal")
    textNombre.config(state="normal")
    btnEntrenar.configure(state="normal")
    btnReconocerFacial.configure(state="normal")
    
    detecciones_totales = 0
    frame_count = 0
    tiempo_inicio_simulacion = None
    lblEstado.config(text="Estado: Procesando...", fg="blue")
    visualizar()

def finalizar_limpiar():
    global cap, after_id, modo_reconocerFacial, guardar
    
    if after_id is not None:
        lblVideo.after_cancel(after_id)
        after_id = None

    if cap and cap.isOpened():
        cap.release()
    
    modo_reconocerFacial = False
    guardar = False
    
    lblVideo.image = ""
    lblInfoVideoPath.configure(text="Ningún video seleccionado")
    btnVIdeo.configure(state="normal")
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
    btnReconocerFacial.configure(state="disabled")
    if cap and cap.isOpened():
        cap.release()

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

    tiempo_fin_simulacion = datetime.datetime.now()
    tiempo_total_ms = round((tiempo_fin_simulacion - tiempo_inicio_simulacion).total_seconds() * 1000, 3) if tiempo_inicio_simulacion else 0

    fecha_simulacion_str = tiempo_fin_simulacion.strftime('%Y-%m-%d %H:%M:%S')
    insertar_Resultado_Deteccion(
        algoritmo=MODELO_ACTIVO, 
        video_prueba=os.path.basename(video_actual_path) if video_actual_path else "Camara en Directo",
        detecciones_correctas=detecciones_totales,
        total_rostros_video=0, # Este valor debe ser un conteo manual del video
        tiempo_respuesta_ms=tiempo_total_ms,
        falsos_positivos=0, # Este valor debe ser validado manualmente
        falsos_negativos=0, # Este valor debe ser validado manualmente
        configuracion="Video grabado con cámara de video vigilancia.",
        fecha_simulacion = fecha_simulacion_str

    )
    limpiar()

def insertar_Resultado_Deteccion(algoritmo, video_prueba, detecciones_correctas, total_rostros_video, tiempo_respuesta_ms, falsos_positivos, falsos_negativos, configuracion, fecha_simulacion):
    """
    Inserta un nuevo registro en la tabla SI_FinRendimientoModelos llamando a un procedimiento almacenado.
    """
    global conn
    
    if conn is None:
        print("Error: No se puede conectar a la base de datos.")
        return

    cursor = conn.cursor()
    
    # Prepara la llamada al Stored Procedure
    sp_call = "EXEC DET_InsertDataDeteccionModelo_SP ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?"
    
    # La precisión se calcula en Python antes de enviar los datos
    precision = (detecciones_correctas / total_rostros_video) * 100 if total_rostros_video > 0 else 0
    #aCCURACY
    total_casos_evaluados = detecciones_correctas + falsos_positivos + falsos_negativos
    exactitud = (detecciones_correctas / total_casos_evaluados) * 100 if total_casos_evaluados > 0 else 0
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
    try:
        # Pasa los parámetros a la ejecución.
        cursor.execute(sp_call, params)
        conn.commit()
        print("Datos insertados correctamente en la base de datos.")
        finalizar_limpiar()
    except pyodbc.Error as ex:
        print(f"Error al insertar datos: {ex.args[0]}")
    finally:
        cursor.close()

def limpiar():
    finalizar_limpiar()

def cargar_formulario():
    global root, lblInfoVideoPath, lblVideo, btnVIdeo, btnCamara, btnEnd
    global btnGuardar, textNombre, guardar, btnEntrenar, btnReconocerFacial
    global lblEstado, lblDetecciones, lblContador, cap, MODELO_ACTIVO
    
    cap = None  # Inicializar cap como None
    
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

    btnVIdeo = Button(seccion_entrada, text="📂 Elegir Video", font=("Arial", 9),
                      bg="#3498db", fg="white", relief=FLAT, cursor="hand2",
                      command=lambda: video_de_entrada(1))
    btnVIdeo.pack(fill=X, pady=3, ipady=3)

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
        global MODELO_ACTIVO
        MODELO_ACTIVO = modelo_seleccionado.get()
        lblEstado.config(text=f"Modelo activo: {MODELO_ACTIVO}", fg="blue")
        print(MODELO_ACTIVO)
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
    btnRegistro = Button(panel_controles, text="📝 Registro de Asistencia", font=("Arial", 10, "bold"),
                   bg="#16a085", fg="white", relief=FLAT, cursor="hand2",
                   state="normal", command=lambda: RegistroTrabajador.RegistroPersona(conn))
    btnRegistro.pack(fill=X, padx=10, pady=8, ipady=5)

    # --- Botón Finalizar ---
    btnEnd = Button(panel_controles, text="⏹ Finalizar", font=("Arial", 10, "bold"),
                   bg="#e74c3c", fg="white", relief=FLAT, cursor="hand2",
                   state="disabled", command=finalizar_limpiar)
    btnEnd.pack(fill=X, padx=10, pady=8, ipady=5)

    root.mainloop()


if __name__ == "__main__":
    cargar_formulario()
    