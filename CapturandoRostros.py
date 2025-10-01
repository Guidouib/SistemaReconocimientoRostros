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

faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
frame_count = 0
limite_imagenes = 300
modo_reconocerFacial = False
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "yolo11n-face.pt")
model = YOLO(model_path)
conn = Conexion.get_db_connection()

detecciones_totales = 0
video_actual_path = ""
tiempo_inicio_simulacion = None
after_id = None

def guardar_frame(frame, x, y, w, h):
    global frame_count, limite_imagenes

    if frame_count < limite_imagenes:
        rostro = frame[y:y+h, x:x+w]
        rostro = cv2.resize(rostro,(150,150),interpolation=cv2.INTER_CUBIC)
        frame_count += 1
        filename = os.path.join(personPath, f"frame_{frame_count:05d}.jpg")
        cv2.imwrite(filename, rostro)
        lblContador.config(text=f"Imágenes capturadas: {frame_count}/{limite_imagenes}")
    else:
        print("Se alcanzó el límite de 300 imágenes")

def deteccion_facial(frame):
    global detecciones_totales
    results = model.predict(frame, stream=False, verbose=False)[0]
    boxes = results.boxes.xyxy.cpu().numpy()
    for (x1, y1, x2, y2) in boxes:
        x, y = int(x1), int(y1)
        w, h = int(x2-x1), int(y2 - y1)
        detecciones_totales += 1
        if modo_reconocerFacial:
            nombre, color = ReconocimientoFacial.ReconocimiendoFacial(frame, x, y, w, h)
            cv2.putText(frame, nombre, (x, y - 25), 2, 1.1, color, 1, cv2.LINE_AA)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        else:
            guardar_frame(frame.copy(), x, y, w, h)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    lblDetecciones.config(text=f"Detecciones: {detecciones_totales}")
    return frame
    #gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    #faces= faceClassif.detectMultiScale(gray, 1.3, 5)
    #for (x, y, w, h) in faces:
    #    if modo_reconocerFacial:
    #        #ReconocimientoFacial.ReconocimiendoFacial(frame,x, y, w, h)
    #        nombre, color = ReconocimientoFacial.ReconocimiendoFacial(frame, x, y, w, h)
    #        cv2.putText(frame, nombre,(x,y-25),2,1.1, color, 1, cv2.LINE_AA)
    #        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
    #    else:
    #        guardar_frame(frame.copy(), x, y, w, h)
    #        cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255,0), 2)
    #    
    #return frame

def visualizar():
    global cap, personPath, tiempo_inicio_simulacion, detecciones_totales, after_id
    ret, frame = cap.read()

    if not cap or not cap.isOpened():
        return

    if ret == True:
        if tiempo_inicio_simulacion is None:
            tiempo_inicio_simulacion = datetime.datetime.now()
        #Redimencionar manteniendo aspecto
        max_width = 800#640
        max_height = 600#480
        h, w = frame.shape[:2]
        scale_w = max_width / w
        scale_h = max_height / h
        scale = min(scale_w, scale_h, 1.0)
        new_w = int(w * scale)
        new_h = int (h * scale)

        frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

        if guardar == True and btnGuardar['state'] == NORMAL or modo_reconocerFacial:
            personName = textNombre.get()
            output_dir = 'C:/Users/GuidoUib/Desktop/TESIS OFICIAL/APP-PYTHON/REC_FACIAL_UIB/PRIMCIPAL/Data'
            personPath = output_dir + '/' + personName
            if not os.path.exists(personPath):
                os.makedirs(personPath)

            frame = deteccion_facial(frame) 

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(frame)
        img = ImageTk.PhotoImage(image = im)

        lblVideo.configure(image = img)
        lblVideo.image = img
        after_id = lblVideo.after(10, visualizar)
    else:
        finalizar_guardar_resultado()

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
        video_actual_path= "Cámara en directo."
        lblInfoVideoPath.configure(text="Cámara en Directo")
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    btnVIdeo.configure(state="disabled")
    btnCamara.configure(state="disabled")
    btnEnd.configure(state="normal")
    btnGuardar.configure(state="normal")
    textNombre.config(state="normal")
    btnReconocerFacial.configure(state="normal")
    
    detecciones_totales = 0
    frame_count = 0
    tiempo_inicio_simulacion = None
    lblEstado.config(text="Estado: Procesando...", fg="blue")
    visualizar()

def finalizar_limpiar():
    global cap, after_id, modo_reconocerFacial
    
    if after_id is not None:
        lblVideo.after_cancel(after_id)
        after_id = None

    if cap and cap.isOpened():
        cap.release()
    
    modo_reconocerFacial = False
    
    lblVideo.image = ""
    lblInfoVideoPath.configure(text="Ningún video seleccionado")
    btnVIdeo.configure(state="normal")
    btnCamara.configure(state="normal")
    btnGuardar.configure(state="disabled")
    textNombre.config(state="disabled")
    btnEntrenar.configure(state="normal")
    btnEnd.configure(state="disabled")
    lblEstado.config(text="Estado: Inactivo", fg="gray")
    lblDetecciones.config(text="Detecciones: 0")
    lblContador.config(text="Imágenes capturadas: 0/300")
    if cap and cap.isOpened():
        cap.release()

def guardar_nombre(textNombre):
    global personName, guardar, frame_count
    personName = textNombre.get()
    
    if personName.strip() == "":
        lblEstado.config(text="Estado: ⚠ Ingrese un nombre válido", fg="red")
        return
    
    guardar = True
    frame_count = 0
    btnGuardar.configure(state="disabled")
    textNombre.config(state="disabled")
    lblEstado.config(text=f"Estado: Guardando rostros para '{personName}'", fg="green")

def finalizar_guardar_resultado():
    global cap, detecciones_totales, tiempo_inicio_simulacion, video_actual_path

    tiempo_fin_simulacion = datetime.datetime.now()
    tiempo_total_ms = round((tiempo_fin_simulacion - tiempo_inicio_simulacion).total_seconds() * 1000, 3) if tiempo_inicio_simulacion else 0

    fecha_simulacion_str = tiempo_fin_simulacion.strftime('%Y-%m-%d %H:%M:%S')
    insertar_Resultado_Deteccion(
        algoritmo="YOLO", 
        video_prueba=os.path.basename(video_actual_path) if video_actual_path else "Camara en Directo",
        detecciones_correctas=detecciones_totales,
        total_rostros_video=0, # Este valor debe ser un conteo manual del video
        tiempo_respuesta_ms=tiempo_total_ms,
        falsos_positivos=0, # Este valor debe ser validado manualmente
        falsos_negativos=0, # Este valor debe ser validado manualmente
        configuracion="Video grabado con celular.",
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

    except pyodbc.Error as ex:
        print(f"Error al insertar datos: {ex.args[0]}")
    finally:
        cursor.close()

def limpiar():
    finalizar_limpiar()

def cargar_formulario():
    global root, lblInfoVideoPath, lblVideo, btnVIdeo, btnCamara, btnEnd
    global btnGuardar, textNombre, guardar, btnEntrenar, btnReconocerFacial
    global lblEstado, lblDetecciones, lblContador, cap
    
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
                        bg="#9b59b6", fg="white", relief=FLAT, cursor="hand2",
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
                   state="normal", command=lambda: print("Registro de asistencia"))
    btnRegistro.pack(fill=X, padx=10, pady=8, ipady=5)

    # --- Botón Finalizar ---
    btnEnd = Button(panel_controles, text="⏹ Finalizar", font=("Arial", 10, "bold"),
                   bg="#e74c3c", fg="white", relief=FLAT, cursor="hand2",
                   state="disabled", command=finalizar_limpiar)
    btnEnd.pack(fill=X, padx=10, pady=8, ipady=5)

    root.mainloop()


if __name__ == "__main__":
    cargar_formulario()
    