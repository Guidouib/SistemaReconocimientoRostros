# Diagrama del Proceso de Desarrollo e Integración del Modelo
## Sistema de Reconocimiento Facial (Tesis)

Este documento describe, de manera detallada, el flujo completo del sistema: desde la captura de rostros, pasando por el entrenamiento del reconocedor (ArcFace/DeepFace), hasta la inferencia en tiempo real con los tres detectores intercambiables (YOLOv11-face, SSD300, Faster R-CNN), el pipeline en dos etapas, el hilo de reconocimiento asíncrono, el cálculo de métricas y la persistencia en SQL Server.

---

## 1. Visión general de módulos

| Módulo | Responsabilidad |
|---|---|
| `CapturandoRostros.py` | Orquestador principal (GUI Tkinter). Captura de video/cámara/lote, detección, hilo de reconocimiento, UI y flujo de estados. |
| `entrenandoRF.py` | Entrenamiento offline: recorre `Data/<DNI>/*.jpg`, genera embeddings ArcFace y serializa `face_embeddings.pkl`. |
| `ReconocimientoFacial.py` | Inferencia: carga `face_embeddings.pkl`, calcula distancia coseno contra los embeddings conocidos y clasifica TP/FP/FN/TN/FILTRADO. |
| `Conexion.py` | Conexión pyodbc a SQL Server (`ControlAccesoBD`). |
| `RegistroTrabajador.py` | CRUD de personal (formulario Tkinter) vía stored procedures. |
| `metricas.py` | Cálculo centralizado de Precisión, Recall, F1, Exactitud, Especificidad, tiempo/frame. |
| `check_gpu.py` | Diagnóstico de disponibilidad de CUDA y benchmark de Faster R-CNN. |
| `BD/01_MD.sql`, `BD/03_SP.sql` | Modelo de datos y stored procedures (SQL Server). |
| `tests/` | Suite de pruebas unitarias (conexión, detección, reconocimiento, métricas, benchmark). |

---

## 2. Diagrama general del pipeline (extremo a extremo)

```mermaid
flowchart TD
    subgraph FASE_1["FASE 1 · Adquisición de datos"]
        A1[Usuario abre GUI\nCapturandoRostros.py] --> A2{Selecciona fuente}
        A2 -->|Video único| A3[cv2.VideoCapture path]
        A2 -->|Cámara en vivo| A4[cv2.VideoCapture 0]
        A2 -->|Carpeta / Lote| A5[glob *.mp4 *.avi\ncola_videos]
        A3 --> A6[Ingresa DNI de 8 dígitos]
        A4 --> A6
        A6 --> A7[Detección facial YOLO\nrecorte 150x150]
        A7 --> A8[(Data/DNI/frame_00001.jpg\n...\nframe_00300.jpg)]
    end

    subgraph FASE_2["FASE 2 · Entrenamiento del reconocedor"]
        B1[Click Entrenar Modelo] --> B2[entrenandoRF.entrenar_reconocedor_facil]
        B2 --> B3[Recorre Data/&lt;DNI&gt;/*.jpg]
        B3 --> B4[DeepFace.represent\nmodel_name=ArcFace\nenforce_detection=True]
        B4 --> B5{¿Rostro detectado\ny confiable?}
        B5 -->|No| B6[Descarta imagen\nlog de advertencia]
        B5 -->|Sí| B7[Embedding 512-d\n+ nombre DNI]
        B7 --> B8[(face_embeddings.pkl\nembeddings + names)]
    end

    subgraph FASE_3["FASE 3 · Inferencia en tiempo real"]
        C1[visualizar· loop 10ms\nTkinter after] --> C2[Frame BGR redimensionado\nal tamaño del label]
        C2 --> C3{Modelo activo}
        C3 -->|YOLO| C4[YOLOv11n-face\nconf >= 0.6\nbbox = rostro directo]
        C3 -->|SSD| C5[SSD300-VGG16\nlabel=persona, score>0.5\nskip_interval=3 frames\nbbox = cuerpo completo]
        C3 -->|FASTER R-CNN| C6[FasterRCNN-ResNet50-FPN\nlabel=persona, score>0.5\nskip_interval=3 frames\nbbox = cuerpo completo]
        C4 --> C7[procesar_deteccion]
        C5 --> C7X{¿Es SSD o F-RCNN?}
        C6 --> C7X
        C7X -->|Sí: pipeline 2 etapas| C7Y[Haar Cascade\ndentro del bbox del cuerpo]
        C7Y -->|Sin rostro| C7Z[Clasifica FILTRADO\nNo Autorizado]
        C7Y -->|Con rostro| C7
        C7 --> C8{Modo activo}
        C8 -->|Detección pura| C9[Solo dibuja bbox naranja\ncuenta detecciones]
        C8 -->|Reconocimiento| C10[Cache de rostros por\ndistancia euclidiana entre centros]
    end

    subgraph FASE_4["FASE 4 · Reconocimiento asíncrono"]
        D1[Hilo daemon\nworker_reconocimiento] --> D2[recognition_queue.get\nbloqueante]
        D2 --> D3[ReconocimientoFacial.\nReconocimiendoFacial]
        D3 --> D4{Validaciones defensivas}
        D4 -->|rostro vacío / w,h < 80 / blur < 50| D5[FILTRADO\nNo Autorizado]
        D4 -->|OK| D6[DeepFace.represent\nArcFace, enforce_detection=False]
        D6 --> D7[Distancia coseno contra\nknown_embeddings]
        D7 --> D8{Umbrales}
        D8 -->|dist < 0.50| D9[TP · Autorizado · Verde]
        D8 -->|0.50-0.60| D10[FP · Autorizado· Naranja]
        D8 -->|0.60-0.70| D11[FN · No Autorizado · Rojo-naranja]
        D8 -->|>= 0.70| D12[TN · No Autorizado · Rojo]
        D9 & D10 & D11 & D12 --> D13[Actualiza face_data_cache\nnombre, color, clasificación]
    end

    subgraph FASE_5["FASE 5 · Métricas y persistencia"]
        E1[Fin de video / EOF] --> E2{Modo}
        E2 -->|Reconocimiento| E3[metricas.generar_reporte\nTP FP FN TN]
        E2 -->|Detección pura| E4[Cálculo FPS y\ntiempo/frame promedio]
        E3 --> E5[insertar_Resultado_Deteccion]
        E4 --> E6[insertar_Resultado_Deteccion_Pura]
        E5 --> E7[EXEC DET_InsertDataDeteccionModelo_SP]
        E6 --> E8[EXEC DET_InsertDataDeteccionPura_SP]
        E7 --> E9[(SQL Server\nControlAccesoBD\nSI_FinRendimientoModelos)]
        E8 --> E10[(SQL Server\nControlAccesoBD\nSI_FinRendimientoDeteccionPura)]
    end

    A8 -.-> B1
    B8 -.->|cargar_modelo al iniciar\ny bajo demanda| D3
    C10 --> D1
    C9 --> E1
    C10 --> E1
    E9 -.reintento en fallo.-> E7
    E10 -.reintento en fallo.-> E8

    style FASE_1 fill:#eaf6ff,stroke:#3498db
    style FASE_2 fill:#f3eaff,stroke:#9b59b6
    style FASE_3 fill:#eafff1,stroke:#2ecc71
    style FASE_4 fill:#fff6ea,stroke:#e67e22
    style FASE_5 fill:#ffeaea,stroke:#e74c3c
```

---

## 3. Detalle Fase 1 — Captura de rostros (`CapturandoRostros.py`, `guardar_frame`)

1. El usuario elige fuente: **video único**, **cámara en vivo (DirectShow)** o **procesamiento por lote** (carpeta completa, ordenada alfabéticamente, con reanudación desde el video seleccionado).
2. `limite_imagenes` se adapta dinámicamente:
   - Video: `CAP_PROP_FRAME_COUNT` del archivo (fallback 300).
   - Cámara: fijo en 1000.
3. El usuario ingresa un **DNI de 8 dígitos** (validado con `.isdigit()` y `len == 8`) → crea `Data/<DNI>/`.
4. En cada frame, YOLO detecta el rostro; `guardar_frame()` recorta la caja, la reescala a **150×150 px** (`INTER_CUBIC`) y la guarda como `frame_%05d.jpg`.
5. Se actualiza el contador visual `Imágenes capturadas: X/N`.

## 4. Detalle Fase 2 — Entrenamiento (`entrenandoRF.py`)

1. Recorre cada subcarpeta de `Data/` (una por DNI).
2. Para cada imagen `.jpg/.jpeg/.png`, invoca `DeepFace.represent(model_name="ArcFace", enforce_detection=True)`.
3. Si DeepFace no logra detectar un rostro confiable, la imagen se descarta (log de advertencia), evitando contaminar el dataset de embeddings.
4. Acumula `known_embeddings` (vectores 512-d) y `known_names` (DNI asociado).
5. Serializa el resultado con `pickle` en `face_embeddings.pkl` — este archivo es el **modelo entrenado** que consume la fase de reconocimiento.

## 5. Detalle Fase 3 — Detección en tiempo real (tres backends intercambiables)

| Modelo | Framework | Objeto detectado | Umbral | Particularidad |
|---|---|---|---|---|
| **YOLO** (`yolo11n-face.pt`) | Ultralytics YOLOv11 | Rostro directo | conf ≥ 0.6 | Recorte = rostro, listo para ArcFace. |
| **SSD** (`ssd300_vgg16`) | Torchvision | Persona/cuerpo (label=1) | score > 0.5 | Requiere **pipeline en 2 etapas** (Haar Cascade) para extraer el rostro del cuerpo. Inferencia cada 3 frames (`_heavy_skip_interval`), con caché de detecciones entre saltos. |
| **Faster R-CNN** (`fasterrcnn_resnet50_fpn`) | Torchvision | Persona/cuerpo (label=1) | score > 0.5 | Igual que SSD: pipeline de 2 etapas + skip de frames. Carga asíncrona en hilo aparte (`cargar_modelo_en_background`) para no congelar la UI. |

**Pipeline en dos etapas (SSD/Faster R-CNN → Haar Cascade → ArcFace):**
Cuando el modelo activo detecta un *cuerpo completo*, `extraer_rostro_de_cuerpo()` aplica `haarcascade_frontalface_default.xml` dentro de ese bounding box para aislar el rostro (toma el más grande si hay varios). Si Haar no encuentra nada, el resultado se marca directamente como `FILTRADO / No Autorizado` sin llamar a ArcFace.

**Modos de operación (mutuamente excluyentes):**
- **Detección pura** (`modo_deteccion_pura`): solo mide velocidad/volumen (conteo, FPS, tiempo/frame). No llama a ArcFace.
- **Reconocimiento** (`modo_reconocerFacial`): además de detectar, identifica la identidad y clasifica TP/FP/FN/TN.

## 6. Detalle Fase 4 — Reconocimiento asíncrono (`worker_reconocimiento` + `ReconocimientoFacial.py`)

- Un **hilo demonio** (`threading.Thread`) consume una `queue.Queue()` para no bloquear el loop de UI (`visualizar()`, refrescado cada 10 ms).
- **Caché de identidad por posición**: cada rostro detectado se compara por distancia euclidiana de centros (`< 50 px`) contra `face_data_cache` para no reprocesar la misma cara en cada frame; usa un contador `skip` para espaciar las llamadas costosas a ArcFace.
- **Validaciones defensivas antes de invocar DeepFace**: rostro vacío, `w < 80 or h < 80`, o nitidez insuficiente (`cv2.Laplacian(...).var() < 50`) → clasificación inmediata `FILTRADO`.
- **Clasificación por umbral de distancia coseno** (`calcular_distancia_coseno`, ArcFace embeddings):

  | Distancia | Clasificación | Color UI |
  |---|---|---|
  | `< 0.50` | **TP** (Autorizado, alta confianza) | Verde |
  | `0.50 – 0.60` | **FP** (Autorizado, dudoso) | Naranja |
  | `0.60 – 0.70` | **FN** (No Autorizado, cercano al umbral) | Rojo-naranja |
  | `≥ 0.70` | **TN** (No Autorizado) | Rojo |

## 7. Detalle Fase 5 — Métricas y persistencia (`metricas.py`, `Conexion.py`, SQL Server)

**Cálculo de métricas** (`metricas.generar_reporte`):
`Precisión = TP/(TP+FP)`, `Recall = TP/(TP+FN)`, `F1 = 2·P·R/(P+R)`, `Exactitud = (TP+TN)/Total`, `Especificidad = TN/(TN+FP)`, `Tiempo/Frame = tiempo_total/frames`.

**Persistencia en SQL Server** (`ControlAccesoBD`, vía `pyodbc` + ODBC Driver 17):

- `_asegurar_conexion()` verifica la conexión (`SELECT 1`) y reconecta si se perdió, antes de cada inserción — con **reintento automático** en caso de fallo del `INSERT`.
- Modo *reconocimiento* → `EXEC DET_InsertDataDeteccionModelo_SP` → tabla `SI_FinRendimientoModelos` (algoritmo, video, aciertos, precisión, exactitud, FP, FN, tiempo, fecha).
- Modo *detección pura* → `EXEC DET_InsertDataDeteccionPura_SP` → tabla `SI_FinRendimientoDeteccionPura` (algoritmo, video, total detecciones, frames, FPS promedio, tiempo/frame).
- Módulo `RegistroTrabajador.py` gestiona altas/bajas/lectura de personal vía `DET_InsertDatosPersonal_SP`, `DET_VigenciaDatosPersonal_SP`, `DET_LeerDatosPersonal_SP` sobre la tabla `SI_FinPersonal`.

---

## 8. Diagrama de secuencia — un ciclo de reconocimiento sobre un frame

```mermaid
sequenceDiagram
    participant UI as Tkinter (visualizar)
    participant Det as Detector activo (YOLO/SSD/F-RCNN)
    participant Haar as Haar Cascade (solo SSD/F-RCNN)
    participant Cache as face_data_cache
    participant Q as recognition_queue
    participant Worker as worker_reconocimiento (hilo)
    participant RF as ReconocimientoFacial (ArcFace)
    participant DB as SQL Server (ControlAccesoBD)

    UI->>Det: frame BGR redimensionado
    Det->>Det: inferencia (bbox persona/rostro)
    alt SSD o Faster R-CNN
        Det->>Haar: recorte del cuerpo
        Haar-->>Det: recorte del rostro o None
    end
    Det->>Cache: procesar_deteccion(x,y,w,h)
    Cache->>Cache: match por distancia de centros (<50px)
    alt rostro nuevo o skip agotado
        Cache->>Q: put(rostro_recorte, item)
        Q->>Worker: get() (bloqueante)
        Worker->>RF: ReconocimiendoFacial(rostro)
        RF->>RF: validaciones (tamaño, nitidez)
        RF->>RF: DeepFace.represent (ArcFace)
        RF->>RF: distancia coseno vs known_embeddings
        RF-->>Worker: (nombre, color, distancia, clasificación)
        Worker->>Cache: actualiza item (name, color, skip=10)
    end
    Cache-->>UI: bbox + etiqueta coloreada
    UI->>UI: contadores TP/FP/FN/TN en pantalla

    Note over UI,DB: Al finalizar el video (EOF)
    UI->>DB: EXEC DET_InsertDataDeteccionModelo_SP (o _Pura)
    DB-->>UI: commit / reintento si falla conexión
```

---

## 9. Notas de arquitectura relevantes

- **Desacoplamiento detección/reconocimiento**: la detección de caja (YOLO/SSD/F-RCNN) corre en el hilo de UI a 10 ms de refresco; el reconocimiento (costoso, ArcFace) corre en un hilo aparte consumiendo una cola, evitando congelar el video.
- **Optimización de modelos pesados**: SSD y Faster R-CNN solo infieren cada 3 frames (`_heavy_skip_interval`), reutilizando las últimas detecciones cacheadas en los frames intermedios; además se cargan en segundo plano (`threading.Thread`) la primera vez que se seleccionan en el combobox, mostrando un overlay "Cargando..." mientras tanto.
- **GPU/CPU**: `check_gpu.py` diagnostica CUDA; si hay GPU disponible, SSD y Faster R-CNN se mueven a `cuda` y usan `torch.amp.autocast` para inferencia mixta.
- **Resiliencia de BD**: toda inserción verifica la conexión antes de escribir y reintenta una vez reconectando si `pyodbc.Error` ocurre — pensado para sesiones largas de procesamiento por lote.
- **Modo lote (`cargar_carpeta_videos`)**: encola todos los videos `.mp4/.avi` de una carpeta desde el video seleccionado en adelante, preguntando una sola vez si el lote completo corre en modo reconocimiento o detección pura, y encadena automáticamente el siguiente video al finalizar cada uno (`iniciar_siguiente_video_de_cola`).
- **Pruebas** (`tests/`): cubren conexión a BD, detección, reconocimiento, cálculo de métricas y benchmarking — validando cada fase del pipeline de forma aislada.

---

## 10. Modelo de datos (resumen)

```mermaid
erDiagram
    SI_FinPersonal {
        int DNI PK
        string Nombres
        string ApellidoPaterno
        string ApellidoMaterno
        string Direccion
        string Correo
        string Celular
        int TipoTrabajador
        bit Vigente
    }
    SI_FinRendimientoModelos {
        int Id PK
        string Algoritmo
        string VideoPrueba
        int DeteccionesCorrectas
        int TotalRostrosVideo
        float Precision
        float Exactitud
        float TiempoRespuestaMs
        int FalsosPositivos
        int FalsosNegativos
        string Configuracion
        datetime FechaSimulacion
    }
    SI_FinRendimientoDeteccionPura {
        int Id PK
        string Algoritmo
        string VideoPrueba
        int TotalDetecciones
        int TotalFramesProcesados
        float TiempoRespuestaMs
        float FpsPromedio
        float TiempoPromedioFrameMs
        datetime FechaSimulacion
        string Configuracion
    }
```
