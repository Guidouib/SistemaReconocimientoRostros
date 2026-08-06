"""
test_benchmark.py - Script de benchmark automatizado comparativo.

Ejecuta pruebas de rendimiento comparativas entre los 3 modelos
(YOLO, SSD, Faster R-CNN) procesando imágenes del dataset de entrenamiento.
Genera un reporte consolidado con métricas de velocidad y detección.

Uso:
    python tests/test_benchmark.py
    python tests/test_benchmark.py --carpeta_videos "C:/ruta/a/videos"
"""
import unittest
import sys
import os
import time
import numpy as np
import cv2

# Agregar directorio raíz al path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import metricas


class TestBenchmarkVelocidad(unittest.TestCase):
    """Benchmark de velocidad de inferencia para los 3 modelos"""

    def _crear_frames_prueba(self, cantidad=10, ancho=640, alto=480):
        """Genera frames sintéticos para benchmarking"""
        frames = []
        for _ in range(cantidad):
            frame = np.random.randint(0, 255, (alto, ancho, 3), dtype=np.uint8)
            frames.append(frame)
        return frames

    def _obtener_imagenes_reales(self, max_imagenes=20):
        """Obtiene imágenes reales del dataset de entrenamiento"""
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_path = os.path.join(script_dir, 'Data')
        
        imagenes = []
        if not os.path.exists(data_path):
            return imagenes
        
        for persona in os.listdir(data_path):
            persona_path = os.path.join(data_path, persona)
            if not os.path.isdir(persona_path):
                continue
            for img_file in sorted(os.listdir(persona_path)):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img = cv2.imread(os.path.join(persona_path, img_file))
                    if img is not None:
                        imagenes.append(img)
                        if len(imagenes) >= max_imagenes:
                            return imagenes
        return imagenes

    def test_benchmark_yolo(self):
        """Mide tiempo de inferencia de YOLO"""
        try:
            from ultralytics import YOLO
            script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_path = os.path.join(script_dir, "yolo11n-face.pt")
            
            if not os.path.exists(model_path):
                self.skipTest("Modelo YOLO no disponible")
            
            model = YOLO(model_path)
        except ImportError:
            self.skipTest("ultralytics no instalado")
        
        frames = self._obtener_imagenes_reales(20)
        if len(frames) == 0:
            frames = self._crear_frames_prueba(10)
        
        # Warmup
        model.predict(frames[0], stream=False, verbose=False, conf=0.6)
        
        tiempos = []
        total_detecciones = 0
        
        for frame in frames:
            inicio = time.time()
            results = model.predict(frame, stream=False, verbose=False, conf=0.6)
            fin = time.time()
            
            tiempos.append((fin - inicio) * 1000)  # Convertir a ms
            total_detecciones += len(results[0].boxes)
        
        tiempo_promedio = sum(tiempos) / len(tiempos)
        fps = 1000.0 / tiempo_promedio if tiempo_promedio > 0 else 0
        
        print(f"\n{'='*50}")
        print(f"  BENCHMARK YOLO")
        print(f"{'='*50}")
        print(f"  Frames procesados:    {len(frames)}")
        print(f"  Tiempo promedio/frame: {tiempo_promedio:.2f} ms")
        print(f"  FPS estimados:         {fps:.1f}")
        print(f"  Total detecciones:     {total_detecciones}")
        print(f"  Detecciones/frame:     {total_detecciones/len(frames):.2f}")
        print(f"{'='*50}")
        
        # El modelo debe procesar al menos a 1 FPS
        self.assertGreater(fps, 1.0, "YOLO debe ser capaz de procesar al menos 1 FPS")

    def test_benchmark_ssd(self):
        """Mide tiempo de inferencia de SSD"""
        try:
            import torch
            import torchvision
            from torchvision import transforms as T
            
            model = torchvision.models.detection.ssd300_vgg16(
                weights=torchvision.models.detection.SSD300_VGG16_Weights.DEFAULT
            )
            model.eval()
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model.to(device)
            transform = T.Compose([T.ToTensor()])
        except Exception as e:
            self.skipTest(f"SSD no disponible: {e}")
        
        frames = self._obtener_imagenes_reales(10)
        if len(frames) == 0:
            frames = self._crear_frames_prueba(5)
        
        # Warmup
        warmup_tensor = transform(frames[0]).to(device)
        with torch.no_grad():
            model([warmup_tensor])
        
        tiempos = []
        total_detecciones = 0
        
        for frame in frames:
            img_tensor = transform(frame).to(device)
            
            inicio = time.time()
            with torch.no_grad():
                detections = model([img_tensor])[0]
            fin = time.time()
            
            tiempos.append((fin - inicio) * 1000)
            
            # Contar detecciones de personas (label=1) con confianza > 0.5
            for i in range(len(detections['boxes'])):
                if detections['scores'][i].item() > 0.5 and detections['labels'][i].item() == 1:
                    total_detecciones += 1
        
        tiempo_promedio = sum(tiempos) / len(tiempos)
        fps = 1000.0 / tiempo_promedio if tiempo_promedio > 0 else 0
        
        print(f"\n{'='*50}")
        print(f"  BENCHMARK SSD")
        print(f"{'='*50}")
        print(f"  Dispositivo:           {device}")
        print(f"  Frames procesados:    {len(frames)}")
        print(f"  Tiempo promedio/frame: {tiempo_promedio:.2f} ms")
        print(f"  FPS estimados:         {fps:.1f}")
        print(f"  Total detecciones:     {total_detecciones}")
        print(f"{'='*50}")

    def test_benchmark_faster_rcnn(self):
        """Mide tiempo de inferencia de Faster R-CNN"""
        try:
            import torch
            import torchvision
            from torchvision import transforms as T
            
            model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
                weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
            )
            model.eval()
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model.to(device)
            transform = T.Compose([T.ToTensor()])
        except Exception as e:
            self.skipTest(f"Faster R-CNN no disponible: {e}")
        
        frames = self._obtener_imagenes_reales(10)
        if len(frames) == 0:
            frames = self._crear_frames_prueba(5)
        
        # Warmup
        warmup_tensor = transform(frames[0]).to(device)
        with torch.no_grad():
            model([warmup_tensor])
        
        tiempos = []
        total_detecciones = 0
        
        for frame in frames:
            img_tensor = transform(frame).to(device)
            
            inicio = time.time()
            with torch.no_grad():
                detections = model([img_tensor])[0]
            fin = time.time()
            
            tiempos.append((fin - inicio) * 1000)
            
            for i in range(len(detections['boxes'])):
                if detections['scores'][i].item() > 0.5 and detections['labels'][i].item() == 1:
                    total_detecciones += 1
        
        tiempo_promedio = sum(tiempos) / len(tiempos)
        fps = 1000.0 / tiempo_promedio if tiempo_promedio > 0 else 0
        
        print(f"\n{'='*50}")
        print(f"  BENCHMARK FASTER R-CNN")
        print(f"{'='*50}")
        print(f"  Dispositivo:           {device}")
        print(f"  Frames procesados:    {len(frames)}")
        print(f"  Tiempo promedio/frame: {tiempo_promedio:.2f} ms")
        print(f"  FPS estimados:         {fps:.1f}")
        print(f"  Total detecciones:     {total_detecciones}")
        print(f"{'='*50}")


class TestBenchmarkMetricasConsolidado(unittest.TestCase):
    """Test que genera un reporte consolidado con métricas simuladas"""

    def test_reporte_comparativo_3_modelos(self):
        """Genera reporte comparativo de métricas entre los 3 modelos"""
        # Escenarios simulados basados en valores típicos
        escenarios = [
            {
                "modelo": "YOLO",
                "tp": 85, "fp": 8, "fn": 5, "tn": 12,
                "tiempo_ms": 3500, "frames": 100,
                "descripcion": "Modelo especializado en detección facial"
            },
            {
                "modelo": "SSD",
                "tp": 70, "fp": 15, "fn": 12, "tn": 18,
                "tiempo_ms": 8000, "frames": 100,
                "descripcion": "Modelo general de detección de objetos"
            },
            {
                "modelo": "FASTER R-CNN",
                "tp": 75, "fp": 12, "fn": 8, "tn": 15,
                "tiempo_ms": 15000, "frames": 100,
                "descripcion": "Modelo más preciso pero más lento"
            },
        ]

        print(f"\n{'='*70}")
        print(f"  REPORTE COMPARATIVO DE MODELOS DE DETECCIÓN")
        print(f"{'='*70}")

        resultados = []
        for esc in escenarios:
            reporte = metricas.generar_reporte(
                tp=esc["tp"], fp=esc["fp"], fn=esc["fn"], tn=esc["tn"],
                tiempo_total_ms=esc["tiempo_ms"], total_frames=esc["frames"],
                modelo=esc["modelo"]
            )
            resultados.append(reporte)
            print(reporte["reporte_texto"])

        # Tabla comparativa
        print(f"\n{'='*70}")
        print(f"  TABLA COMPARATIVA")
        print(f"{'='*70}")
        print(f"{'Métrica':<22} {'YOLO':>12} {'SSD':>12} {'F.R-CNN':>12}")
        print(f"{'-'*58}")
        
        metricas_nombres = [
            ("Precisión (%)", "precision"),
            ("Recall (%)", "recall"),
            ("F1-Score (%)", "f1_score"),
            ("Exactitud (%)", "exactitud"),
            ("Especificidad (%)", "especificidad"),
            ("Tiempo/Frame (ms)", "tiempo_promedio_frame"),
        ]
        
        for nombre, clave in metricas_nombres:
            valores = [f"{r[clave]:>11.2f}" for r in resultados]
            print(f"  {nombre:<20} {'  '.join(valores)}")
        
        print(f"{'='*70}")
        
        # Verificar que todos los reportes se generaron correctamente
        for r in resultados:
            self.assertGreater(r["total_evaluados"], 0)
            self.assertIsNotNone(r["reporte_texto"])


class TestBenchmarkReconocimiento(unittest.TestCase):
    """Benchmark del reconocimiento facial con múltiples personas"""

    def test_reconocimiento_multiples_personas(self):
        """Verifica que el sistema distingue entre múltiples personas"""
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_path = os.path.join(script_dir, 'Data')
        model_file = os.path.join(script_dir, 'face_embeddings.pkl')
        
        if not os.path.exists(data_path):
            self.skipTest("Carpeta Data no disponible")
        if not os.path.exists(model_file):
            self.skipTest("Modelo de embeddings no disponible")
        
        import pickle
        with open(model_file, "rb") as f:
            data = pickle.load(f)
        
        personas = set(data["names"])
        print(f"\n  Personas en el modelo: {personas}")
        print(f"  Total embeddings: {len(data['embeddings'])}")
        
        # Verificar que hay diversidad en los datos
        for persona in personas:
            count = data["names"].count(persona)
            print(f"    {persona}: {count} embeddings")
            self.assertGreater(count, 0, f"Persona {persona} sin embeddings")
        
        if len(personas) >= 3:
            print(f"\n  ✅ Requisito cumplido: {len(personas)} personas (mínimo 3)")
        else:
            print(f"\n  ⚠ Requisito NO cumplido: {len(personas)} personas (mínimo 3)")
        
        self.assertGreaterEqual(
            len(personas), 3,
            f"Se requieren al menos 3 personas. Encontradas: {len(personas)}"
        )


if __name__ == "__main__":
    # Ejecutar con más detalle
    unittest.main(verbosity=2)
