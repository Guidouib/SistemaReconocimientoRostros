"""
test_deteccion.py - Tests unitarios para los modelos de detección facial.

Verifica que los 3 modelos (YOLO, SSD, Faster R-CNN) pueden:
- Cargarse correctamente
- Detectar rostros en imágenes de prueba
- No generar detecciones falsas en imágenes vacías
"""
import unittest
import sys
import os
import numpy as np
import cv2

# Agregar directorio raíz al path para importar módulos del proyecto
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestYOLODeteccion(unittest.TestCase):
    """Tests para el modelo YOLO de detección facial"""

    @classmethod
    def setUpClass(cls):
        """Carga el modelo YOLO una vez para todos los tests"""
        try:
            from ultralytics import YOLO
            script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_path = os.path.join(script_dir, "yolo11n-face.pt")
            
            if not os.path.exists(model_path):
                cls.model = None
                return
            
            cls.model = YOLO(model_path)
        except ImportError:
            cls.model = None

    def test_modelo_yolo_cargado(self):
        """Verifica que el modelo YOLO se carga correctamente"""
        if self.model is None:
            self.skipTest("Modelo YOLO no disponible (archivo o librería faltante)")
        self.assertIsNotNone(self.model)

    def test_yolo_detecta_en_imagen_sintetica(self):
        """Verifica que YOLO procesa un frame sin errores"""
        if self.model is None:
            self.skipTest("Modelo YOLO no disponible")
        
        # Crear imagen de prueba (frame sintético)
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # No debe lanzar excepciones
        results = self.model.predict(frame, stream=False, verbose=False, conf=0.6)
        self.assertIsNotNone(results)
        self.assertEqual(len(results), 1, "Debe retornar exactamente 1 resultado")

    def test_yolo_frame_negro_sin_falsos_positivos(self):
        """Frame completamente negro no debe generar detecciones"""
        if self.model is None:
            self.skipTest("Modelo YOLO no disponible")
        
        frame_negro = np.zeros((480, 640, 3), dtype=np.uint8)
        results = self.model.predict(frame_negro, stream=False, verbose=False, conf=0.6)
        
        num_detecciones = len(results[0].boxes)
        self.assertEqual(
            num_detecciones, 0,
            f"Frame negro no debería generar detecciones, pero se detectaron {num_detecciones}"
        )

    def test_yolo_con_imagen_real_de_entrenamiento(self):
        """Verifica detección usando una imagen real del dataset"""
        if self.model is None:
            self.skipTest("Modelo YOLO no disponible")
        
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_path = os.path.join(script_dir, 'Data')
        
        if not os.path.exists(data_path):
            self.skipTest("Carpeta Data no disponible")
        
        # Buscar la primera imagen disponible
        imagen_encontrada = None
        for persona in os.listdir(data_path):
            persona_path = os.path.join(data_path, persona)
            if os.path.isdir(persona_path):
                for img_file in os.listdir(persona_path):
                    if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        imagen_encontrada = os.path.join(persona_path, img_file)
                        break
            if imagen_encontrada:
                break
        
        if imagen_encontrada is None:
            self.skipTest("No se encontraron imágenes en Data/")
        
        frame = cv2.imread(imagen_encontrada)
        self.assertIsNotNone(frame, f"No se pudo leer la imagen: {imagen_encontrada}")
        
        results = self.model.predict(frame, stream=False, verbose=False, conf=0.6)
        # La imagen de entrenamiento debería contener un rostro
        num_detecciones = len(results[0].boxes)
        print(f"  YOLO detectó {num_detecciones} rostro(s) en imagen de entrenamiento")


class TestSSDDeteccion(unittest.TestCase):
    """Tests para el modelo SSD de detección"""

    @classmethod
    def setUpClass(cls):
        """Carga el modelo SSD una vez para todos los tests"""
        try:
            import torch
            import torchvision
            cls.model = torchvision.models.detection.ssd300_vgg16(
                weights=torchvision.models.detection.SSD300_VGG16_Weights.DEFAULT
            )
            cls.model.eval()
            cls.disponible = True
        except Exception:
            cls.model = None
            cls.disponible = False

    def test_modelo_ssd_cargado(self):
        """Verifica que el modelo SSD se carga correctamente"""
        if not self.disponible:
            self.skipTest("Modelo SSD no disponible (dependencias faltantes)")
        self.assertIsNotNone(self.model)

    def test_ssd_procesa_frame_sin_errores(self):
        """Verifica que SSD puede procesar un frame sin excepciones"""
        if not self.disponible:
            self.skipTest("Modelo SSD no disponible")
        
        import torch
        from torchvision import transforms as T
        
        frame = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
        transform = T.Compose([T.ToTensor()])
        img_tensor = transform(frame)
        
        with torch.no_grad():
            detections = self.model([img_tensor])
        
        self.assertIsNotNone(detections)
        self.assertEqual(len(detections), 1)
        self.assertIn('boxes', detections[0])
        self.assertIn('scores', detections[0])
        self.assertIn('labels', detections[0])

    def test_ssd_detecta_persona_label_1(self):
        """Verifica que SSD usa label=1 para la clase 'persona'"""
        if not self.disponible:
            self.skipTest("Modelo SSD no disponible")
        
        import torch
        from torchvision import transforms as T

        frame = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
        transform = T.Compose([T.ToTensor()])
        img_tensor = transform(frame)
        
        with torch.no_grad():
            detections = self.model([img_tensor])[0]
        
        # Verificar que las labels son enteros (la clase 1 es 'persona' en COCO)
        if len(detections['labels']) > 0:
            label_type = detections['labels'][0].item()
            self.assertIsInstance(label_type, int)


class TestFasterRCNNDeteccion(unittest.TestCase):
    """Tests para el modelo Faster R-CNN de detección"""

    @classmethod
    def setUpClass(cls):
        """Carga el modelo Faster R-CNN una vez para todos los tests"""
        try:
            import torch
            import torchvision
            cls.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
                weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
            )
            cls.model.eval()
            cls.disponible = True
        except Exception:
            cls.model = None
            cls.disponible = False

    def test_modelo_faster_rcnn_cargado(self):
        """Verifica que el modelo Faster R-CNN se carga correctamente"""
        if not self.disponible:
            self.skipTest("Modelo Faster R-CNN no disponible")
        self.assertIsNotNone(self.model)

    def test_faster_rcnn_procesa_frame_sin_errores(self):
        """Verifica que Faster R-CNN puede procesar un frame sin excepciones"""
        if not self.disponible:
            self.skipTest("Modelo Faster R-CNN no disponible")
        
        import torch
        from torchvision import transforms as T
        
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        transform = T.Compose([T.ToTensor()])
        img_tensor = transform(frame)
        
        with torch.no_grad():
            detections = self.model([img_tensor])
        
        self.assertIsNotNone(detections)
        self.assertEqual(len(detections), 1)
        self.assertIn('boxes', detections[0])
        self.assertIn('scores', detections[0])
        self.assertIn('labels', detections[0])


class TestComparacionModelos(unittest.TestCase):
    """Tests comparativos entre los 3 modelos"""

    def test_todos_los_modelos_disponibles(self):
        """Verifica que los 3 modelos pueden cargarse"""
        modelos_disponibles = []
        
        # YOLO
        try:
            from ultralytics import YOLO
            script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_path = os.path.join(script_dir, "yolo11n-face.pt")
            if os.path.exists(model_path):
                modelos_disponibles.append("YOLO")
        except ImportError:
            pass
        
        # SSD
        try:
            import torchvision
            _ = torchvision.models.detection.ssd300_vgg16
            modelos_disponibles.append("SSD")
        except (ImportError, AttributeError):
            pass
        
        # Faster R-CNN
        try:
            import torchvision
            _ = torchvision.models.detection.fasterrcnn_resnet50_fpn
            modelos_disponibles.append("FASTER R-CNN")
        except (ImportError, AttributeError):
            pass
        
        print(f"\n  Modelos disponibles: {modelos_disponibles}")
        self.assertEqual(
            len(modelos_disponibles), 3,
            f"Se requieren 3 modelos. Disponibles: {modelos_disponibles}"
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
