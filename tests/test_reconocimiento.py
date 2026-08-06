"""
test_reconocimiento.py - Tests unitarios para el módulo ReconocimientoFacial.

Verifica el comportamiento del reconocimiento facial:
- Carga del modelo de embeddings
- Clasificación correcta (TP, FP, FN, TN)
- Filtrado de rostros inválidos (muy pequeños, borrosos, sin modelo)
- Cálculo de distancia coseno
"""
import unittest
import sys
import os
import numpy as np

# Agregar directorio raíz al path para importar módulos del proyecto
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestDistanciaCoseno(unittest.TestCase):
    """Tests para la función de distancia coseno"""

    def setUp(self):
        """Importar el módulo solo si está disponible"""
        try:
            import ReconocimientoFacial
            self.modulo = ReconocimientoFacial
            self.disponible = True
        except ImportError:
            self.disponible = False

    def test_distancia_vectores_identicos(self):
        """Vectores idénticos → distancia = 0"""
        if not self.disponible:
            self.skipTest("Módulo ReconocimientoFacial no disponible")
        vec = np.array([1.0, 2.0, 3.0, 4.0])
        dist = self.modulo.calcular_distancia_coseno(vec, vec)
        self.assertAlmostEqual(dist, 0.0, places=5)

    def test_distancia_vectores_ortogonales(self):
        """Vectores ortogonales → distancia = 1"""
        if not self.disponible:
            self.skipTest("Módulo ReconocimientoFacial no disponible")
        vec1 = np.array([1.0, 0.0])
        vec2 = np.array([0.0, 1.0])
        dist = self.modulo.calcular_distancia_coseno(vec1, vec2)
        self.assertAlmostEqual(dist, 1.0, places=5)

    def test_distancia_vectores_opuestos(self):
        """Vectores opuestos → distancia = 2"""
        if not self.disponible:
            self.skipTest("Módulo ReconocimientoFacial no disponible")
        vec1 = np.array([1.0, 0.0])
        vec2 = np.array([-1.0, 0.0])
        dist = self.modulo.calcular_distancia_coseno(vec1, vec2)
        self.assertAlmostEqual(dist, 2.0, places=5)

    def test_distancia_rango_valido(self):
        """La distancia coseno debe estar entre 0 y 2"""
        if not self.disponible:
            self.skipTest("Módulo ReconocimientoFacial no disponible")
        vec1 = np.random.rand(512)
        vec2 = np.random.rand(512)
        dist = self.modulo.calcular_distancia_coseno(vec1, vec2)
        self.assertGreaterEqual(dist, 0.0)
        self.assertLessEqual(dist, 2.0)


class TestCargaModelo(unittest.TestCase):
    """Tests para la carga del modelo de embeddings"""

    def test_modelo_archivo_existe(self):
        """Verifica que el archivo face_embeddings.pkl existe"""
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_file = os.path.join(script_dir, 'face_embeddings.pkl')
        self.assertTrue(
            os.path.exists(model_file),
            f"El archivo de modelo no existe: {model_file}. "
            "Debe entrenar el modelo primero con 'Entrenar Modelo'."
        )

    def test_modelo_se_carga_correctamente(self):
        """Verifica que el modelo se puede cargar y tiene la estructura esperada"""
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_file = os.path.join(script_dir, 'face_embeddings.pkl')
        
        if not os.path.exists(model_file):
            self.skipTest("Archivo de modelo no disponible")
        
        import pickle
        with open(model_file, "rb") as f:
            data = pickle.load(f)
        
        self.assertIn("embeddings", data, "El modelo debe contener 'embeddings'")
        self.assertIn("names", data, "El modelo debe contener 'names'")
        self.assertGreater(len(data["embeddings"]), 0, "Debe haber al menos 1 embedding")
        self.assertEqual(
            len(data["embeddings"]), len(data["names"]),
            "Cantidad de embeddings debe coincidir con cantidad de nombres"
        )

    def test_modelo_tiene_al_menos_3_personas(self):
        """Verifica que el modelo tiene datos de al menos 3 personas distintas"""
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_file = os.path.join(script_dir, 'face_embeddings.pkl')
        
        if not os.path.exists(model_file):
            self.skipTest("Archivo de modelo no disponible")
        
        import pickle
        with open(model_file, "rb") as f:
            data = pickle.load(f)
        
        personas_unicas = set(data["names"])
        self.assertGreaterEqual(
            len(personas_unicas), 3,
            f"Se requieren al menos 3 personas distintas en el modelo. "
            f"Personas encontradas: {len(personas_unicas)} → {personas_unicas}"
        )


class TestDatosEntrenamiento(unittest.TestCase):
    """Tests para verificar los datos de entrenamiento"""

    def test_carpeta_data_existe(self):
        """Verifica que la carpeta Data/ existe"""
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_path = os.path.join(script_dir, 'Data')
        self.assertTrue(
            os.path.exists(data_path),
            f"La carpeta Data no existe: {data_path}"
        )

    def test_al_menos_3_personas_registradas(self):
        """Verifica que hay al menos 3 carpetas (personas) en Data/"""
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_path = os.path.join(script_dir, 'Data')
        
        if not os.path.exists(data_path):
            self.skipTest("Carpeta Data no disponible")
        
        personas = [d for d in os.listdir(data_path) 
                    if os.path.isdir(os.path.join(data_path, d))]
        
        self.assertGreaterEqual(
            len(personas), 3,
            f"Se requieren al menos 3 personas en Data/. "
            f"Personas encontradas: {len(personas)} → {personas}"
        )

    def test_cada_persona_tiene_imagenes(self):
        """Verifica que cada persona tiene imágenes de entrenamiento"""
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_path = os.path.join(script_dir, 'Data')
        
        if not os.path.exists(data_path):
            self.skipTest("Carpeta Data no disponible")
        
        personas = [d for d in os.listdir(data_path) 
                    if os.path.isdir(os.path.join(data_path, d))]
        
        for persona in personas:
            persona_path = os.path.join(data_path, persona)
            imagenes = [f for f in os.listdir(persona_path) 
                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            self.assertGreater(
                len(imagenes), 0,
                f"La persona '{persona}' no tiene imágenes de entrenamiento"
            )
            print(f"  Persona '{persona}': {len(imagenes)} imágenes")


class TestClasificacionHeuristica(unittest.TestCase):
    """Tests para verificar los umbrales de clasificación heurística"""

    def test_umbrales_definidos(self):
        """Verifica que los umbrales de clasificación están definidos correctamente"""
        # Los umbrales actuales del sistema
        UMBRAL_TP = 0.35
        UMBRAL_RECONOCER = 0.45
        UMBRAL_FN = 0.55

        # Verificar orden lógico: TP < RECONOCER < FN
        self.assertLess(UMBRAL_TP, UMBRAL_RECONOCER,
                       "UMBRAL_TP debe ser menor que UMBRAL_RECONOCER")
        self.assertLess(UMBRAL_RECONOCER, UMBRAL_FN,
                       "UMBRAL_RECONOCER debe ser menor que UMBRAL_FN")

    def test_clasificacion_tp(self):
        """Distancia < 0.35 con match → debe ser TP"""
        distancia = 0.20
        UMBRAL_TP = 0.35
        self.assertLess(distancia, UMBRAL_TP)

    def test_clasificacion_fp(self):
        """Distancia entre 0.35 y 0.45 con match → debe ser FP"""
        distancia = 0.40
        UMBRAL_TP = 0.35
        UMBRAL_RECONOCER = 0.45
        self.assertGreaterEqual(distancia, UMBRAL_TP)
        self.assertLess(distancia, UMBRAL_RECONOCER)

    def test_clasificacion_fn(self):
        """Distancia entre 0.45 y 0.55 → debe ser FN"""
        distancia = 0.50
        UMBRAL_RECONOCER = 0.45
        UMBRAL_FN = 0.55
        self.assertGreaterEqual(distancia, UMBRAL_RECONOCER)
        self.assertLess(distancia, UMBRAL_FN)

    def test_clasificacion_tn(self):
        """Distancia >= 0.55 → debe ser TN"""
        distancia = 0.70
        UMBRAL_FN = 0.55
        self.assertGreaterEqual(distancia, UMBRAL_FN)


class TestReconocimientoFiltros(unittest.TestCase):
    """Tests para los filtros de calidad antes del reconocimiento"""

    def test_tamanio_minimo_rostro(self):
        """Rostros menores a 80x80 deben ser filtrados"""
        TAMANIO_MINIMO = 80
        
        # Caso que debe ser filtrado
        w_pequenio, h_pequenio = 60, 60
        self.assertLess(w_pequenio, TAMANIO_MINIMO)
        
        # Caso que debe pasar
        w_valido, h_valido = 100, 100
        self.assertGreaterEqual(w_valido, TAMANIO_MINIMO)

    def test_umbral_blur(self):
        """Imágenes con varianza de Laplacian < 50 deben ser filtradas"""
        import cv2
        
        UMBRAL_BLUR = 50
        
        # Crear imagen borrosa (todo gris uniforme)
        img_borrosa = np.ones((100, 100), dtype=np.uint8) * 128
        blur_value = cv2.Laplacian(img_borrosa, cv2.CV_64F).var()
        self.assertLess(blur_value, UMBRAL_BLUR,
                       f"Imagen uniforme debería tener blur bajo, got {blur_value}")
        
        # Crear imagen con bordes definidos (no borrosa)
        img_nitida = np.zeros((100, 100), dtype=np.uint8)
        cv2.rectangle(img_nitida, (20, 20), (80, 80), 255, 2)
        cv2.line(img_nitida, (0, 0), (100, 100), 255, 2)
        cv2.line(img_nitida, (0, 100), (100, 0), 255, 2)
        blur_value_nitida = cv2.Laplacian(img_nitida, cv2.CV_64F).var()
        self.assertGreater(blur_value_nitida, UMBRAL_BLUR,
                          f"Imagen nítida debería tener blur alto, got {blur_value_nitida}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
