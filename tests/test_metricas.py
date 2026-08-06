"""
test_metricas.py - Tests unitarios para el módulo de métricas.

Verifica que todas las fórmulas de métricas de rendimiento se calculan
correctamente, incluyendo casos normales, extremos y divisiones por cero.
"""
import unittest
import sys
import os

# Agregar directorio raíz al path para importar módulos del proyecto
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import metricas


class TestPrecision(unittest.TestCase):
    """Tests para el cálculo de Precisión = TP / (TP + FP)"""

    def test_precision_caso_normal(self):
        """Precisión con valores típicos: 80/(80+20) = 80%"""
        resultado = metricas.calcular_precision(tp=80, fp=20)
        self.assertAlmostEqual(resultado, 80.0, places=2)

    def test_precision_perfecta(self):
        """Precisión perfecta: sin falsos positivos → 100%"""
        resultado = metricas.calcular_precision(tp=50, fp=0)
        self.assertAlmostEqual(resultado, 100.0, places=2)

    def test_precision_cero_tp(self):
        """Sin verdaderos positivos pero con falsos positivos → 0%"""
        resultado = metricas.calcular_precision(tp=0, fp=10)
        self.assertAlmostEqual(resultado, 0.0, places=2)

    def test_precision_division_por_cero(self):
        """Sin datos (TP=0, FP=0) → debe retornar 0% sin error"""
        resultado = metricas.calcular_precision(tp=0, fp=0)
        self.assertAlmostEqual(resultado, 0.0, places=2)

    def test_precision_valores_grandes(self):
        """Verificar con valores grandes: 900/(900+100) = 90%"""
        resultado = metricas.calcular_precision(tp=900, fp=100)
        self.assertAlmostEqual(resultado, 90.0, places=2)


class TestRecall(unittest.TestCase):
    """Tests para el cálculo de Recall = TP / (TP + FN)"""

    def test_recall_caso_normal(self):
        """Recall con valores típicos: 70/(70+30) = 70%"""
        resultado = metricas.calcular_recall(tp=70, fn=30)
        self.assertAlmostEqual(resultado, 70.0, places=2)

    def test_recall_perfecto(self):
        """Recall perfecto: sin falsos negativos → 100%"""
        resultado = metricas.calcular_recall(tp=50, fn=0)
        self.assertAlmostEqual(resultado, 100.0, places=2)

    def test_recall_cero(self):
        """Sin detecciones correctas → 0%"""
        resultado = metricas.calcular_recall(tp=0, fn=20)
        self.assertAlmostEqual(resultado, 0.0, places=2)

    def test_recall_division_por_cero(self):
        """Sin datos (TP=0, FN=0) → debe retornar 0% sin error"""
        resultado = metricas.calcular_recall(tp=0, fn=0)
        self.assertAlmostEqual(resultado, 0.0, places=2)


class TestF1Score(unittest.TestCase):
    """Tests para F1-Score = 2 × (Precisión × Recall) / (Precisión + Recall)"""

    def test_f1_caso_normal(self):
        """F1 con Precisión=80%, Recall=60% → 2*(80*60)/(80+60) ≈ 68.57%"""
        resultado = metricas.calcular_f1_score(precision=80.0, recall=60.0)
        self.assertAlmostEqual(resultado, 68.57, places=1)

    def test_f1_precision_y_recall_iguales(self):
        """Si Precisión == Recall, F1 debe ser igual a ambos"""
        resultado = metricas.calcular_f1_score(precision=75.0, recall=75.0)
        self.assertAlmostEqual(resultado, 75.0, places=2)

    def test_f1_perfecto(self):
        """Precisión y Recall perfectos → F1 = 100%"""
        resultado = metricas.calcular_f1_score(precision=100.0, recall=100.0)
        self.assertAlmostEqual(resultado, 100.0, places=2)

    def test_f1_precision_cero(self):
        """Precisión 0 → F1 = 0 (independiente del Recall)"""
        resultado = metricas.calcular_f1_score(precision=0.0, recall=80.0)
        self.assertAlmostEqual(resultado, 0.0, places=2)

    def test_f1_recall_cero(self):
        """Recall 0 → F1 = 0 (independiente de la Precisión)"""
        resultado = metricas.calcular_f1_score(precision=80.0, recall=0.0)
        self.assertAlmostEqual(resultado, 0.0, places=2)

    def test_f1_division_por_cero(self):
        """Ambos en 0 → debe retornar 0% sin error"""
        resultado = metricas.calcular_f1_score(precision=0.0, recall=0.0)
        self.assertAlmostEqual(resultado, 0.0, places=2)


class TestExactitud(unittest.TestCase):
    """Tests para Exactitud = (TP + TN) / Total"""

    def test_exactitud_caso_normal(self):
        """Exactitud: (80+10)/100 = 90%"""
        resultado = metricas.calcular_exactitud(tp=80, tn=10, total=100)
        self.assertAlmostEqual(resultado, 90.0, places=2)

    def test_exactitud_perfecta(self):
        """Todo correcto → 100%"""
        resultado = metricas.calcular_exactitud(tp=60, tn=40, total=100)
        self.assertAlmostEqual(resultado, 100.0, places=2)

    def test_exactitud_cero(self):
        """Todo incorrecto → 0%"""
        resultado = metricas.calcular_exactitud(tp=0, tn=0, total=100)
        self.assertAlmostEqual(resultado, 0.0, places=2)

    def test_exactitud_total_cero(self):
        """Sin datos → debe retornar 0% sin error"""
        resultado = metricas.calcular_exactitud(tp=0, tn=0, total=0)
        self.assertAlmostEqual(resultado, 0.0, places=2)


class TestEspecificidad(unittest.TestCase):
    """Tests para Especificidad = TN / (TN + FP)"""

    def test_especificidad_caso_normal(self):
        """Especificidad: 40/(40+10) = 80%"""
        resultado = metricas.calcular_especificidad(tn=40, fp=10)
        self.assertAlmostEqual(resultado, 80.0, places=2)

    def test_especificidad_perfecta(self):
        """Sin falsos positivos → 100%"""
        resultado = metricas.calcular_especificidad(tn=50, fp=0)
        self.assertAlmostEqual(resultado, 100.0, places=2)

    def test_especificidad_cero(self):
        """Sin verdaderos negativos → 0%"""
        resultado = metricas.calcular_especificidad(tn=0, fp=20)
        self.assertAlmostEqual(resultado, 0.0, places=2)

    def test_especificidad_division_por_cero(self):
        """Sin datos → debe retornar 0% sin error"""
        resultado = metricas.calcular_especificidad(tn=0, fp=0)
        self.assertAlmostEqual(resultado, 0.0, places=2)


class TestTiempoPromedioFrame(unittest.TestCase):
    """Tests para el cálculo de tiempo promedio por frame"""

    def test_tiempo_promedio_normal(self):
        """1000ms / 100 frames = 10ms por frame"""
        resultado = metricas.calcular_tiempo_promedio_frame(1000.0, 100)
        self.assertAlmostEqual(resultado, 10.0, places=2)

    def test_tiempo_promedio_cero_frames(self):
        """Sin frames procesados → 0 sin error"""
        resultado = metricas.calcular_tiempo_promedio_frame(5000.0, 0)
        self.assertAlmostEqual(resultado, 0.0, places=2)

    def test_tiempo_promedio_un_frame(self):
        """Un solo frame → tiempo total = tiempo por frame"""
        resultado = metricas.calcular_tiempo_promedio_frame(45.5, 1)
        self.assertAlmostEqual(resultado, 45.5, places=2)


class TestGenerarReporte(unittest.TestCase):
    """Tests para la generación de reportes consolidados"""

    def test_reporte_estructura_completa(self):
        """Verifica que el reporte contiene todas las métricas"""
        resultado = metricas.generar_reporte(
            tp=80, fp=10, fn=5, tn=5,
            tiempo_total_ms=5000.0, total_frames=100,
            modelo="YOLO", video="test.mp4"
        )

        # Verificar que es un diccionario con todas las claves
        self.assertIn("precision", resultado)
        self.assertIn("recall", resultado)
        self.assertIn("f1_score", resultado)
        self.assertIn("exactitud", resultado)
        self.assertIn("especificidad", resultado)
        self.assertIn("tiempo_promedio_frame", resultado)
        self.assertIn("reporte_texto", resultado)
        self.assertIn("total_evaluados", resultado)

    def test_reporte_metricas_correctas(self):
        """Verifica que las métricas dentro del reporte son correctas"""
        resultado = metricas.generar_reporte(
            tp=80, fp=20, fn=10, tn=10
        )

        # Precisión = 80/(80+20) = 80%
        self.assertAlmostEqual(resultado["precision"], 80.0, places=2)
        # Recall = 80/(80+10) ≈ 88.89%
        self.assertAlmostEqual(resultado["recall"], 88.89, places=1)
        # Total evaluados = 80+20+10+10 = 120
        self.assertEqual(resultado["total_evaluados"], 120)

    def test_reporte_sin_datos(self):
        """Reporte con todo en 0 → no debe lanzar errores"""
        resultado = metricas.generar_reporte(
            tp=0, fp=0, fn=0, tn=0
        )
        self.assertAlmostEqual(resultado["precision"], 0.0)
        self.assertAlmostEqual(resultado["recall"], 0.0)
        self.assertAlmostEqual(resultado["f1_score"], 0.0)
        self.assertAlmostEqual(resultado["exactitud"], 0.0)
        self.assertAlmostEqual(resultado["especificidad"], 0.0)

    def test_reporte_texto_contiene_info(self):
        """Verifica que el texto del reporte contiene información clave"""
        resultado = metricas.generar_reporte(
            tp=50, fp=5, fn=3, tn=2,
            modelo="SSD", video="video_prueba.mp4"
        )
        texto = resultado["reporte_texto"]
        self.assertIn("SSD", texto)
        self.assertIn("video_prueba.mp4", texto)
        self.assertIn("Precisión", texto)
        self.assertIn("Recall", texto)
        self.assertIn("F1-Score", texto)
        self.assertIn("Exactitud", texto)
        self.assertIn("Especificidad", texto)


class TestMetricasIntegracion(unittest.TestCase):
    """Tests de integración: escenarios realistas completos"""

    def test_escenario_modelo_perfecto(self):
        """Modelo que clasifica todo correctamente"""
        resultado = metricas.generar_reporte(tp=50, fp=0, fn=0, tn=50)
        self.assertAlmostEqual(resultado["precision"], 100.0)
        self.assertAlmostEqual(resultado["recall"], 100.0)
        self.assertAlmostEqual(resultado["f1_score"], 100.0)
        self.assertAlmostEqual(resultado["exactitud"], 100.0)
        self.assertAlmostEqual(resultado["especificidad"], 100.0)

    def test_escenario_modelo_pesimo(self):
        """Modelo que clasifica todo incorrectamente"""
        resultado = metricas.generar_reporte(tp=0, fp=50, fn=50, tn=0)
        self.assertAlmostEqual(resultado["precision"], 0.0)
        self.assertAlmostEqual(resultado["recall"], 0.0)
        self.assertAlmostEqual(resultado["f1_score"], 0.0)
        self.assertAlmostEqual(resultado["exactitud"], 0.0)
        self.assertAlmostEqual(resultado["especificidad"], 0.0)

    def test_escenario_alta_precision_bajo_recall(self):
        """Modelo conservador: pocas detecciones pero precisas"""
        resultado = metricas.generar_reporte(tp=10, fp=1, fn=40, tn=49)
        # Precisión alta: 10/11 ≈ 90.91%
        self.assertGreater(resultado["precision"], 90.0)
        # Recall bajo: 10/50 = 20%
        self.assertLess(resultado["recall"], 25.0)

    def test_escenario_bajo_precision_alto_recall(self):
        """Modelo agresivo: detecta mucho pero con muchos falsos positivos"""
        resultado = metricas.generar_reporte(tp=45, fp=40, fn=5, tn=10)
        # Recall alto: 45/50 = 90%
        self.assertGreater(resultado["recall"], 85.0)
        # Precisión baja: 45/85 ≈ 52.94%
        self.assertLess(resultado["precision"], 60.0)

    def test_consistencia_f1_entre_precision_y_recall(self):
        """F1 siempre debe estar entre Precisión y Recall (o igual)"""
        resultado = metricas.generar_reporte(tp=30, fp=10, fn=20, tn=40)
        p = resultado["precision"]
        r = resultado["recall"]
        f1 = resultado["f1_score"]
        self.assertGreaterEqual(f1, min(p, r) - 0.01)
        self.assertLessEqual(f1, max(p, r) + 0.01)


if __name__ == "__main__":
    unittest.main(verbosity=2)
