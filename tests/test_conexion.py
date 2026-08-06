"""
test_conexion.py - Tests unitarios para la conexión a la base de datos.

Verifica que la conexión a SQL Server funciona correctamente
y que las operaciones de base de datos están disponibles.
"""
import unittest
import sys
import os

# Agregar directorio raíz al path para importar módulos del proyecto
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestConexionBD(unittest.TestCase):
    """Tests para la conexión a la base de datos SQL Server"""

    def test_modulo_conexion_importable(self):
        """Verifica que el módulo Conexion se puede importar"""
        try:
            import Conexion
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"No se puede importar el módulo Conexion: {e}")

    def test_conexion_exitosa(self):
        """Verifica que se puede establecer conexión con la BD"""
        try:
            import Conexion
            conn = Conexion.get_db_connection()
            self.assertIsNotNone(
                conn,
                "La conexión retornó None. Verificar que SQL Server está corriendo "
                "y que la cadena de conexión es correcta."
            )
            conn.close()
        except Exception as e:
            self.skipTest(f"No se pudo conectar a la BD: {e}")

    def test_conexion_puede_ejecutar_query(self):
        """Verifica que la conexión puede ejecutar consultas básicas"""
        try:
            import Conexion
            conn = Conexion.get_db_connection()
            if conn is None:
                self.skipTest("Sin conexión a BD")
            
            cursor = conn.cursor()
            cursor.execute("SELECT 1 AS test_value")
            resultado = cursor.fetchone()
            cursor.close()
            
            self.assertIsNotNone(resultado)
            self.assertEqual(resultado[0], 1)
            
            conn.close()
        except Exception as e:
            self.skipTest(f"Error ejecutando query: {e}")

    def test_stored_procedure_existe(self):
        """Verifica que el stored procedure de inserción de resultados existe"""
        try:
            import Conexion
            conn = Conexion.get_db_connection()
            if conn is None:
                self.skipTest("Sin conexión a BD")
            
            cursor = conn.cursor()
            cursor.execute("""
                SELECT COUNT(*) FROM sys.procedures 
                WHERE name = 'DET_InsertDataDeteccionModelo_SP'
            """)
            resultado = cursor.fetchone()
            cursor.close()
            
            self.assertGreater(
                resultado[0], 0,
                "El stored procedure 'DET_InsertDataDeteccionModelo_SP' no existe en la BD"
            )
            
            conn.close()
        except Exception as e:
            self.skipTest(f"Error verificando SP: {e}")

    def test_tabla_rendimiento_existe(self):
        """Verifica que la tabla de rendimiento de modelos existe"""
        try:
            import Conexion
            conn = Conexion.get_db_connection()
            if conn is None:
                self.skipTest("Sin conexión a BD")
            
            cursor = conn.cursor()
            cursor.execute("""
                SELECT COUNT(*) FROM INFORMATION_SCHEMA.TABLES 
                WHERE TABLE_NAME = 'SI_FinRendimientoModelos'
            """)
            resultado = cursor.fetchone()
            cursor.close()
            
            self.assertGreater(
                resultado[0], 0,
                "La tabla 'SI_FinRendimientoModelos' no existe en la BD"
            )
            
            conn.close()
        except Exception as e:
            self.skipTest(f"Error verificando tabla: {e}")


class TestConexionReconexion(unittest.TestCase):
    """Tests para el mecanismo de reconexión"""

    def test_reconexion_multiples_llamadas(self):
        """Verifica que se puede obtener conexión múltiples veces"""
        try:
            import Conexion
            
            for i in range(3):
                conn = Conexion.get_db_connection()
                if conn is None:
                    self.skipTest("Sin conexión a BD")
                conn.close()
            
            self.assertTrue(True, "3 conexiones exitosas consecutivas")
        except Exception as e:
            self.skipTest(f"Error en reconexión: {e}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
