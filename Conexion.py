import pyodbc

server = r'GUIDO\SQLEXPRESS01' 
database = 'ControlAccesoBD'

cnxn_string = (
    f'DRIVER={{ODBC Driver 17 for SQL Server}};'
    f'SERVER={server};'
    f'DATABASE={database};'
    f'Trusted_Connection=yes;'
)

def get_db_connection():
    conn = None
    try:
        conn = pyodbc.connect(cnxn_string)
        print("Conexión a la base de datos exitosa.")
        return conn
    except pyodbc.Error as ex:
        print(f"ERROR COMPLETO AL CONECTAR: {ex}") 
        return None
