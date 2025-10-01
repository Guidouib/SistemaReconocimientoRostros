import pyodbc

# La cadena de conexión con autenticación de SQL Server
server = 'DESKTOP-0763GQ9\\SQLEXPRESS' 
database = 'ControlAccesoBD'
username = 'sa'
password = '199725'

cnxn_string = (
    f'DRIVER={{ODBC Driver 17 for SQL Server}};'
    f'SERVER={server};'
    f'DATABASE={database};'
    f'UID={username};'
    f'PWD={password};'
)

def get_db_connection():
    """
    Establece y retorna una conexión a la base de datos SQL Server.
    Retorna None en caso de error.
    """
    conn = None
    try:
        conn = pyodbc.connect(cnxn_string)
        print("Conexión a la base de datos exitosa.")
        return conn
    except pyodbc.Error as ex:
        sqlstate = ex.args[0]
        print(f"Error al conectar a la base de datos: {sqlstate}")
        return None
