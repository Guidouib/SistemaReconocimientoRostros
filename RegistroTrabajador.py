from tkinter import *
from tkinter import filedialog, messagebox
import pyodbc

def RegistroPersona(conn):
    global nombre,ap_paterno,ap_materno,dni,direccion,correo,celular 
    """Abre una ventana con el formulario de registro de personas"""
    ventana_registro = Toplevel()
    ventana_registro.title("Registro de Personal")
    ventana_registro.geometry("1060x550")  # Tamaño fijo
    ventana_registro.resizable(False, False)  # No se puede redimensionar
    ventana_registro.configure(bg="#ecf0f1")
    
    ventana_registro.transient() 
    ventana_registro.grab_set()

    frame_titulo = Frame(ventana_registro, bg="#475566")
    frame_titulo.pack(fill=X)
    Label(frame_titulo, text="👤 REGISTRO DE PERSONAL", font=("Arial", 16, "bold"),
          bg="#475566", fg="white", pady=15).pack()
    
    main_container = Frame(ventana_registro, bg="#ecf0f1")
    main_container.pack(fill=BOTH, expand=True, padx=40, pady=20)
    
    frame_form = Frame(main_container, bg="white", relief=FLAT)
    frame_form.pack(fill=BOTH, expand=True, padx=15, pady=10)
    
    frame_interno = Frame(frame_form, bg="white")
    frame_interno.pack(fill=BOTH, expand=True, padx=30, pady=20)
    
    frame_interno.columnconfigure(0, weight=1)
    frame_interno.columnconfigure(1, weight=1)
    frame_interno.columnconfigure(2, weight=1)
    
    def crear_campo(padre, label_text, row, col, es_requerido=True):
        texto_label = f"{label_text} {'*' if es_requerido else ''}"
        Label(padre, text=texto_label, font=("Arial", 10, "bold"),
              bg="white", fg="#2c3e50", anchor=W).grid(
                  row=row, column=col, sticky=W, padx=5, pady=(10, 2))
        
        entry = Entry(padre, font=("Arial", 10), relief=SOLID, bd=1)
        entry.grid(row=row+1, column=col, sticky=EW, padx=5, ipady=6)
        return entry
    
    entry_nombre = crear_campo(frame_interno, "Nombres", 0, 0)
    entry_ap_paterno = crear_campo(frame_interno, "Apellido Paterno", 0, 1)
    entry_ap_materno = crear_campo(frame_interno, "Apellido Materno", 0, 2)
    
    entry_dni = crear_campo(frame_interno, "DNI", 2, 0)
    entry_correo = crear_campo(frame_interno, "Correo Electrónico", 2, 1, es_requerido=False)
    entry_celular = crear_campo(frame_interno, "Número Celular", 2, 2, es_requerido=False)
    
    Label(frame_interno, text="Dirección", font=("Arial", 10, "bold"),
          bg="white", fg="#2c3e50", anchor=W).grid(
              row=4, column=0, columnspan=3, sticky=W, padx=5, pady=(10, 2))
    entry_direccion = Entry(frame_interno, font=("Arial", 10), relief=SOLID, bd=1)
    entry_direccion.grid(row=5, column=0, columnspan=3, sticky=EW, padx=5, ipady=6)
    
    Label(frame_interno, text="Tipo de Trabajador *", font=("Arial", 10, "bold"),
          bg="white", fg="#2c3e50", anchor=W).grid(
              row=6, column=0, columnspan=3, sticky=W, padx=5, pady=(15, 5))
    
    tipo_var = StringVar(master=ventana_registro, value="1")
    ventana_registro.tipo_var = tipo_var 

    frame_radios = Frame(frame_interno, bg="white")
    frame_radios.grid(row=7, column=0, columnspan=3, sticky=W, padx=5, pady=(0, 5))
    
    Radiobutton(frame_radios, text="Ventas", variable=tipo_var, value="1",
                font=("Arial", 9), bg="white").pack(side=LEFT, padx=(0, 15))
    Radiobutton(frame_radios, text="Adiministrador", variable=tipo_var, value="2",
                font=("Arial", 9), bg="white").pack(side=LEFT, padx=(0, 15))
    Radiobutton(frame_radios, text="Doctor(a)", variable=tipo_var, value="3",
                font=("Arial", 9), bg="white").pack(side=LEFT, padx=(0, 15))
    Radiobutton(frame_radios, text="Jefe", variable=tipo_var, value="4",
                font=("Arial", 9), bg="white").pack(side=LEFT)
    
    Label(frame_interno, text="* Campos obligatorios", font=("Arial", 8, "italic"),
          bg="white", fg="#95a5a6").grid(
              row=8, column=0, columnspan=3, sticky=W, padx=5, pady=(8, 5))
    
    lbl_mensaje = Label(frame_interno, text="", font=("Arial", 9, "bold"),
                       bg="white", fg="green")
    lbl_mensaje.grid(row=9, column=0, columnspan=3, pady=8)
    
    def validar_campos():
        """Valida que los campos obligatorios estén completos"""
        nombre = entry_nombre.get().strip()
        ap_paterno = entry_ap_paterno.get().strip()
        ap_materno = entry_ap_materno.get().strip()
        dni = entry_dni.get().strip()
        
        if not nombre or not ap_paterno or not ap_materno or not dni:
            lbl_mensaje.config(text="⚠ Por favor complete todos los campos obligatorios", fg="red")
            return False
        
        if len(dni) < 8:
            lbl_mensaje.config(text="⚠ El DNI debe tener al menos 8 dígitos", fg="red")
            return False
        
        if not dni.isdigit():
            lbl_mensaje.config(text="⚠ El DNI debe contener solo números", fg="red")
            return False
        
        return True
    
    def guardar_registro():
        
        if not validar_campos():
            return
        
        nombre = entry_nombre.get().strip()
        ap_paterno = entry_ap_paterno.get().strip()
        ap_materno = entry_ap_materno.get().strip()
        dni = entry_dni.get().strip()
        direccion = entry_direccion.get().strip()
        correo = entry_correo.get().strip()
        celular = entry_celular.get().strip()
        
        tipo = int(tipo_var.get())
        print(f"Tipo guardado: {tipo}") # Debug

        if conn is None:
            messagebox.showerror("Registro Personal","No se puede conectar a la base de datos.",parent=ventana_registro)
            return
        cursor = conn.cursor()
        sp_call = "EXEC DET_InsertDatosPersonal_SP ?, ?, ?, ?, ?, ?, ?, ?, ?"
        params = (
            nombre,
            ap_paterno,
            ap_materno,
            dni ,
            direccion,
            correo,
            celular,
            tipo,
            1
        )
        try:
            cursor.execute(sp_call, params)
            resultado = cursor.fetchone()

            if resultado:
                MensajeInsercion = str(resultado[0])

                if MensajeInsercion == '1':
                    conn.commit()
                    messagebox.showinfo("Registro Personal","Se inserto Correctamente.",parent=ventana_registro)
                elif MensajeInsercion == '0':
                    conn.rollback()
                    messagebox.showwarning("Registro Personal","Personal ya existe en la Base de Datos.",parent=ventana_registro)
                elif MensajeInsercion == '2':
                    conn.commit()
                    messagebox.showinfo("Registro Personal", "Personal Modificado correctamente.", parent=ventana_registro)
            else:
                conn.rollback()
                messagebox.showerror("Registro Personal","Ocurrio un error al insertar personal.",parent=ventana_registro)
        except pyodbc.Error as ex:
            messagebox.showerror("Registro Personal",f"Error al insertar datos: {ex.args[0]}",parent=ventana_registro)
        finally:
            ventana_registro.after(2000, limpiar_campos)
            cursor.close()
            
    
    def modificar_registro():
        """Modifica un registro existente"""
        if not validar_campos():
            return
        campos_a_controlar = [
        entry_direccion,
        entry_correo,
        entry_celular
        ]

        for campo in campos_a_controlar:
            campo.config(state=NORMAL)
        for radio in frame_radios.winfo_children():
            radio.config(state=NORMAL)
    def leer_datos():
        dni = entry_dni.get().strip()
        if not dni:
            lbl_mensaje.config(text="⚠ Ingrese un DNI para eliminar", fg="red")
            return
        if conn is None:
            messagebox.showerror("Registro Personal","No se puede conectar a la base de datos.",parent=ventana_registro)
            return
        cursor = conn.cursor()
        sp_call = "EXEC DET_LeerDatosPersonal_SP ?"
        params = (
            dni 
        )
        try:
            cursor.execute(sp_call, params)
            datos_retornados = cursor.fetchall()
            mostrarPersonal(datos_retornados)
            conn.commit()
            habilitarCampos()
            messagebox.showinfo("Registro Personal","Datos Leidos Correctamente.",parent=ventana_registro)
        except pyodbc.Error as ex:
            messagebox.showerror("Registro Personal",f"Error al leer los datos: {ex.args[0]}",parent=ventana_registro)
        finally:
            cursor.close()
    def mostrarPersonal(Datos):
        if not Datos:
            lbl_mensaje.config(text="⚠ No se encontró personal con ese DNI.", fg="red")
            return
        limpiar_campos()
        registroPersonal= Datos[0]
        entry_nombre.insert(0, registroPersonal[1])
        entry_ap_paterno.insert(0 , registroPersonal[2])
        entry_ap_materno.insert(0, registroPersonal[3])
        entry_dni.insert(0, registroPersonal[4])
        entry_direccion.insert(0, registroPersonal[5])
        entry_correo.insert(0, registroPersonal[6])
        entry_celular.insert(0, registroPersonal[7])
        tipo_var.set(str(registroPersonal[8]))
        lbl_mensaje.config(text="✅ Datos de personal cargados correctamente.", fg="green")
    def habilitarCampos():
        campos_a_controlar = [
        entry_nombre,
        entry_ap_paterno,
        entry_ap_materno,
        entry_direccion,
        entry_correo,
        entry_celular
        ]
    
        for campo in campos_a_controlar:
            campo.config(state=DISABLED)
        for radio in frame_radios.winfo_children():
            radio.config(state=DISABLED)
    def eliminar_registro():
        dni = entry_dni.get().strip()
        if not dni:
            lbl_mensaje.config(text="⚠ Ingrese un DNI para eliminar", fg="red")
            return
        dni = entry_dni.get().strip()
        if conn is None:
            messagebox.showerror("Registro Personal","Error: No se puede conectar a la base de datos.",parent=ventana_registro)
            return
        cursor = conn.cursor()
        sp_call = "EXEC DET_VigenciaDatosPersonal_SP ?, ?"
        params = (
            dni ,
            0
        )
        try:
            cursor.execute(sp_call, params)
            conn.commit()
            messagebox.showinfo("Registro Personal","Personal dado de baja correctamente en la base de datos.",parent=ventana_registro)
        except pyodbc.Error as ex:
            messagebox.showerror("Registro Personal",f"Error al dar de baja los datos: {ex.args[0]}",parent=ventana_registro)
        finally:
            ventana_registro.after(2000, limpiar_campos)
            cursor.close()
    
    def limpiar_campos():
        """Limpia todos los campos del formulario"""
        entry_nombre.config(state=NORMAL)
        entry_ap_paterno.config(state=NORMAL)
        entry_ap_materno.config(state=NORMAL)
        entry_nombre.delete(0, END)
        entry_ap_paterno.delete(0, END)
        entry_ap_materno.delete(0, END)
        entry_dni.delete(0, END)
        entry_direccion.delete(0, END)
        entry_correo.delete(0, END)
        entry_celular.delete(0, END)
        tipo_var.set("1")
    # ========== FRAME DE BOTONES ==========
    frame_botones = Frame(frame_interno, bg="white")
    frame_botones.grid(row=9, column=0, columnspan=3, pady=15)
    
    # Botón Guardar
    Button(frame_botones, text="💾 Guardar", font=("Arial", 11, "bold"),
           bg="#27ae60", fg="white", relief=FLAT, cursor="hand2",
           command=guardar_registro, width=15).pack(side=LEFT, padx=8, ipady=10)
    
    # Botón Modificar
    Button(frame_botones, text="🔄 Modificar", font=("Arial", 11, "bold"),
           bg="#f39c12", fg="white", relief=FLAT, cursor="hand2",
           command=modificar_registro, width=15).pack(side=LEFT, padx=8, ipady=10)
    
    # Botón Eliminar
    Button(frame_botones, text="🗑️ Eliminar", font=("Arial", 11, "bold"),
           bg="#e74c3c", fg="white", relief=FLAT, cursor="hand2",
           command=eliminar_registro, width=15).pack(side=LEFT, padx=8, ipady=10)
    Button(frame_botones, text="📖 Leer Datos", font=("Arial", 11, "bold"),
           bg="#3498db", fg="white", relief=FLAT, cursor="hand2",
           command=leer_datos, width=15).pack(side=LEFT, padx=8, ipady=10)
    
    # Esperar a que se cierre la ventana
    ventana_registro.wait_window()
