"""
metricas.py - Módulo centralizado de cálculo de métricas de rendimiento.

Proporciona funciones para calcular Precisión, Recall, F1-Score,
Exactitud, Especificidad y generar reportes consolidados.
Utilizado tanto por la aplicación principal como por los tests.
"""


def calcular_precision(tp, fp):
    """
    Precisión = TP / (TP + FP)
    Mide qué proporción de las detecciones positivas fueron correctas.
    
    Args:
        tp (int): Verdaderos Positivos
        fp (int): Falsos Positivos
    Returns:
        float: Precisión en porcentaje (0-100), o 0 si no hay datos.
    """
    denominador = tp + fp
    if denominador == 0:
        return 0.0
    return (tp / denominador) * 100


def calcular_recall(tp, fn):
    """
    Recall (Sensibilidad) = TP / (TP + FN)
    Mide qué proporción de los positivos reales fueron detectados.
    
    Args:
        tp (int): Verdaderos Positivos
        fn (int): Falsos Negativos
    Returns:
        float: Recall en porcentaje (0-100), o 0 si no hay datos.
    """
    denominador = tp + fn
    if denominador == 0:
        return 0.0
    return (tp / denominador) * 100


def calcular_f1_score(precision, recall):
    """
    F1-Score = 2 × (Precisión × Recall) / (Precisión + Recall)
    Media armónica entre Precisión y Recall. Balance entre ambas métricas.
    
    Args:
        precision (float): Precisión en porcentaje (0-100)
        recall (float): Recall en porcentaje (0-100)
    Returns:
        float: F1-Score en porcentaje (0-100), o 0 si no hay datos.
    """
    denominador = precision + recall
    if denominador == 0:
        return 0.0
    return 2 * (precision * recall) / denominador


def calcular_exactitud(tp, tn, total):
    """
    Exactitud (Accuracy) = (TP + TN) / Total
    Mide la proporción total de clasificaciones correctas.
    
    Args:
        tp (int): Verdaderos Positivos
        tn (int): Verdaderos Negativos
        total (int): Total de casos evaluados (TP + FP + FN + TN)
    Returns:
        float: Exactitud en porcentaje (0-100), o 0 si no hay datos.
    """
    if total == 0:
        return 0.0
    return ((tp + tn) / total) * 100


def calcular_especificidad(tn, fp):
    """
    Especificidad = TN / (TN + FP)
    Mide qué proporción de los negativos reales fueron correctamente rechazados.
    
    Args:
        tn (int): Verdaderos Negativos
        fp (int): Falsos Positivos
    Returns:
        float: Especificidad en porcentaje (0-100), o 0 si no hay datos.
    """
    denominador = tn + fp
    if denominador == 0:
        return 0.0
    return (tn / denominador) * 100


def calcular_tiempo_promedio_frame(tiempo_total_ms, total_frames):
    """
    Tiempo promedio por frame = Tiempo total / Total de frames.
    
    Args:
        tiempo_total_ms (float): Tiempo total de procesamiento en milisegundos
        total_frames (int): Total de frames procesados
    Returns:
        float: Tiempo promedio por frame en milisegundos, o 0 si no hay datos.
    """
    if total_frames == 0:
        return 0.0
    return tiempo_total_ms / total_frames


def generar_reporte(tp, fp, fn, tn, tiempo_total_ms=0, total_frames=0,
                    modelo="", video=""):
    """
    Genera un diccionario con todas las métricas calculadas y un string 
    de reporte formateado para consola.
    
    Args:
        tp (int): Verdaderos Positivos
        fp (int): Falsos Positivos
        fn (int): Falsos Negativos
        tn (int): Verdaderos Negativos
        tiempo_total_ms (float): Tiempo total de procesamiento en ms
        total_frames (int): Total de frames procesados
        modelo (str): Nombre del modelo utilizado
        video (str): Nombre del video procesado
        
    Returns:
        dict: Diccionario con todas las métricas calculadas:
            - precision, recall, f1_score, exactitud, especificidad
            - tiempo_promedio_frame, total_evaluados
            - tp, fp, fn, tn
            - reporte_texto (str formateado para consola)
    """
    total_evaluados = tp + fp + fn + tn

    precision = calcular_precision(tp, fp)
    recall = calcular_recall(tp, fn)
    f1_score = calcular_f1_score(precision, recall)
    exactitud = calcular_exactitud(tp, tn, total_evaluados)
    especificidad = calcular_especificidad(tn, fp)
    tiempo_promedio = calcular_tiempo_promedio_frame(tiempo_total_ms, total_frames)

    reporte = []
    reporte.append("")
    reporte.append("=" * 55)
    reporte.append("  RESUMEN DE SESIÓN DE RECONOCIMIENTO")
    reporte.append("=" * 55)
    if video:
        reporte.append(f"  Video:               {video}")
    if modelo:
        reporte.append(f"  Modelo:              {modelo}")
    reporte.append("-" * 55)
    reporte.append(f"  Verdaderos Positivos  (TP): {tp}")
    reporte.append(f"  Falsos Positivos      (FP): {fp}")
    reporte.append(f"  Falsos Negativos      (FN): {fn}")
    reporte.append(f"  Verdaderos Negativos  (TN): {tn}")
    reporte.append(f"  Total evaluados:            {total_evaluados}")
    reporte.append("-" * 55)
    reporte.append(f"  Precisión:           {precision:>7.2f}%")
    reporte.append(f"  Recall:              {recall:>7.2f}%")
    reporte.append(f"  F1-Score:            {f1_score:>7.2f}%")
    reporte.append(f"  Exactitud:           {exactitud:>7.2f}%")
    reporte.append(f"  Especificidad:       {especificidad:>7.2f}%")
    if total_frames > 0:
        reporte.append(f"  Tiempo/Frame:        {tiempo_promedio:>7.2f} ms")
        reporte.append(f"  Total Frames:        {total_frames}")
    reporte.append("=" * 55)
    reporte.append("")

    reporte_texto = "\n".join(reporte)

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "total_evaluados": total_evaluados,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "exactitud": exactitud,
        "especificidad": especificidad,
        "tiempo_promedio_frame": tiempo_promedio,
        "tiempo_total_ms": tiempo_total_ms,
        "total_frames": total_frames,
        "reporte_texto": reporte_texto,
    }
