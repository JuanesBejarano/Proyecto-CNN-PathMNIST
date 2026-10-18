"""Compatibilidad de corridas guardadas con la arquitectura actual"""

import ast
import hashlib

LEGACY_BATCH_NORM_HASH = "33b3da51455212e25fc98fa8e7bd2643289a4477c8612c44134c960037402195"  # Huella de una versión antigua con BatchNorm fijo que se reconoce como compatible
PREVIOUS_MODEL_HASH = "c343c55e126c1727318a5eda02f0a4ac6a3f6898a87c3eaa6d2003455f096dfd"  # Huella de una versión anterior que ya registraba si se usaba BatchNorm
PRE_CHECKPOINT_CODE_HASH = "ae2f3da1df82e0c8d57c09d1aa794cba291f80e0cffa5bd00fd6dafc84836497"  # Huella ejecutable anterior al checkpointing, cuyos pesos y cálculo de inferencia siguen siendo compatibles


PRE_GROUPNORM_CODE_HASH = "def8a78ffbe24cf42a4d90e9e2b951f473410c81d33a492c56a39f01ba0b3913"  # Huella ejecutable anterior a GroupNorm, compatible con las corridas guardadas de esa versión

def model_code_hash(source: str) -> str:
    """Calcula una huella del código que ignora comentarios y docstrings

    Args:
        source: Código fuente de model.py

    Returns:
        Hash SHA-256 de la estructura ejecutable del archivo
    """
    tree = ast.parse(source)  # Convierte el código fuente en un árbol de sintaxis que ya excluye los comentarios
    for node in ast.walk(tree):  # Recorre todos los nodos del árbol para encontrar documentación que no afecta a la ejecución
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):  # Busca módulos, clases y funciones, que pueden tener un docstring al inicio
            if node.body and isinstance(node.body[0], ast.Expr):  # Comprueba que el cuerpo tenga instrucciones y que la primera sea una expresión
                value = node.body[0].value  # Obtiene el valor de la primera expresión para comprobar si es una cadena de documentación
                if isinstance(value, ast.Constant) and isinstance(value.value, str):  # Reconoce una cadena literal inicial como docstring
                    node.body.pop(0)  # Elimina el docstring del árbol temporal sin modificar el archivo original
    structure = ast.dump(tree, include_attributes=False)  # Representa la estructura del código sin posiciones de línea ni columna para ignorar cambios de formato
    return hashlib.sha256(structure.encode()).hexdigest()  # Calcula una huella SHA-256 hexadecimal de la estructura convertida a bytes


def model_source_compatible(
    config: dict, current_hash: str, current_code_hash: str | None = None
) -> bool:
    """Reconoce versiones guardadas con la misma estructura de parámetros

    Args:
        config: Configuración guardada junto al modelo
        current_hash: Hash SHA-256 del código actual de la arquitectura
        current_code_hash: Huella que ignora comentarios y docstrings

    Returns:
        True si la corrida se puede cargar con SimpleCNN actual
    """
    adaptive_pool = config.get("adaptive_pool", True)  # Lee si se usó promedio global y asume True para las corridas antiguas que no guardaban esta opción
    expected_architecture = (  # Determina el nombre de arquitectura esperado según el tipo de clasificador
        "SimpleCNN-3blocks-GAP" if adaptive_pool else "SimpleCNN-3blocks-Flatten"
    )
    if config.get("architecture") != expected_architecture:  # Rechaza una corrida cuya arquitectura declarada no coincide con su configuración de pooling
        return False  
    saved_code_hash = config.get("model_code_sha256")  # Busca la huella de la estructura ejecutable guardada durante el entrenamiento
    if saved_code_hash is not None:  # Prioriza la comparación del código ejecutable cuando la corrida incluye esa huella
        return saved_code_hash in (current_code_hash, PRE_CHECKPOINT_CODE_HASH, PRE_GROUPNORM_CODE_HASH)  # Acepta el código actual o las versiones anteriores reconocidas explícitamente como compatibles
    saved_hash = config.get("model_sha256")  # Recupera la huella del texto completo para corridas antiguas sin huella de código ejecutable
    if saved_hash == current_hash:  # Comprueba si el archivo de arquitectura guardado coincide exactamente con el actual
        return True  
    if not adaptive_pool:  # Impide aplicar las excepciones históricas a modelos con Flatten
        return False 
    if saved_hash == LEGACY_BATCH_NORM_HASH:  # Reconoce la versión antigua que siempre utilizaba BatchNorm
        return config.get("batch_norm", True) is True  # La acepta solo si BatchNorm está activado y asume True cuando el dato no se guardó
    return saved_hash == PREVIOUS_MODEL_HASH and isinstance(config.get("batch_norm"), bool)  # Acepta la otra versión conocida solo si la configuración declara BatchNorm con un booleano
