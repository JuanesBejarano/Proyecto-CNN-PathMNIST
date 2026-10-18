"""Extracción de mapas de activación de la CNN base."""

import torch

from .model import SimpleCNN


@torch.inference_mode() # Este decorador hace que el contenido de la función se ejecute solo en modo inferencia, evitando así crear el grafo computacional para backward
def feature_maps(
    model: SimpleCNN, image: torch.Tensor
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Extrae activaciones tras ReLU y antes del pooling de cada bloque.

    Args:
        model: SimpleCNN entrenada.
        image: Una imagen RGB de forma [1, 3, alto, ancho].

    Returns:
        Logits y diccionario de mapas por bloque, todos en CPU.
    """
    if image.ndim != 4 or image.shape[0] != 1 or image.shape[1] != 3:
        raise ValueError("Se espera una sola imagen RGB con forma [1, 3, alto, ancho].")

    maps: dict[str, torch.Tensor] = {} # diccionario con el número de bloque y su mapa de activación correspondiente
    hooks = [] # un hook es una función que PyTorch ejecuta después de que ejecutamos una función, en este caso, la activación, aquí almacenamos esos hooks
    was_training = model.training
    try:
        for index, block in enumerate(model.features, start=1): # obtenemos tuplas de cada bloque de la red convolucional enumeradas desde uno 
            name = f"bloque_{index}"

            # key fija el nombre de cada bloque al crear el hook
            def save_activation(_module, _inputs, output, *, key=name):
                maps[key] = output.cpu()

            hooks.append(block[2].register_forward_hook(save_activation)) # block[2] es la capa de activación relu, y register_forward_hook es un método que se hereda de nn.Module, la cual ejecuta la función "save_activation" después de cada ejecución relu, es decir, guarda la activación en el dict con "ejecuta maps[key] = output.cpu()"
        model.eval() # Setea el modelo en modo evaluación, batch_norm deja de usar las estadísticas de los batches en entrenamiento, y usa las acumuladas del mismo (con la media móvil) y dropout deja de apagar neuronas
        device = next(model.parameters()).device
        logits = model(image.to(device)).cpu()
    finally:
        # Los hooks son temporales; también se retiran si falla el forward
        for hook in hooks:
            hook.remove()
        model.train(was_training)
    return logits, maps
