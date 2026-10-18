"""Recalcula activaciones durante backward para ahorrar memoria."""

from contextlib import contextmanager, nullcontext
from typing import Generator

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


@contextmanager
def preserve_batch_norm_stats(block: nn.Module) -> Generator[None, None, None]:
    """Evita contar dos veces el mismo lote al recalcular un bloque.

    Args:
        block: Bloque cuyas estadísticas de BatchNorm se deben conservar.

    Yields:
        Contexto que restaura las estadísticas al terminar la recomputación.
    """
    # BatchNorm debe seguir usando las estadísticas del lote durante backward.
    # Guardamos sus buffers para deshacer únicamente la segunda actualización.
    buffers = []
    for layer in block.modules():
        if isinstance(layer, nn.BatchNorm2d) and layer.track_running_stats:
            for value in (layer.running_mean, layer.running_var, layer.num_batches_tracked):
                buffers.append((value, value.clone()))
    try:
        yield
    finally:
        # No usamos eval(): cambiaría el cálculo y, por tanto, los gradientes.
        with torch.no_grad():
            for value, saved in buffers:
                value.copy_(saved)


def checkpoint_block(block: nn.Module, images: torch.Tensor) -> torch.Tensor:
    """Ejecuta un bloque y permite recalcular sus activaciones en backward.

    Args:
        block: Bloque convolucional que se quiere recalcular.
        images: Activaciones de entrada; no necesitan requires_grad=True.

    Returns:
        Salida del bloque con el mismo cálculo que sin checkpointing.
    """
    # El primer contexto corresponde al forward; el segundo, a su recomputación.
    # use_reentrant=False permite entrenar también el primer bloque con imágenes
    # que no requieren gradientes. PyTorch conserva el contexto de autocast.
    return checkpoint(
        block, images, use_reentrant=False,
        context_fn=lambda: (nullcontext(), preserve_batch_norm_stats(block)),
    )
