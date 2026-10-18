"""Un paso de entrenamiento o evaluación."""

import torch
from torch import nn
from torch.utils.data import DataLoader


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    accumulation_steps: int = 1,
    amp_enabled: bool = True,
) -> dict[str, float]:
    """Ejecuta una época de entrenamiento o validación.

    Args:
        model: Modelo que produce logits.
        loader: Lotes de imágenes y etiquetas.
        criterion: Función de pérdida.
        device: CPU o GPU donde se calcula la época.
        optimizer: Si está presente, se actualizan los parámetros.
        scaler: Escalador para gradientes con precisión mixta en CUDA.
        amp_enabled: Activa autocast FP16 en CUDA; False conserva FP32.
        accumulation_steps: Lotes que aportan gradientes antes de actualizar pesos.
            Con 1 se actualizan los pesos después de cada lote.

    Returns:
        Pérdida media por imagen y proporción de aciertos.
    """
    if accumulation_steps < 1:
        raise ValueError("accumulation_steps debe ser al menos 1.")
    training = optimizer is not None # Devuelve un bool, True si optimizer no es None, es decir, vamos a entrenar (actualizamos pesos)
    model.train(training) # Si training es True, setea el modelo en modo entrenamiento, batch_norm usa las estadísticas del batch y dropout apaga activaciones aleatoriamente
    total_loss = total_correct = total = 0
    accumulated_samples = 0
    if training:
        optimizer.zero_grad(set_to_none=True)
    with torch.set_grad_enabled(training):
        for batch_index, (images, targets) in enumerate(loader, start=1):
            images = images.to(device, non_blocking=True) # non_blocking permite que el hilo de Python se ejecute sin esperar a que la transferencia de los tensores al acelerador termine
            targets = targets.view(-1).long().to(device, non_blocking=True)
            # AMP es opcional en CUDA; en CPU el contexto queda desactivado.
            # Ver: https://docs.pytorch.org/docs/2.14/amp.html#cuda-op-specific-behavior
            # autocast elige la precisión de cada operación del forward; backward
            # utiliza las precisiones correspondientes a esas operaciones.
            # Los parámetros siguen en FP32: no convertimos todo el modelo a FP16.
            with torch.autocast(device_type=device.type, enabled=amp_enabled and device.type == "cuda", dtype=torch.float16):
                logits = model(images)
                loss = criterion(logits, targets)
            if training:
                # loss es una media. La convertimos en suma para que cada imagen
                # pese igual, incluso si el último lote contiene menos imágenes.
                loss_sum = loss * images.size(0)
                accumulated_samples += images.size(0)
                if scaler is not None:
                    scaler.scale(loss_sum).backward()
                else:
                    loss_sum.backward()
                # backward suma en .grad; no borramos esos valores entre batches
                # También actualizamos al acabar la época si queda un grupo incompleto
                if batch_index % accumulation_steps == 0 or batch_index == len(loader):
                    update_weights(optimizer, scaler, accumulated_samples)
                    accumulated_samples = 0
            # La pérdida del lote es una media: ponderarla evita sesgar el último batch
            total_loss += loss.item() * images.size(0)
            total_correct += (logits.argmax(1) == targets).sum().item()
            total += images.size(0)
    return {"loss": total_loss / total, "accuracy": total_correct / total}


def update_weights(
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler | None,
    accumulated_samples: int,
) -> None:
    """Promedia los gradientes acumulados y hace una actualización.

    Args:
        optimizer: Optimizador que actualiza los parámetros.
        scaler: Escalador de AMP, o None si no se usa.
        accumulated_samples: Número real de imágenes del grupo acumulado.
    """
    # Mantenemos el mismo factor de escala durante todo el grupo de batches
    # Quitamos esa escala una sola vez, antes de modificar los gradientes
    if scaler is not None:
        scaler.unscale_(optimizer)
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            if parameter.grad is not None:
                parameter.grad.div_(accumulated_samples)
    if scaler is not None:
        scaler.step(optimizer)  # Omite la actualización si detectó inf o NaN
        scaler.update()  # Ajusta la escala para el siguiente grupo, no cada lote
    else:
        optimizer.step()
    # El siguiente grupo empieza sin los gradientes del anterior
    optimizer.zero_grad(set_to_none=True)
