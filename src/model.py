"""Arquitectura base, compatible con distintas resoluciones de entrada."""

import torch
from torch import nn

from .checkpointing import checkpoint_block


class SimpleCNN(nn.Module):
    """Clasifica imágenes RGB con tres bloques y un clasificador lineal.

    El promedio global opcional conserva el mismo tamaño del clasificador para
    distintas resoluciones. Sin él, el clasificador recibe todos los píxeles
    de los mapas finales y depende de la resolución elegida.
    """

    def __init__(
        self, num_classes: int = 9, batch_norm: bool = True,
        adaptive_pool: bool = True, size: int = 28,
        gradient_checkpointing: bool = False, normalization: str | None = None,
    ) -> None:
        """Construye la red.

        Args:
            num_classes: Número de clases de salida.
            batch_norm: Si se normaliza después de cada convolución.
            adaptive_pool: Si se promedia cada mapa completo antes de aplanar.
            size: Resolución de entrada; determina el clasificador sin promedio global.
            gradient_checkpointing: Recalcula activaciones en backward para ahorrar memoria.
            normalization: batch, group o none; si se omite, respeta batch_norm.
        """
        super().__init__()
        if size < 8:
            raise ValueError("La resolución debe ser al menos 8 para los tres MaxPool2d.")
        normalization = normalization or ("batch" if batch_norm else "none")
        if normalization not in ("batch", "group", "none"):
            raise ValueError("normalization debe ser batch, group o none.")
        self.normalization = normalization
        self.batch_norm = normalization == "batch"
        self.adaptive_pool = adaptive_pool
        self.gradient_checkpointing = gradient_checkpointing
        self.features = nn.Sequential(
            self._block(3, 32, batch_norm, normalization), # 3 canales -> 32 kernels (feature maps)
            self._block(32, 64, batch_norm, normalization), # 32 canales (los feature maps de la capa anterior) -> 64 kernels (feature maps transformados)
            self._block(64, 128, batch_norm, normalization), # 64 canales -> 128 feature maps
        )
        self.pool = nn.AdaptiveAvgPool2d(1) if adaptive_pool else nn.Identity() # AdaptiveAvgPool2d retorna el promedio por cada feature map, nos permite reducir cantidad de parámetros pero perdemos la activación para cada posición (también hacemos el flatten ya que retorna un vector de promedios de cada feature map)
        spatial_size = 1 if adaptive_pool else size // 8 # Dimensión de la imagen si no usamos adaptative_pool, dividimos size entre 8 ya que es el tamaño que nos queda después de aplicar tres veces (en _block) MaxPool2d(2), que nos reduce las dimensiones a la mitad, entonces sería 2**3=8.
        self.classifier = nn.Linear(128 * spatial_size**2, num_classes) # spatial_size**2 ya que al final tenemos 128 feature maps de spatial_size x spatial_size

    @staticmethod
    def _block(in_channels: int, out_channels: int, batch_norm: bool, normalization: str | None = None) -> nn.Sequential:
        """Crea un bloque convolucional con reducción espacial a la mitad.

        Args:
            in_channels: Canales de entrada.
            out_channels: Canales producidos por la convolución.
            batch_norm: Si se incluye BatchNorm cuando normalization se omite.
            normalization: Tipo de normalización; group utiliza ocho grupos.

        Returns:
            Bloque Conv-BatchNorm-ReLU-Pooling; Identity sustituye BatchNorm
            cuando está desactivado y conserva el índice de ReLU para los hooks.
        """
        normalization = normalization or ("batch" if batch_norm else "none")
        norm = nn.Identity()
        if normalization == "batch":
            norm = nn.BatchNorm2d(out_channels)
        elif normalization == "group":
            # Cada imagen se normaliza por separado: ocho grupos de canales,
            # incluyendo sus posiciones espaciales, sin estadísticas del batch.
            norm = nn.GroupNorm(num_groups=8, num_channels=out_channels)
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            norm, # Conserva ReLU en el índice 2 para los hooks de activaciones.
            nn.ReLU(inplace=True), # inplace nos permite modificar el tensor de entrada directamente en vez de hacer una copia, lo que nos permite ahorrar memoria
            nn.MaxPool2d(kernel_size=2), # representa cada vecindad de píxeles 2x2 con su máximo
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Calcula una puntuación por clase para cada imagen.

        Args:
            images: Lote RGB de forma [N, 3, alto, ancho].

        Returns:
            Logits de forma [N, num_classes].
        """
        if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
            # Conservamos las entradas de cada bloque y recalculamos sus
            # activaciones internas en backward: menos memoria, más cómputo.
            features = images
            for block in self.features:
                features = checkpoint_block(block, features)
        else:
            features = self.features(images)
        return self.classifier(self.pool(features).flatten(1)) # el flatten pasa de (B, 128, 1) a (B, 128), aplana desde la dimensión 1 

    def predict_proba(self, images: torch.Tensor) -> torch.Tensor:
        """Convierte los logits del modelo en probabilidades por clase.

        Args:
            images: Lote RGB de forma [N, 3, alto, ancho].

        Returns:
            Probabilidades de forma [N, num_classes]; cada fila suma uno.
        """
        return torch.softmax(self.forward(images), dim=1)
