"""Transferencia de aprendizaje con ResNet-18 y entradas RGB entre cero y uno."""

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

from .checkpointing import checkpoint_block


class TransferResNet18(nn.Module):
    """Conserva el extractor preentrenado y aprende nueve clases de tejido."""

    def __init__(self, pretrained: bool = True, gradient_checkpointing: bool = False):
        """Construye la red y congela inicialmente el extractor.

        Args:
            pretrained: Descarga pesos ImageNet; False permite recargar pesos locales.
            gradient_checkpointing: Recalcula layer4 durante su ajuste.
        """
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.network = resnet18(weights=weights)
        self.network.fc = nn.Linear(self.network.fc.in_features, 9)
        self.gradient_checkpointing = gradient_checkpointing
        self.finetune = False
        # La normalización queda dentro del modelo para usarla también al evaluar.
        # Conservamos la resolución nativa: no redimensionamos a 224 ni recortamos.
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        for parameter in self.network.parameters():
            parameter.requires_grad_(False)
        for parameter in self.network.fc.parameters():
            parameter.requires_grad_(True)
        self.train()

    def unfreeze_last_block(self):
        """Permite ajustar layer4 y mantiene congelados los bloques anteriores."""
        self.finetune = True
        for parameter in self.network.layer4.parameters():
            parameter.requires_grad_(True)
        self.train(self.training)

    def train(self, mode: bool = True):
        """Configura el modo sin alterar estadísticas de los bloques congelados.

        Args:
            mode: True para entrenamiento; False para evaluación.

        Returns:
            El modelo, como nn.Module.train.
        """
        super().train(mode)
        self.network.eval()
        self.network.fc.train(mode)
        self.network.layer4.train(mode and self.finetune)
        return self

    def forward(self, images):
        """Calcula logits para CrossEntropyLoss.

        Args:
            images: Tensor RGB [batch, 3, alto, ancho], en el rango [0, 1].

        Returns:
            Nueve logits por imagen.
        """
        net = self.network
        x = (images - self.mean) / self.std
        x = net.maxpool(net.relu(net.bn1(net.conv1(x))))
        x = net.layer3(net.layer2(net.layer1(x)))
        if self.gradient_checkpointing and self.training and self.finetune and torch.is_grad_enabled():
            x = checkpoint_block(net.layer4, x)
        else:
            x = net.layer4(x)
        return net.fc(net.avgpool(x).flatten(1))

    def predict_proba(self, images):
        """Devuelve probabilidades por clase para un lote RGB."""
        return self(images).softmax(dim=1)
