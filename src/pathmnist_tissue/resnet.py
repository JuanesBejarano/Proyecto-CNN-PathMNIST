"""Transferencia de aprendizaje con ResNet-18 y entradas RGB entre cero y uno"""

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

from .checkpointing import checkpoint_block


class TransferResNet18(nn.Module):
    """Conserva el extractor preentrenado y aprende nueve clases de tejido"""

    def __init__(self, pretrained: bool = True, gradient_checkpointing: bool = False):
        """Construye la red y congela inicialmente el extractor

        Args:
            pretrained: Descarga pesos ImageNet; False permite recargar pesos locales
            gradient_checkpointing: Recalcula layer4 durante su ajuste
        """
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None  # Usa pesos aprendidos en ImageNet o construye la arquitectura sin descargarlos para cargar un checkpoint local
        self.network = resnet18(weights=weights)
        self.network.fc = nn.Linear(self.network.fc.in_features, 9)  # Sustituye las 1.000 salidas de ImageNet por las nueve clases de tejido
        self.gradient_checkpointing = gradient_checkpointing
        self.finetune = False  # Al inicio solo se entrena el clasificador, layer4 se habilita después desde el entrenamiento
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))  # Guarda la media RGB de ImageNet como buffer que viaja con el modelo y sus checkpoints sin ser entrenable
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))  # La forma [1, 3, 1, 1] permite normalizar por canal mediante broadcasting con cualquier lote y resolución
        for parameter in self.network.parameters():
            parameter.requires_grad_(False)  # Congela los pesos preentrenados para que no se calculen sus gradientes
        for parameter in self.network.fc.parameters():  # Reactiva únicamente los parámetros del nuevo clasificador durante la primera fase
            parameter.requires_grad_(True)
        self.train()

    def unfreeze_last_block(self):
        """Permite ajustar layer4 y mantiene congelados los bloques anteriores"""
        self.finetune = True
        for parameter in self.network.layer4.parameters():
            parameter.requires_grad_(True)
        self.train(self.training)  # Reaplica el modo actual para que layer4 actualice también sus estadísticas de BatchNorm si está entrenando

    def train(self, mode: bool = True):
        """Configura el modo sin alterar estadísticas de los bloques congelados

        Args:
            mode: True para entrenamiento; False para evaluación

        Returns:
            El modelo, como nn.Module.train
        """
        super().train(mode)
        self.network.eval()  # Mantiene las estadísticas de BatchNorm de los bloques congelados aunque el modelo completo esté entrenando
        self.network.fc.train(mode)
        self.network.layer4.train(mode and self.finetune)  # Solo permite actualizar BatchNorm de layer4 cuando se habilitó su ajuste y no se está evaluando
        return self

    def forward(self, images):
        """Calcula logits para CrossEntropyLoss

        Args:
            images: Tensor RGB [batch, 3, alto, ancho], en el rango [0, 1]

        Returns:
            Nueve logits por imagen
        """
        net = self.network
        x = (images - self.mean) / self.std  # Normaliza entradas RGB entre 0 y 1 con las estadísticas de ImageNet sin redimensionar ni recortar
        x = net.maxpool(net.relu(net.bn1(net.conv1(x))))  # Aplica la entrada convolucional de ResNet antes de sus bloques residuales
        x = net.layer3(net.layer2(net.layer1(x)))  # Extrae características con los tres grupos residuales que permanecen congelados
        if self.gradient_checkpointing and self.training and self.finetune and torch.is_grad_enabled():  # Recalcula activaciones solo al entrenar layer4 con gradientes, evitando este costo durante la inferencia
            x = checkpoint_block(net.layer4, x)  # Ahorra memoria de activaciones a cambio de repetir cálculos durante backward
        else:
            x = net.layer4(x)
        return net.fc(net.avgpool(x).flatten(1))  # Promedia cada canal espacialmente y entrega nueve logits sin softmax para CrossEntropyLoss

    def predict_proba(self, images):
        """Devuelve probabilidades por clase para un lote RGB"""
        return self(images).softmax(dim=1)  # Convierte los logits en probabilidades cuya suma por imagen es 1
