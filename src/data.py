"""Carga de PathMNIST con las particiones oficiales"""

from pathlib import Path

from medmnist import PathMNIST
import torch
from torch.utils.data import DataLoader
from torchvision import transforms


CLASS_NAMES = [  # El orden coincide con los identificadores numéricos de las nueve clases de PathMNIST
    "adipose", "background", "debris", "lymphocytes", "mucus",
    "smooth_muscle", "normal_colon", "stroma", "tumor_epithelium",
]


def rotate_quarter_turn(image):
    """Gira un tensor un múltiplo de 90° sin interpolar sus píxeles

    Args:
        image: Imagen con dimensiones canal, alto y ancho

    Returns:
        Imagen girada 0, 90, 180 o 270 grados con igual probabilidad
    """
    turns = int(torch.randint(4, ()).item())  # Elige un entero entre 0 y 3 para decidir cuántos giros de 90 grados aplicar
    return torch.rot90(image, turns, dims=(-2, -1))  # Gira las dimensiones de alto y ancho sin cambiar los canales ni interpolar píxeles


def make_transform(split: str, augmentation: str = "none"):
    """Construye los aumentos; validación y test solo convierten a tensor

    Args:
        split: Partición train, val o test
        augmentation: none, geometric o geometric_color

    Returns:
        Transformación que conserva tamaño y canales de la imagen
    """
    if augmentation not in ("none", "geometric", "geometric_color"):  # Rechaza nombres de aumentos que no estén implementados
        raise ValueError("Augmentation debe ser none, geometric o geometric_color.")
    steps = [transforms.ToTensor()]  # Convierte la imagen RGB de uint8 a un tensor [canales, alto, ancho] con valores entre 0 y 1
    if split == "train" and augmentation != "none":  # Aplica aumentos solo al entrenamiento para conservar validación y test sin modificaciones aleatorias
        steps.extend([
            rotate_quarter_turn,  # Escoge una orientación de 0, 90, 180 o 270 grados cada vez que se carga la imagen
            transforms.RandomHorizontalFlip(),  # Invierte la imagen de izquierda a derecha con probabilidad de 0,5
            transforms.RandomVerticalFlip(),  # Invierte la imagen de arriba abajo con probabilidad de 0,5
        ])
        if augmentation == "geometric_color":  # Añade variaciones suaves de color a los giros y volteos
            steps.append(transforms.ColorJitter(
                brightness=0.1, contrast=0.1, saturation=0.1, hue=0,  # Varía brillo, contraste y saturación con factores entre 0,9 y 1,1 sin alterar el tono
            ))
    return transforms.Compose(steps)  # Agrupa las transformaciones para ejecutarlas en el orden definido


def make_loader(
    *, split: str, size: int, batch_size: int, num_workers: int,
    seed: int, data_dir: Path, augmentation: str = "none",
) -> DataLoader:
    """Carga una partición oficial de PathMNIST

    Args:
        split: Partición train, val o test
        size: Resolución cuadrada ofrecida por MedMNIST
        batch_size: Imágenes por lote
        num_workers: Procesos de carga en paralelo
        seed: Semilla para el orden aleatorio de entrenamiento
        data_dir: Carpeta de los archivos .npz
        augmentation: Aumentos aplicados únicamente a entrenamiento

    Returns:
        DataLoader; validación y test conservan el orden original, necesario
        para relacionar cada predicción con su índice en el dataset
    """
    if size not in (28, 64, 128, 224):  # Comprueba que la resolución solicitada esté disponible en PathMNIST
        raise ValueError("PathMNIST ofrece resoluciones 28, 64, 128 y 224.")
    if split not in ("train", "val", "test"):  # Comprueba que se haya pedido una de las tres particiones oficiales
        raise ValueError("La partición debe ser train, val o test.")
    transform = make_transform(split, augmentation)  # Prepara la conversión a tensor y los aumentos permitidos para esta partición
    data_dir.mkdir(parents=True, exist_ok=True)  # Crea la carpeta de datos y sus directorios padre si todavía no existen
    dataset = PathMNIST(
        split=split,  # Selecciona las imágenes y etiquetas de train, val o test
        size=size,  # Resolución de las imágenes
        root=str(data_dir),  # Indica dónde buscar o guardar los archivos de datos
        download=True,  # Permite descargar el dataset cuando no está disponible localmente
        transform=transform,  # Aplica las transformaciones cada vez que se obtiene una imagen
    )
    generator = torch.Generator().manual_seed(seed)  # Fija el generador usado para el orden aleatorio y las semillas de los procesos de carga
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=split == "train",  # Mezcla entrenamiento y conserva el orden de validación y test para identificar cada predicción
        generator=generator,  # Utiliza el generador inicializado con la semilla indicada
        num_workers=num_workers,  # Número de procesos en paralelo, con 0 se carga en el proceso principal
        pin_memory=torch.cuda.is_available(),  # Usa memoria de CPU fijada cuando hay CUDA para facilitar transferencias a la GPU (lo de fijar)
    )
