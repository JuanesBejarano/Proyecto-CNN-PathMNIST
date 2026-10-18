"""Comprueba equivalencia de gradientes y conservación de BatchNorm"""

import copy
import unittest

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.engine import run_epoch
from src.model import SimpleCNN


class TrainingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        torch.set_num_threads(1)  # Reduce el costo de paralelismo en estas operaciones pequeñas de prueba

    def test_accumulation_matches_large_batch_with_incomplete_final_group(self):
        # 19 imágenes: grupos efectivos de 8, 8 y 3. El último contiene
        # microlotes de 2 y 1, para comprobar la ponderación por imagen
        dataset = TensorDataset(torch.randn(19, 3, 16, 16), torch.randint(9, (19, 1)))
        for use_scaler in (False, True):
            with self.subTest(scaler=use_scaler):  # Identifica por separado los casos con y sin escalado si falla la prueba
                large = SimpleCNN(batch_norm=False, size=16)  # Desactiva BatchNorm porque sus estadísticas dependen del lote físico y romperían esta comparación
                small = copy.deepcopy(large)  # Parte de los mismos pesos para aislar el efecto de acumular gradientes
                large_optimizer = torch.optim.AdamW(large.parameters(), lr=1e-3)
                small_optimizer = torch.optim.AdamW(small.parameters(), lr=1e-3)
                # CPU GradScaler verifica también el orden unscale/step/update
                # sin necesitar una GPU en el entorno de pruebas
                scaler = torch.amp.GradScaler("cpu") if use_scaler else None
                expected = run_epoch(
                    large, DataLoader(dataset, batch_size=8), nn.CrossEntropyLoss(),
                    torch.device("cpu"), large_optimizer,
                )
                actual = run_epoch(
                    small, DataLoader(dataset, batch_size=2), nn.CrossEntropyLoss(),
                    torch.device("cpu"), small_optimizer, scaler, accumulation_steps=4,  # Acumula cuatro microlotes de dos imágenes para reproducir el lote de ocho del control
                )
                for left, right in zip(large.parameters(), small.parameters()):
                    torch.testing.assert_close(left, right, rtol=2e-4, atol=2e-6)  # Comprueba pesos finales equivalentes admitiendo diferencias pequeñas de redondeo
                self.assertAlmostEqual(expected["loss"], actual["loss"], places=5)
                self.assertEqual(expected["accuracy"], actual["accuracy"])
                for state in small_optimizer.state.values():
                    self.assertEqual(state["step"].item(), 3)  # Exige tres actualizaciones, una por cada grupo efectivo de 8, 8 y 3 imágenes

    def test_checkpoint_matches_gradients_and_batch_norm_buffers(self):
        for batch_norm in (False, True):
            for adaptive_pool in (False, True):
                with self.subTest(batch_norm=batch_norm, adaptive_pool=adaptive_pool):  # Comprueba checkpointing con las cuatro combinaciones de normalización y pooling
                    plain = SimpleCNN(batch_norm=batch_norm, adaptive_pool=adaptive_pool, size=16)
                    checked = copy.deepcopy(plain)  # Mantiene idénticos pesos y buffers entre el control y el modelo con checkpointing
                    checked.gradient_checkpointing = True
                    images = torch.randn(3, 3, 16, 16)
                    targets = torch.randint(9, (3,))
                    for _ in range(2):  # Repite forward y backward para detectar cambios acumulados en los buffers de BatchNorm
                        plain.zero_grad()
                        checked.zero_grad()
                        expected = plain(images)
                        actual = checked(images)
                        torch.testing.assert_close(expected, actual)
                        nn.functional.cross_entropy(expected, targets).backward()
                        nn.functional.cross_entropy(actual, targets).backward()
                        # Incluye el primer bloque: sus gradientes deben existir
                        # aunque las imágenes no tengan requires_grad=True
                        for left, right in zip(plain.parameters(), checked.parameters()):
                            self.assertIsNotNone(right.grad)  # Detecta parámetros que dejaron de recibir gradientes por un checkpoint incorrecto
                            torch.testing.assert_close(left.grad, right.grad)
                        for name, value in plain.state_dict().items():  # Incluye medias, varianzas y contadores de BatchNorm para detectar actualizaciones duplicadas al recalcular
                            torch.testing.assert_close(value, checked.state_dict()[name])
                    plain.eval()
                    checked.eval()
                    with torch.inference_mode():  # Comprueba que ambos modelos también coincidan al evaluar con las estadísticas guardadas
                        torch.testing.assert_close(plain(images), checked(images))

    def test_checkpoint_with_autocast(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if device.type == "cuda" else torch.bfloat16  # Usa la precisión mixta habitual de cada dispositivo
        plain = SimpleCNN(size=16).to(device)
        checked = copy.deepcopy(plain)
        checked.gradient_checkpointing = True
        images = torch.randn(3, 3, 16, 16, device=device)
        targets = torch.randint(9, (3,), device=device)
        with torch.autocast(device_type=device.type, dtype=dtype):  # Verifica que la recomputación del checkpoint respete el contexto de precisión mixta
            expected = nn.functional.cross_entropy(plain(images), targets)
            actual = nn.functional.cross_entropy(checked(images), targets)
        expected.backward()
        actual.backward()
        for left, right in zip(plain.parameters(), checked.parameters()):
            torch.testing.assert_close(left.grad, right.grad)
        for left, right in zip(plain.buffers(), checked.buffers()):  # Confirma que la recomputación no altere las estadísticas de BatchNorm frente al control
            torch.testing.assert_close(left, right)

    def test_invalid_accumulation_steps(self):
        with self.assertRaisesRegex(ValueError, "accumulation_steps"):  # Exige rechazar cero pasos de acumulación con un error que identifique el parámetro
            run_epoch(SimpleCNN(), [], nn.CrossEntropyLoss(), torch.device("cpu"),
                      accumulation_steps=0)


if __name__ == "__main__":
    unittest.main()
