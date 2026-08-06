import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA disponible: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memoria GPU total: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB")
else:
    print("NO GPU DETECTADA - USANDO CPU")
    print("Este es el motivo de la lentitud extrema de Faster R-CNN")

import torchvision
print(f"Torchvision version: {torchvision.__version__}")

# Test rapido de velocidad
import time
print("\n--- Test de velocidad Faster R-CNN ---")
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
    weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
)
model.eval()

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print(f"Modelo cargado en: {device}")

# Simular un frame de 640x480
dummy = torch.rand(3, 480, 640).to(device)

# Warmup
with torch.no_grad():
    _ = model([dummy])

# Medir tiempo real
times = []
for i in range(5):
    start = time.time()
    with torch.no_grad():
        _ = model([dummy])
    elapsed = time.time() - start
    times.append(elapsed)
    print(f"  Frame {i+1}: {elapsed:.3f} segundos")

avg = sum(times) / len(times)
print(f"\nPromedio por frame: {avg:.3f} segundos")
print(f"FPS estimados: {1/avg:.1f}")
