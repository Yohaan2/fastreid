from PIL import Image
from torchvision import transforms
import torch


# Dimensiones estándar FastReID
PERSON_HEIGHT = 384
PERSON_WIDTH = 128
VEHICLE_HEIGHT = 256
VEHICLE_WIDTH = 256

# Media y desviación estándar ImageNet (usadas por FastReID)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class PadToRatio:
    """
    Rellena la imagen con negro para alcanzar el aspect ratio objetivo
    ANTES del resize, evitando distorsión de identidad visual.

    Sin esto, un crop cuadrado (200x200) se deforma a 256x128 y el modelo
    pierde las features de persona — solo queda textura de fondo.
    """

    def __init__(self, height: int, width: int) -> None:
        self.target_ratio = height / width  # e.g. 256/128 = 2.0

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        current_ratio = h / w

        if abs(current_ratio - self.target_ratio) < 0.05:
            return img

        if current_ratio < self.target_ratio:
            # Imagen más ancha que el target: añadir padding arriba/abajo
            target_h = int(round(w * self.target_ratio))
            new_img = Image.new(img.mode, (w, target_h), 0)
            paste_y = (target_h - h) // 2
            new_img.paste(img, (0, paste_y))
        else:
            # Imagen más alta que el target: añadir padding izquierda/derecha
            target_w = int(round(h / self.target_ratio))
            new_img = Image.new(img.mode, (target_w, h), 0)
            paste_x = (target_w - w) // 2
            new_img.paste(img, (paste_x, 0))

        return new_img


def build_inference_transform(height: int, width: int) -> transforms.Compose:
    """
    Construye el pipeline de transformación estándar para inferencia FastReID.
    PadToRatio → Resize → ToTensor → Normalize (ImageNet stats).
    """
    return transforms.Compose([
        PadToRatio(height, width),
        transforms.Resize(
            (height, width),
            interpolation=transforms.InterpolationMode.BICUBIC,
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def preprocess_image(
    image: Image.Image,
    transform: transforms.Compose,
    device: torch.device,
) -> torch.Tensor:
    """
    Preprocesa una imagen PIL y retorna un tensor listo para inferencia.
    Retorna shape (1, C, H, W) en el device indicado.
    """
    if image.mode != "RGB":
        image = image.convert("RGB")
    tensor = transform(image)          # (C, H, W)
    tensor = tensor.unsqueeze(0)       # (1, C, H, W)
    return tensor.to(device)


PERSON_TRANSFORM = build_inference_transform(PERSON_HEIGHT, PERSON_WIDTH)
VEHICLE_TRANSFORM = build_inference_transform(VEHICLE_HEIGHT, VEHICLE_WIDTH)
