from dataclasses import dataclass
from pathlib import Path

_MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

@dataclass(frozen=True)
class ModelSpec:
    path: Path
    family: str
    notes: str

MODEL_REGISTRY: dict[str, ModelSpec] = {
    "best_model_og_LN": ModelSpec(_MODELS_DIR / "best_model_og_LN.h5", "LN", "LeNet5, no augmentation"),
    "best_model_manual_aug_LN": ModelSpec(_MODELS_DIR / "best_model_manual_aug_LN.h5", "LN", "LeNet5, manual warping"),
    "best_model_gs_aug_LN": ModelSpec(_MODELS_DIR / "best_model_gs_aug_LN.h5", "LN", "LeNet5, grid-search warping"),
    "best_model_gan_aug_LN": ModelSpec(_MODELS_DIR / "best_model_gan_aug_LN.h5", "LN", "LeNet5, DCGAN oversampling"),

    "best_model_og_EN": ModelSpec(_MODELS_DIR / "best_model_og_EN.h5", "EN", "EfficientNetV2B0, no augmentation"),
    "best_model_manual_aug_EN": ModelSpec(_MODELS_DIR / "best_model_manual_aug_EN.h5", "EN", "EfficientNetV2B0, manual warping"),
    "best_model_gs_aug_EN": ModelSpec(_MODELS_DIR / "best_model_gs_aug_EN.h5", "EN", "EfficientNetV2B0, grid-search warping"),
    "best_model_gan_aug_EN": ModelSpec(_MODELS_DIR / "best_model_gan_aug_EN.h5", "EN", "EfficientNetV2B0, DCGAN oversampling"),
}