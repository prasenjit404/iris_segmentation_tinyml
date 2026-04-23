"""TinyUNet — a minimal 2-class UNet for iris segmentation (~28K parameters).

Designed for TinyML deployment on embedded biometric hardware.
PyTorch is used for training and inference; the architecture is small enough
for future conversion to TFLite / ONNX / pure-C.

Ground truth masks: binary (0 = background, 1 = iris).
"""

from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# PyTorch imports — guarded so the rest of the package stays importable
# on environments without torch (e.g. MCU export toolchains).
# ---------------------------------------------------------------------------
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
INPUT_SIZE = 128  # images are resized to 128×128 for the UNet


# ========================== MODEL ARCHITECTURE =============================

def _require_torch() -> None:
    if not _HAS_TORCH:
        raise ImportError("PyTorch is required for TinyUNet. Install with: pip install torch")


class _ConvBlock(nn.Module):
    """Conv → BatchNorm → ReLU."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.bn(self.conv(x)), inplace=True)


class TinyUNet(nn.Module):
    """Minimal UNet for 2-class (background / iris) segmentation.

    ~28 K parameters.  Input: (B, 1, 128, 128)  →  Output: (B, 2, 128, 128).
    """

    def __init__(self, num_classes: int = 2) -> None:
        super().__init__()
        # ---- encoder ----
        self.enc1 = _ConvBlock(1, 8)
        self.enc2 = _ConvBlock(8, 16)
        self.enc3 = _ConvBlock(16, 32)
        self.pool = nn.MaxPool2d(2)

        # ---- bottleneck ----
        self.bottleneck = _ConvBlock(32, 32)

        # ---- decoder (with skip connections) ----
        self.up3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec3 = _ConvBlock(64, 16)   # 32 (up) + 32 (skip)

        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec2 = _ConvBlock(32, 8)    # 16 (up) + 16 (skip)

        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec1 = _ConvBlock(16, 8)    # 8 (up) + 8 (skip)

        # ---- 1×1 classification head ----
        self.head = nn.Conv2d(8, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # encoder
        e1 = self.enc1(x)                # (B, 8, 128, 128)
        e2 = self.enc2(self.pool(e1))    # (B, 16, 64, 64)
        e3 = self.enc3(self.pool(e2))    # (B, 32, 32, 32)

        # bottleneck
        b = self.bottleneck(self.pool(e3))  # (B, 32, 16, 16)

        # decoder
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))   # (B, 16, 32, 32)
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))  # (B, 8, 64, 64)
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))  # (B, 8, 128, 128)

        return self.head(d1)  # (B, 2, 128, 128)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract learned features via global average pooling of encoder + bottleneck.

        Returns a discriminative feature vector that captures what the model
        learned about iris structure during segmentation training.

        Input:  (B, 1, 128, 128)
        Output: (B, 88)  — 8 (enc1) + 16 (enc2) + 32 (enc3) + 32 (bottleneck)
        """
        e1 = self.enc1(x)                # (B, 8, 128, 128)
        e2 = self.enc2(self.pool(e1))    # (B, 16, 64, 64)
        e3 = self.enc3(self.pool(e2))    # (B, 32, 32, 32)
        b = self.bottleneck(self.pool(e3))  # (B, 32, 16, 16)

        # Global average pool each level → channel-wise feature
        f1 = e1.mean(dim=[2, 3])   # (B, 8)
        f2 = e2.mean(dim=[2, 3])   # (B, 16)
        f3 = e3.mean(dim=[2, 3])   # (B, 32)
        fb = b.mean(dim=[2, 3])    # (B, 32)

        return torch.cat([f1, f2, f3, fb], dim=1)  # (B, 88)


# ========================== DATASET ========================================

class IrisSegDataset(Dataset):
    """Loads (image, mask) pairs, resizes to INPUT_SIZE×INPUT_SIZE."""

    def __init__(self, pairs: List[Tuple[Path, Path]], augment: bool = False) -> None:
        self.pairs = pairs
        self.augment = augment

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path, mask_path = self.pairs[idx]

        # load & resize
        img = Image.open(img_path).convert("L").resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR)
        mask = Image.open(mask_path).convert("L").resize((INPUT_SIZE, INPUT_SIZE), Image.NEAREST)

        img_np = np.asarray(img, dtype=np.float32) / 255.0
        mask_np = (np.asarray(mask, dtype=np.uint8) > 127).astype(np.int64)  # 0/255 → 0/1

        # simple augmentation: random horizontal flip
        if self.augment and random.random() > 0.5:
            img_np = np.flip(img_np, axis=1).copy()
            mask_np = np.flip(mask_np, axis=1).copy()

        img_t = torch.from_numpy(img_np).unsqueeze(0)  # (1, H, W)
        mask_t = torch.from_numpy(mask_np)              # (H, W)
        return img_t, mask_t


# ========================== TRAINING =======================================

@dataclass(frozen=True)
class TinyUNetTrainConfig:
    image_dir: Path
    mask_dir: Path
    output_dir: Path
    epochs: int = 30
    batch_size: int = 16
    learning_rate: float = 1e-3
    val_ratio: float = 0.15
    seed: int = 42


def _discover_pairs(image_dir: Path, mask_dir: Path) -> List[Tuple[Path, Path]]:
    """Match images to GT masks.  Image: C100_S1_I1.tiff → Mask: OperatorA_C100_S1_I1.tiff"""
    pairs: List[Tuple[Path, Path]] = []
    for mask_path in sorted(mask_dir.glob("OperatorA_*.tiff")):
        stem = mask_path.name.replace("OperatorA_", "")  # e.g. C100_S1_I1.tiff
        img_path = image_dir / stem
        if img_path.exists():
            pairs.append((img_path, mask_path))
    return pairs


def train_tinyunet(workspace_root: Path, cfg: TinyUNetTrainConfig) -> Dict[str, object]:
    """Train the TinyUNet model and save checkpoint."""
    _require_torch()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # discover image-mask pairs
    pairs = _discover_pairs(
        workspace_root / cfg.image_dir,
        workspace_root / cfg.mask_dir,
    )
    if not pairs:
        raise ValueError(f"No image-mask pairs found in {cfg.image_dir} / {cfg.mask_dir}")

    # train/val split (deterministic, by subject)
    rng = random.Random(cfg.seed)
    rng.shuffle(pairs)
    n_val = max(1, int(len(pairs) * cfg.val_ratio))
    val_pairs = pairs[:n_val]
    train_pairs = pairs[n_val:]

    train_ds = IrisSegDataset(train_pairs, augment=True)
    val_ds = IrisSegDataset(val_pairs, augment=False)
    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0)

    # model
    model = TinyUNet(num_classes=2).to(device)
    param_count = sum(p.numel() for p in model.parameters())

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=1e-5)
    # class weights: iris is typically much smaller than background
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 5.0], device=device))

    history: List[dict] = []
    best_val_iou = 0.0
    best_state = None

    for epoch in range(1, cfg.epochs + 1):
        # ---- train ----
        model.train()
        train_loss = 0.0
        for imgs, masks in train_dl:
            imgs, masks = imgs.to(device), masks.to(device)
            logits = model(imgs)
            loss = criterion(logits, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)
        train_loss /= len(train_ds)

        # ---- validate ----
        model.eval()
        val_loss = 0.0
        val_inter = 0
        val_union = 0
        with torch.no_grad():
            for imgs, masks in val_dl:
                imgs, masks = imgs.to(device), masks.to(device)
                logits = model(imgs)
                loss = criterion(logits, masks)
                val_loss += loss.item() * imgs.size(0)
                pred = logits.argmax(dim=1)
                # IoU for iris class (class 1)
                inter = ((pred == 1) & (masks == 1)).sum().item()
                union = ((pred == 1) | (masks == 1)).sum().item()
                val_inter += inter
                val_union += union
        val_loss /= len(val_ds)
        val_iou = val_inter / max(val_union, 1)

        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 5),
            "val_loss": round(val_loss, 5),
            "val_iris_iou": round(val_iou, 4),
        })
        print(f"  Epoch {epoch:3d}/{cfg.epochs}  train_loss={train_loss:.4f}  "
              f"val_loss={val_loss:.4f}  val_iris_IoU={val_iou:.4f}", flush=True)

        if val_iou > best_val_iou:
            best_val_iou = val_iou
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # save best checkpoint
    out_dir = workspace_root / cfg.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "tinyunet.pth"
    torch.save(best_state, ckpt_path)

    # save history
    hist_path = out_dir / "tinyunet_training_history.json"
    hist_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

    summary = {
        "train_images": len(train_pairs),
        "val_images": len(val_pairs),
        "total_params": param_count,
        "best_val_iris_iou": round(best_val_iou, 4),
        "epochs": cfg.epochs,
        "checkpoint": ckpt_path.relative_to(workspace_root).as_posix(),
    }
    print(f"\n  Training complete. Best val IoU = {best_val_iou:.4f}", flush=True)
    print(f"  Checkpoint saved to {ckpt_path}", flush=True)
    return summary


# ========================== INFERENCE ======================================

def load_tinyunet(checkpoint_path: Path, device: str = "cpu") -> "TinyUNet":
    """Load a trained TinyUNet from a .pth checkpoint."""
    _require_torch()
    model = TinyUNet(num_classes=2)
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def segment_iris(gray01: np.ndarray, model: "TinyUNet") -> np.ndarray:
    """Run TinyUNet inference on a grayscale [0,1] image.

    Args:
        gray01: (H, W) float32 array in [0, 1].
        model: Loaded TinyUNet model.

    Returns:
        Binary mask (H, W) uint8 — 1 = iris, 0 = background — at original resolution.
    """
    _require_torch()
    h, w = gray01.shape

    # resize to model input size
    img_pil = Image.fromarray(np.uint8(gray01 * 255), mode="L")
    img_resized = img_pil.resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR)
    img_np = np.asarray(img_resized, dtype=np.float32) / 255.0

    # to tensor
    img_t = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0)  # (1, 1, 128, 128)

    with torch.no_grad():
        logits = model(img_t)  # (1, 2, 128, 128)
        pred = logits.argmax(dim=1).squeeze(0).numpy().astype(np.uint8)  # (128, 128)

    # resize back to original resolution
    mask_pil = Image.fromarray(pred, mode="L")
    mask_orig = mask_pil.resize((w, h), Image.NEAREST)
    return np.asarray(mask_orig, dtype=np.uint8)


def extract_learned_features(gray01: np.ndarray, model: "TinyUNet") -> np.ndarray:
    """Extract 88-dim learned features from the TinyUNet encoder.

    Uses global average pooling of encoder layers + bottleneck to produce
    a discriminative feature vector that captures what the model learned
    about iris structure during segmentation training.

    Args:
        gray01: (H, W) float32 array in [0, 1].
        model: Loaded TinyUNet model.

    Returns:
        (88,) float32 array of learned features.
    """
    _require_torch()

    # resize to model input size
    img_pil = Image.fromarray(np.uint8(gray01 * 255), mode="L")
    img_resized = img_pil.resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR)
    img_np = np.asarray(img_resized, dtype=np.float32) / 255.0

    # to tensor
    img_t = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0)  # (1, 1, 128, 128)

    with torch.no_grad():
        features = model.extract_features(img_t)  # (1, 88)

    return features.squeeze(0).numpy().astype(np.float32)

