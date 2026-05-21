"""YOLO encoders (Family C — task-supervised priors).

Two variants:
- YOLOBackboneEncoder: features from intermediate layer of YOLOv8 CSPDarknet
- YOLOBBoxEncoder: structured features from detection output (top-K boxes)

Requires `ultralytics`. Install with:
    pip install -e ".[yolo]"
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .base import VisualPriorEncoder


class YOLOBackboneEncoder(VisualPriorEncoder):
    """YOLOv8 backbone features as visual prior.

    Strips detection head, takes features from specified level (P3/P4/P5).
    Pretrained on COCO object detection.

    Output: (B, N_spatial, C) — spatial token sequence.

    TODO(integration): The exact way to slice YOLO internals depends on
    Ultralytics version. We pin ultralytics<9.0 in pyproject.toml, but
    if the slicing fails check `yolo_full.model.model` structure.
    """

    def __init__(
        self,
        model_name: str = "yolov8n",
        feature_level: int = 4,  # P4
    ):
        super().__init__()
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise ImportError(
                'YOLO encoder requires `ultralytics`. Install: pip install -e ".[yolo]"'
            ) from e

        self.feature_level = feature_level

        # Load pretrained YOLO and extract backbone
        yolo_full = YOLO(f"{model_name}.pt")
        # YOLOv8 structure: model.model is a Sequential of blocks
        # First N blocks are backbone (CSPDarknet), last 3 are detection head
        # TODO(integration): verify [:-3] slicing matches your ultralytics version
        full_layers = list(yolo_full.model.model.children())

        if len(full_layers) < feature_level + 1:
            raise ValueError(
                f"feature_level={feature_level} exceeds available layers "
                f"({len(full_layers)})"
            )

        # Keep only backbone layers up to feature_level
        self.layers = nn.ModuleList(full_layers[: feature_level + 1])

        # Determine output shape via dummy forward
        self.output_dim, self.num_spatial_tokens = self._infer_output_shape()

    def _infer_output_shape(self) -> tuple[int, int]:
        """Run dummy forward to find output dim and spatial size."""
        self.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            out = self._forward_backbone(dummy)
        # out: (1, C, H', W')
        _, c, h, w = out.shape
        return c, h * w

    def _forward_backbone(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self._forward_backbone(images)  # (B, C, H', W')
        b, c, h, w = features.shape
        return features.permute(0, 2, 3, 1).reshape(b, h * w, c)


class YOLOBBoxEncoder(VisualPriorEncoder):
    """Structured features: top-K bounding boxes from YOLO inference.

    For each detected object, encodes (x, y, w, h, confidence, class_one_hot).
    Outputs a fixed-size vector by padding to topk slots.

    Note: only works well if YOLO actually detects target objects in your scene.
    For SO-101 cube, you may need YOLO-World with text prompt or finetune YOLO
    on a few annotated frames. See docs for guidance.

    Output: (B, topk, feature_dim) — fixed-size sequence of box features.
    """

    NUM_COCO_CLASSES = 80
    BOX_DIM = 4 + 1  # (x, y, w, h, conf)

    def __init__(self, model_name: str = "yolov8n", topk: int = 5):
        super().__init__()
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise ImportError(
                'YOLO encoder requires `ultralytics`. Install: pip install -e ".[yolo]"'
            ) from e

        self.topk = topk
        self.yolo = YOLO(f"{model_name}.pt")

        # Each box: (x, y, w, h, conf) + class one-hot (80 for COCO)
        per_box_dim = self.BOX_DIM + self.NUM_COCO_CLASSES
        self.output_dim = per_box_dim
        self.num_spatial_tokens = topk

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Run YOLO inference, extract top-K boxes, encode."""
        # YOLO expects uint8 or [0,1] tensors of shape (B, 3, H, W)
        # We're given normalized ImageNet tensors — denormalize for YOLO
        # then YOLO will re-normalize internally
        # Simpler: pass denormalized [0, 1]
        from torchvision.transforms.functional import normalize

        mean = torch.tensor([0.485, 0.456, 0.406], device=images.device).view(
            1, 3, 1, 1
        )
        std = torch.tensor([0.229, 0.224, 0.225], device=images.device).view(
            1, 3, 1, 1
        )
        denorm = (images * std + mean).clamp(0, 1)

        B = denorm.shape[0]
        device = images.device
        out = torch.zeros(B, self.topk, self.output_dim, device=device)

        # YOLO predict expects list of images (B, C, H, W) handled internally
        results = self.yolo.predict(denorm, verbose=False)

        for i, res in enumerate(results):
            if res.boxes is None or len(res.boxes) == 0:
                continue
            # Sort by confidence
            confs = res.boxes.conf
            idx = torch.argsort(confs, descending=True)[: self.topk]

            xywhn = res.boxes.xywhn[idx]  # normalized (x, y, w, h)
            conf = res.boxes.conf[idx].unsqueeze(-1)  # (K, 1)
            cls = res.boxes.cls[idx].long()  # (K,)

            box_features = torch.cat([xywhn, conf], dim=-1)  # (K, 5)
            class_onehot = torch.zeros(
                len(cls), self.NUM_COCO_CLASSES, device=device
            )
            class_onehot.scatter_(1, cls.unsqueeze(1), 1.0)

            features = torch.cat(
                [box_features, class_onehot], dim=-1
            )  # (K, 5+80)
            out[i, : features.shape[0]] = features

        return out
