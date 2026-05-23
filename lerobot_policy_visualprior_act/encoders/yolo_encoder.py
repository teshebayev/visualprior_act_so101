"""YOLO encoders (Family C — task-supervised priors).

Two variants:
- YOLOBackboneEncoder: features from intermediate layer of YOLOv8 CSPDarknet.
  Bypasses YOLO's built-in preprocessing, so we apply ImageNet normalization
  manually (YOLOv8 was pretrained on COCO with ImageNet stats).

- YOLOBBoxEncoder: structured features from detection output (top-K boxes).
  Uses YOLO.predict() which handles preprocessing internally — we pass raw
  [0, 1] tensors to it directly.

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
    Pretrained on COCO object detection. Input expected in [0, 1].

    Output: (B, N_spatial, C) — spatial token sequence.
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

        yolo_full = YOLO(f"{model_name}.pt")
        full_layers = list(yolo_full.model.model.children())

        if len(full_layers) < feature_level + 1:
            raise ValueError(
                f"feature_level={feature_level} exceeds available layers "
                f"({len(full_layers)})"
            )

        self.layers = nn.ModuleList(full_layers[: feature_level + 1])

        # YOLOv8 expects ImageNet-normalized inputs when called directly
        # (we bypass yolo.predict() which would normalize internally).
        self._register_imagenet_norm()

        # Determine output shape via dummy forward.
        # NOTE: we run the dummy through the SAME normalization the real path
        # uses, otherwise BN running stats would be inferred from off-distribution
        # data. Buffers are on CPU here; the dummy is on CPU too — fine.
        self.output_dim, self.num_spatial_tokens = self._infer_output_shape()

    def _infer_output_shape(self) -> tuple[int, int]:
        self.eval()
        with torch.no_grad():
            # Use mid-range values so BN doesn't see degenerate zeros input
            dummy = torch.full((1, 3, 224, 224), 0.5)
            x = self._imagenet_normalize(dummy)
            out = self._forward_backbone(x)
        _, c, h, w = out.shape
        return c, h * w

    def _forward_backbone(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # Input in [0, 1] → ImageNet-normalize → backbone.
        x = self._imagenet_normalize(images)
        features = self._forward_backbone(x)  # (B, C, H', W')
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

        per_box_dim = self.BOX_DIM + self.NUM_COCO_CLASSES
        self.output_dim = per_box_dim
        self.num_spatial_tokens = topk

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Run YOLO inference, extract top-K boxes, encode."""
        # We get images in [0, 1] from the policy (preprocessor is IDENTITY for
        # VISUAL). YOLO.predict() handles its own normalization internally, so
        # we pass [0, 1] tensors directly. NO denormalization needed.
        B = images.shape[0]
        device = images.device
        out = torch.zeros(B, self.topk, self.output_dim, device=device)

        # Clamp to [0, 1] defensively (e.g. floating-point drift from resize).
        clipped = images.clamp(0.0, 1.0)
        results = self.yolo.predict(clipped, verbose=False)

        for i, res in enumerate(results):
            if res.boxes is None or len(res.boxes) == 0:
                continue
            confs = res.boxes.conf
            idx = torch.argsort(confs, descending=True)[: self.topk]

            xywhn = res.boxes.xywhn[idx]  # normalized (x, y, w, h) in [0, 1]
            conf = res.boxes.conf[idx].unsqueeze(-1)  # (K, 1)
            cls = res.boxes.cls[idx].long()  # (K,)

            box_features = torch.cat([xywhn, conf], dim=-1)  # (K, 5)
            class_onehot = torch.zeros(
                len(cls), self.NUM_COCO_CLASSES, device=device
            )
            class_onehot.scatter_(1, cls.unsqueeze(1), 1.0)

            features = torch.cat(
                [box_features, class_onehot], dim=-1
            )  # (K, 85)
            out[i, : features.shape[0]] = features

        return out
