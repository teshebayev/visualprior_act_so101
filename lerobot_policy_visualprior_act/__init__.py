"""Visual Prior ACT policy plugin for LeRobot.

Provides ACT-based policy with replaceable visual front-end:
- ResNet-18 baseline (M0) and linear bottleneck control (M1)
- VAE / β-VAE / VQ-VAE (M2-M7)
- YOLO / U-Net (M8-M11)
- SAM2 / DINOv2 (M12-M13)

All variants share an identical ACT transformer head with internal
action-CVAE preserved unchanged. Visual encoders project to a fixed
d_model via a learned MLP to ensure fair comparison.
"""

try:
    import lerobot  # noqa: F401
except ImportError as e:
    raise ImportError(
        "lerobot is not installed. Install lerobot >= 0.4.0 first:\n"
        "    cd ~/projects/lerobot && pip install -e ."
    ) from e

from .configuration_visualprior_act import VisualPriorACTConfig
from .modeling_visualprior_act import VisualPriorACTPolicy
from .processor_visualprior_act import make_visualprior_act_pre_post_processors

__version__ = "0.1.0"

__all__ = [
    "VisualPriorACTConfig",
    "VisualPriorACTPolicy",
    "make_visualprior_act_pre_post_processors",
]
