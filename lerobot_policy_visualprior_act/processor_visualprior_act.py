"""Pre/post processors for VisualPriorACTPolicy.

Required by LeRobot plugin convention — function name must match pattern
`make_{policy_name}_pre_post_processors` so that LeRobot can find it automatically.

The pre-processor pipeline handles:
- Image resize to a canonical size (224x224 by default — matches what most
  pretrained encoders expect: ImageNet, SAM2, DINOv2, YOLO)
- ImageNet-style normalization (mean/std)

The post-processor pipeline handles output un-normalization. In practice,
LeRobot's Normalize/Unnormalize modules inside the policy do most of the
work — the processor here is the OUTER pipeline that runs before/after.

TODO(integration): The exact imports below may need adjustment depending on
lerobot version. If `lerobot.processor` doesn't expose these names, search
your local lerobot/src/lerobot/processor/ for the right module paths.
"""

from __future__ import annotations

from typing import Any

# TODO(integration): adjust imports based on your lerobot version
try:
    from lerobot.processor import (
        PolicyProcessorPipeline,
        NormalizeProcessor,
        ResizeImageProcessor,
    )
    from lerobot.processor.types import PolicyAction
except ImportError as e:
    # Fallback for older lerobot versions
    raise ImportError(
        "Could not import LeRobot processor modules. Your lerobot version may "
        "use different module paths. See docs/INTEGRATION_NOTES.md for guidance."
    ) from e

from .configuration_visualprior_act import VisualPriorACTConfig


# ImageNet normalization — used by most pretrained models
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def make_visualprior_act_pre_post_processors(
    config: VisualPriorACTConfig,
) -> tuple[
    "PolicyProcessorPipeline[dict[str, Any], dict[str, Any]]",
    "PolicyProcessorPipeline[PolicyAction, PolicyAction]",
]:
    """Create pre and post processor pipelines.

    The function name MUST be `make_{policy_name}_pre_post_processors` —
    this is how LeRobot finds the right processors for this policy type.
    """
    pre = PolicyProcessorPipeline(
        [
            # Resize all images to canonical 224x224.
            # If your camera natively records 640x480 or 1280x720, this
            # downsamples first before encoder forward.
            ResizeImageProcessor(target_size=(224, 224)),
            # ImageNet-style normalization. Most pretrained encoders expect this.
            # For encoders that have their own normalization (SAM2 has a different
            # one), the encoder class internally re-normalizes from ImageNet to
            # its own statistics — handled inside the encoder, not here.
            NormalizeProcessor(
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD,
                apply_to=["observation.images"],
            ),
        ]
    )

    post = PolicyProcessorPipeline([])  # action un-normalization done inside policy

    return pre, post
