"""Pre/post processors for VisualPriorACTPolicy (lerobot >= 0.4.0 API).

Required by LeRobot plugin convention — function name must match pattern
`make_{policy_name}_pre_post_processors` so that LeRobot finds it automatically.

Mirrors the structure of lerobot's own `make_act_pre_post_processors`, with
one addition: an `ImageCropResizeProcessorStep` that resizes camera frames
to 224x224 (what all our visual encoders — VAE family, YOLO, U-Net, SAM2,
DINOv2 — expect).

Pipeline order matters:
  pre:  rename -> add_batch_dim -> device -> resize -> normalize
  post: unnormalize -> device(cpu)

Resize is placed AFTER device transfer so that interpolation happens on GPU,
and BEFORE normalization (ImageNet mean/std is applied on the final 224x224
tensor, not on the original resolution).
"""

from __future__ import annotations

from typing import Any

import torch

from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    RenameObservationsProcessorStep,
    UnnormalizerProcessorStep,
    policy_action_to_transition,
    transition_to_policy_action,
)
from lerobot.utils.constants import (
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)

from .configuration_visualprior_act import VisualPriorACTConfig


def make_visualprior_act_pre_post_processors(
    config: VisualPriorACTConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Build the pre- and post-processing pipelines for VisualPriorACTPolicy.

    Args:
        config: Policy configuration.
        dataset_stats: Dataset statistics (mean/std) used by NormalizerProcessorStep.
            Pass `dataset.meta.stats` from a LeRobotDataset.

    Returns:
        (pre_pipeline, post_pipeline) — both are PolicyProcessorPipeline instances
        ready to be plugged into lerobot training/eval infrastructure.
    """

    # NOTE: Mirrors stock ACT processor (see lerobot/policies/act/processor_act.py).
    # Resize to 224x224 happens INSIDE the policy in _encode_all_cameras —
    # not here. The HIL ImageCropResizeProcessorStep is designed for a
    # different code path and breaks on non-image observation keys.
    input_steps = [
        # No-op rename (kept for parity with stock ACT processor; future-proof
        # if we ever need to alias camera keys).
        RenameObservationsProcessorStep(rename_map={}),
        AddBatchDimensionProcessorStep(),
        DeviceProcessorStep(device=config.device),
        NormalizerProcessorStep(
            features={**config.input_features, **config.output_features},
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
            device=config.device,
        ),
    ]

    output_steps = [
        UnnormalizerProcessorStep(
            features=config.output_features,
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
        DeviceProcessorStep(device="cpu"),
    ]

    return (
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
            steps=input_steps,
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
        ),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=output_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )
