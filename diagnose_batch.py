"""Diagnose batch shapes after the preprocessor pipeline.

Run this to see exactly what shape tensors are arriving at policy.forward().
Then we know whether to fix the policy, the processor, or the dataset config.

Usage:
    python diagnose_batch.py rtx409011/ep_120
"""

import sys
from pprint import pprint

import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot_policy_visualprior_act import (
    VisualPriorACTConfig,
    make_visualprior_act_pre_post_processors,
)
from lerobot.configs.types import FeatureType, PolicyFeature


def main():
    if len(sys.argv) < 2:
        print("Usage: python diagnose_batch.py <repo_id>")
        sys.exit(1)
    repo_id = sys.argv[1]

    print(f"=== Loading dataset {repo_id} ===")
    ds = LeRobotDataset(repo_id)

    print(f"  Features in dataset:")
    for k, v in ds.features.items():
        print(f"    {k}: {v}")

    print()
    print("=== Raw sample[0] shapes (BEFORE any processor) ===")
    sample = ds[0]
    for k, v in sample.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: shape={tuple(v.shape)}, dtype={v.dtype}")
        else:
            print(f"  {k}: type={type(v).__name__}, value={v}")

    print()
    print("=== Constructing config + processor ===")
    # Mimic what lerobot_train does to build features
    input_features = {}
    output_features = {}
    for k, ft in ds.features.items():
        if k == "action":
            output_features["action"] = PolicyFeature(
                type=FeatureType.ACTION, shape=ft.shape
            )
        elif k == "observation.state":
            input_features["observation.state"] = PolicyFeature(
                type=FeatureType.STATE, shape=ft.shape
            )
        elif k.startswith("observation.images"):
            input_features[k] = PolicyFeature(
                type=FeatureType.VISUAL, shape=ft.shape
            )

    print(f"  input_features: {input_features}")
    print(f"  output_features: {output_features}")

    cfg = VisualPriorACTConfig(
        input_features=input_features,
        output_features=output_features,
        encoder="vqvae",
        vae_pretrained_path="./pretrained/vqvae_c512_g4_d32.safetensors",
        push_to_hub=False,
    )

    preprocessor, _ = make_visualprior_act_pre_post_processors(
        cfg, dataset_stats=ds.meta.stats
    )

    print()
    print("=== sample[0] shapes AFTER preprocessor ===")
    processed = preprocessor(sample)
    for k, v in processed.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: shape={tuple(v.shape)}, dtype={v.dtype}, device={v.device}")
        else:
            print(f"  {k}: type={type(v).__name__}, value={v}")


if __name__ == "__main__":
    main()
