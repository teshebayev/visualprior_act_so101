# Integration Notes

**Read this BEFORE running `lerobot-train` with this plugin.**

This document tells you what parts of the code need adaptation to your specific
LeRobot version and how to do it.

---

## Why this is needed

The plugin is mostly self-contained, but there is **one critical integration
point** that depends on LeRobot's internal ACT implementation:

`modeling_visualprior_act.py::_run_act_head` — bypasses ACT's built-in ResNet
backbone and feeds our pre-computed visual tokens directly into the transformer.

Standard LeRobot ACT does this internally:

```
images -> backbone (ResNet-18) -> cam_features (B, C, H', W')
       -> spatial flatten + projection -> (B, N, dim_model)
       -> concat with state and z_act tokens
       -> transformer encoder -> transformer decoder -> action chunk
```

We want to skip step 1 (backbone) and inject our own `visual_tokens` at step 2.
The exact attribute names and method calls to do this depend on which version
of LeRobot you have installed.

---

## Step-by-step integration

### Step 1. Check your lerobot version

```bash
cd ~/projects/lerobot
git log -1 --oneline
git describe --tags --always
```

Note the commit hash or tag — useful if you need to rollback later.

### Step 2. Read modeling_act.py

```bash
less ~/projects/lerobot/src/lerobot/policies/act/modeling_act.py
```

You're looking for the **`class ACT`** (NOT `class ACTPolicy`) — this is the
internal nn.Module that contains the transformer. Read its `__init__` and
`forward` methods carefully.

Take note of:

1. **What attribute holds the visual backbone?**
   Common names: `self.backbone`, `self.vision_encoder`, `self.image_encoder`.

2. **How are camera features projected to `dim_model`?**
   Usually `self.encoder_img_feat_input_proj` or `self.input_proj` — a Conv2d
   or Linear that maps backbone output channels to `dim_model`.

3. **How is the state encoded as a token?**
   Usually `self.encoder_robot_state_input_proj` — a Linear from state_dim to dim_model.

4. **Where is the action-CVAE (`z_act`) encoder?**
   Usually `self.vae_encoder` — only used during training.

5. **What does forward return?**
   Likely a tuple: `(actions, (mu, logvar))` if `use_vae=True`, else
   `(actions, None)`.

### Step 3. Decide your integration strategy

Three options, from least to most invasive:

**Option A: Patch ACT.forward via batch dict (least invasive)**

Add an entry to the batch dict that ACT.forward should check. Then in our
`_run_act_head`, call `self.act_head(modified_batch)` and have a monkey-patched
ACT that uses our tokens if the entry is present.

Requires modifying ACT internals slightly (patching forward) but keeps
most of ACT logic intact.

**Option B: Subclass ACT (cleaner)**

Create `class VisualPriorACT(ACT)` that overrides only the `forward` method,
replacing the backbone-call with our visual_tokens. Reuse all other logic
(transformer, action-CVAE, decoder) unchanged.

This is what I'd recommend for production work. See `_act_subclass_example.py`
in this folder for a template.

**Option C: Self-contained head (most robust)**

Write a minimal transformer encoder+decoder ourselves and skip LeRobot's ACT
entirely. ~200-300 LoC. Maximum robustness to future LeRobot updates.

### Step 4. Implement `_run_act_head`

Replace the `NotImplementedError` in `modeling_visualprior_act.py::_run_act_head`
with actual code. Below is a template based on a typical lerobot ACT structure
(your version may differ — adjust accordingly).

```python
def _run_act_head(self, batch, visual_tokens):
    """visual_tokens: (B, N, dim_model) already projected by self.projector"""
    
    act = self.act_head  # Internal lerobot ACT instance
    
    # 1. Project state to a single token
    state = batch["observation.state"]
    if state.dim() == 3:  # (B, T, state_dim) -> (B, state_dim)
        state = state[:, -1]
    state_token = act.encoder_robot_state_input_proj(state)  # (B, dim_model)
    
    # 2. Action-CVAE encoder (training only)
    if self.training and act.config.use_vae and "action" in batch:
        # ACT encodes action sequence to z_act mean/logvar
        # The exact API varies — find it in modeling_act.py
        cls_token, mu, logvar = act.vae_encoder(state, batch["action"])
        std = torch.exp(0.5 * logvar)
        z_act = mu + std * torch.randn_like(std)
    else:
        # Inference: z_act = 0
        B = visual_tokens.shape[0]
        z_act = torch.zeros(
            B, act.config.latent_dim, device=visual_tokens.device
        )
        mu, logvar = None, None
    
    z_act_token = act.encoder_latent_input_proj(z_act)  # (B, dim_model)
    
    # 3. Assemble encoder input: [z_act_token, state_token, *visual_tokens]
    encoder_in = torch.cat(
        [
            z_act_token.unsqueeze(1),     # (B, 1, dim_model)
            state_token.unsqueeze(1),     # (B, 1, dim_model)
            visual_tokens,                # (B, N, dim_model)
        ],
        dim=1,
    )
    
    # 4. Run transformer encoder
    # ACT uses src_key_padding_mask=None typically
    pos_embed = act._build_positional_encoding(encoder_in)  # if exposed
    # or compute pos_embed manually based on length
    encoder_out = act.encoder(encoder_in + pos_embed)  # (B, 2+N, dim_model)
    
    # 5. Decoder
    # ACT uses learned query embeddings for action chunks
    query_embed = act.decoder_pos_embed.weight.unsqueeze(0).expand(B, -1, -1)
    decoder_out = act.decoder(query_embed, encoder_out)  # (B, chunk_size, dim_model)
    
    # 6. Action head
    actions_hat = act.action_head(decoder_out)  # (B, chunk_size, action_dim)
    
    return actions_hat, (mu, logvar) if mu is not None else None
```

**This template will NOT work as-is.** Specific names like
`encoder_robot_state_input_proj`, `encoder_latent_input_proj`, `vae_encoder`,
`decoder_pos_embed`, `action_head` may differ. Replace each with whatever
your lerobot version uses.

### Step 5. Verify M0 reproduces standard ACT

The critical sanity check:

```bash
# 1. Train standard ACT
lerobot-train --policy.type=act \
    --dataset.repo_id=... \
    --output_dir=outputs/standard_act \
    --seed=42 --steps=1000

# 2. Train M0 baseline through our plugin
lerobot-train --policy.type=visualprior_act \
    --policy.encoder=resnet18 \
    --dataset.repo_id=... \
    --output_dir=outputs/M0_baseline \
    --seed=42 --steps=1000

# 3. Compare training loss at each step — should be very close (<1% difference)
```

If losses diverge significantly, your `_run_act_head` is doing something
different from standard ACT, and you need to debug **before** running any
other experiments.

---

## Common pitfalls

### Pitfall 1: Image normalization mismatch

The processor normalizes images with ImageNet mean/std. If your VAE was
pretrained on un-normalized images (in [0, 1] range), the encoder will see
inputs in the wrong range during policy training.

**Fix:** ensure consistency. Either:
- Pretrain VAE with the same ImageNet normalization (recommended)
- Or, in `VAEEncoder.forward`, undo ImageNet normalization before the conv backbone

### Pitfall 2: Input resolution mismatch

Plugin assumes 224×224 inputs (the standard for ImageNet, SAM2, DINOv2, YOLO).
LeRobot may natively record at 480×640 or 1280×720.

**Fix:** the `ResizeImageProcessor` in `processor_visualprior_act.py` handles
this. Verify it's actually applied by checking actual image shape at forward.

### Pitfall 3: ACTConfig kwargs mismatch

In `modeling_visualprior_act.py::_build_act_head`, we construct `ACTConfig`
with specific kwargs. If your lerobot version's ACTConfig has different
parameter names (e.g. `dim_feedforward` -> `feedforward_dim`), this fails.

**Fix:** open `lerobot/policies/act/configuration_act.py`, copy the actual
parameter list, adjust our config to match.

### Pitfall 4: `self.backbone = nn.Identity()` doesn't disable backbone use

Setting backbone to Identity doesn't help if ACT.forward calls
`self.backbone(images)` somewhere — Identity will pass images through
unchanged, breaking downstream tensor shapes.

**Fix:** see Step 3 — actually need to either patch ACT.forward (Option A)
or override it (Option B).

### Pitfall 5: Visual token count expectation

Standard ACT typically expects ~50 visual tokens (7×7 from ResNet on 224×224
input). If your encoder produces a different number (e.g. VAE returns 1
token), the transformer's positional embeddings may be sized for 50 and
break.

**Fix:** the encoder either provides spatial tokens (preferred), or we
adjust ACT's pos_embed to expected length. The `num_spatial_tokens` attribute
on each encoder helps you check this.

---

## Subclass example

Here's a sketch of Option B (subclass ACT) — adapt to your lerobot version:

```python
# In your modeling_visualprior_act.py, replace _build_act_head with:

from lerobot.policies.act.modeling_act import ACT

class VisualPriorACT(ACT):
    """ACT that accepts pre-computed visual tokens instead of running its own backbone."""
    
    def forward(self, batch, visual_tokens=None):
        if visual_tokens is None:
            return super().forward(batch)
        
        # Replicate ACT.forward but skip backbone, use visual_tokens directly
        # ... (copy logic from lerobot ACT.forward, adapt)
        # ...

def _build_act_head(self, config):
    # ... build ACTConfig as before ...
    return VisualPriorACT(act_cfg)

def _run_act_head(self, batch, visual_tokens):
    return self.act_head(batch, visual_tokens=visual_tokens)
```

---

## Getting help

If integration is blocking you:

1. Run the encoder tests first: `pytest tests/test_encoders.py -v`.
   These don't require lerobot integration and verify encoders work.

2. Try Option C (self-contained head). Write a minimal transformer that
   doesn't depend on lerobot ACT. Slower to get to first experiment, but
   doesn't depend on getting Option A/B right.

3. Open an issue with your `git log -1` output and the first 100 lines of
   your `lerobot/policies/act/modeling_act.py::class ACT`. The exact integration
   code can then be written for your specific version.
