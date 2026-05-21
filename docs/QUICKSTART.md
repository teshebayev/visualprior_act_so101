# Quickstart

Workflow первых дней после установки.

---

## День 0 — Установка

```bash
# 1. Убедись что lerobot установлен
cd ~/projects/lerobot
pip install -e .

# 2. Установи этот плагин
cd ~/projects/visualprior_act_so101/lerobot_policy_visualprior_act
pip install -e ".[vae]"   # минимум для начала; добавишь остальные позже

# 3. Smoke test — должны пройти все тесты encoders
pytest tests/test_encoders.py -v
```

Если encoder тесты падают — что-то с зависимостями (torch, torchvision).
Если тесты config падают — проблема с импортом lerobot.

---

## День 1 — Интеграция с твоей версией lerobot

**Это самый важный шаг.** Прочитай `docs/INTEGRATION_NOTES.md` целиком.

В двух словах: нужно реализовать `_run_act_head` в `modeling_visualprior_act.py`
под твою версию LeRobot ACT. Без этого `lerobot-train` упадёт с `NotImplementedError`.

После того как реализуешь, проверь sanity check:

```bash
# Тренируй стандартный ACT 1000 шагов
lerobot-train --policy.type=act \
    --dataset.repo_id=lerobot/aloha_sim_insertion_human \
    --output_dir=outputs/check_standard \
    --seed=42 --steps=1000

# Тренируй наш M0 baseline (resnet18 без bottleneck) 1000 шагов
lerobot-train --policy.type=visualprior_act \
    --policy.encoder=resnet18 \
    --dataset.repo_id=lerobot/aloha_sim_insertion_human \
    --output_dir=outputs/check_M0 \
    --seed=42 --steps=1000

# Loss curves должны быть очень похожи (<1% разницы)
```

Если разница больше — что-то в `_run_act_head` работает иначе, чем стандартный ACT.

---

## День 2 — Pretrain один VAE

```bash
pretrain-visual-encoder \
    --dataset-repo-id=your_org/so101_pickplace_v1 \
    --encoder-type=vqvae \
    --latent-dim=32 \
    --codebook-size=512 \
    --grid-size=4 \
    --output-path=./pretrained/vqvae_c512_d32.safetensors \
    --num-epochs=50
```

Что отслеживать в логах:
- `loss` должна убывать
- `codebook_usage` для VQ-VAE должна расти и стабилизироваться >30%
- если usage <10% — codebook collapse, перезапустить с меньшим commitment_weight

---

## День 3 — Тренировка M0 и M7 на реальных данных

```bash
# M0: baseline
lerobot-train \
    --policy.type=visualprior_act \
    --policy.encoder=resnet18 \
    --dataset.repo_id=your_org/so101_pickplace_v1 \
    --output_dir=outputs/M0_baseline_seed42 \
    --seed=42 \
    --steps=100000

# M7: VQ-VAE finetuned
lerobot-train \
    --policy.type=visualprior_act \
    --policy.encoder=vqvae \
    --policy.vae_pretrained_path=./pretrained/vqvae_c512_d32.safetensors \
    --policy.freeze_encoder=false \
    --dataset.repo_id=your_org/so101_pickplace_v1 \
    --output_dir=outputs/M7_vqvae_finetune_seed42 \
    --seed=42 \
    --steps=100000
```

---

## День 4+ — Real-robot evaluation

```bash
lerobot-record \
    --robot.type=so101 \
    --robot.id=so101_main \
    --policy.path=outputs/M0_baseline_seed42 \
    --num_episodes=30 \
    --eval_dir=eval/M0_test_seen
```

Прогнать через все 5 starting positions (P1-P5), записать в session log.

---

## Дальше

См. `scripts/train_all_models.sh` — шаблон для запуска всей matrix Phase 2
из главного research plan.
