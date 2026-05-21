# lerobot_policy_visualprior_act

LeRobot plugin для исследования визуальных priors на SO-101. Заменяет визуальный
front-end ACT policy на encoders из 4 семейств: self-supervised reconstructive
(VAE/β-VAE/VQ-VAE), task-supervised (YOLO/U-Net), foundation models (SAM2/DINOv2),
плюс baseline (ResNet-18) и control (linear bottleneck).

---

## ⚠ Прочитай ПЕРЕД установкой

Этот пакет содержит **минимум одно место**, требующее адаптации под актуальную
версию LeRobot — интеграция с внутренним ACT transformer. Подробно описано в
`docs/INTEGRATION_NOTES.md`. Сначала прочитай его.

В коде помечены `TODO(integration):` комментарии. Не игнорируй их.

---

## Установка

### 1. Опциональные зависимости

Encoders разделены по группам — устанавливай только те, что нужны:

```bash
# Минимум (baseline + VAE family)
pip install -e ".[vae]"

# Если будешь использовать YOLO
pip install -e ".[yolo]"

# Если будешь использовать U-Net
pip install -e ".[unet]"

# Foundation models (SAM2, DINOv2)
pip install -e ".[foundation]"

# Всё сразу
pip install -e ".[all]"

# Dev
pip install -e ".[dev,all]"
```

### 2. Проверка установки

```bash
# Plugin должен зарегистрироваться
python -c "from lerobot_policy_visualprior_act import VisualPriorACTConfig; print('OK')"

# Smoke tests
pytest tests/
```

---

## Структура

```
lerobot_policy_visualprior_act/
├── configuration_visualprior_act.py    # PreTrainedConfig + регистрация
├── modeling_visualprior_act.py         # PreTrainedPolicy
├── processor_visualprior_act.py        # pre/post processors
├── encoders/
│   ├── base.py                          # абстрактный интерфейс
│   ├── resnet_baseline.py               # M0 baseline, M1 linear bottleneck
│   ├── vae_family.py                    # M2-M7: VAE / β-VAE / VQ-VAE
│   ├── yolo_encoder.py                  # M8-M9: YOLO backbone
│   ├── unet_encoder.py                  # M10-M11: U-Net encoder
│   ├── sam2_encoder.py                  # M12: SAM2
│   └── dinov2_encoder.py                # M13: DINOv2
├── pretraining/
│   └── cli.py                           # pretrain-visual-encoder CLI
└── utils/
    └── projector.py
```

---

## Использование

### Pretrain VAE-семьи (только если используешь B-family)

```bash
pretrain-visual-encoder \
    --dataset-repo-id=your_org/so101_pickplace \
    --encoder-type=vqvae \
    --latent-dim=32 \
    --codebook-size=512 \
    --output-path=./pretrained/vqvae_c512_d32.safetensors \
    --num-epochs=50
```

### Train policy через стандартный lerobot CLI

```bash
# M0 baseline
lerobot-train \
    --policy.type=visualprior_act \
    --policy.encoder=resnet18 \
    --dataset.repo_id=your_org/so101_pickplace \
    --output_dir=outputs/M0_baseline

# M7 VQ-VAE frozen
lerobot-train \
    --policy.type=visualprior_act \
    --policy.encoder=vqvae \
    --policy.vae_pretrained_path=./pretrained/vqvae_c512_d32.safetensors \
    --policy.freeze_encoder=true \
    --dataset.repo_id=your_org/so101_pickplace \
    --output_dir=outputs/M7_vqvae_frozen

# Полный список вариантов в scripts/train_all_models.sh
```

### Real-robot evaluation

Стандартный LeRobot workflow:

```bash
lerobot-record \
    --robot.type=so101 \
    --policy.path=outputs/M7_vqvae_frozen \
    --num_episodes=30
```

---

## Где это лежит на диске

**Не клади внутрь форка lerobot.** Этот пакет ставится через pip и работает
рядом с upstream lerobot. Рекомендованная структура:

```
~/projects/
├── lerobot/                              # официальный repo, git clone
└── visualprior_act_so101/
    ├── lerobot_policy_visualprior_act/   # этот пакет
    ├── pretrained/                       # веса VAE/β-VAE/VQ-VAE
    ├── outputs/                          # чекпойнты после lerobot-train
    └── ...
```

---

## Дальше

- `docs/INTEGRATION_NOTES.md` — что нужно адаптировать под твою версию LeRobot
- `docs/QUICKSTART.md` — workflow первых шагов
- `docs/TROUBLESHOOTING.md` — типичные ошибки

## License

Apache 2.0
