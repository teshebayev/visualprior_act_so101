# Troubleshooting

Типичные проблемы при работе с плагином и их решения.

---

## Установка

### `ImportError: lerobot is not installed`

```bash
cd ~/projects/lerobot && pip install -e .
```

### `ImportError: vector_quantize_pytorch`

```bash
pip install -e ".[vae]"
```

### `ImportError: ultralytics` / `segmentation_models_pytorch` / `transformers`

```bash
pip install -e ".[yolo]"          # для YOLO
pip install -e ".[unet]"          # для U-Net
pip install -e ".[foundation]"    # для SAM2/DINOv2
```

---

## Plugin не регистрируется

### `Unknown policy type: visualprior_act`

LeRobot не нашёл наш plugin. Проверь:

```bash
# 1. Пакет установлен?
pip show lerobot_policy_visualprior_act

# 2. Импорт работает?
python -c "from lerobot_policy_visualprior_act import VisualPriorACTConfig; print('OK')"

# 3. Регистрация сработала?
python -c "
from lerobot.configs import PreTrainedConfig
import lerobot_policy_visualprior_act  # это триггерит регистрацию
print(PreTrainedConfig._registered_classes if hasattr(PreTrainedConfig, '_registered_classes') else 'check internals')
"
```

Если пункты 1-2 проходят, но `lerobot-train` всё равно не видит — может быть
проблема с тем, что `lerobot-train` не импортирует наш плагин. **Решение:**

Создай файл `~/.lerobot/plugins.py` или сделай явный импорт в команде:

```bash
PYTHONPATH=$PYTHONPATH python -c "import lerobot_policy_visualprior_act" && \
    lerobot-train --policy.type=visualprior_act ...
```

Или (проще) добавь в свой shell rc:

```bash
# .bashrc
alias lerobot-train-vp='python -c "import lerobot_policy_visualprior_act" && lerobot-train'
```

---

## Forward не работает

### `NotImplementedError: _run_act_head must be adapted`

Это **ожидаемо** до того, как ты прочитаешь `docs/INTEGRATION_NOTES.md` и
реализуешь интеграцию под свою версию lerobot. **Это не баг плагина — это
явное место, требующее адаптации.**

### `AttributeError: 'ACT' has no attribute 'backbone'`

Имя визуального backbone в ACT отличается в твоей версии. Открой
`lerobot/policies/act/modeling_act.py` и найди, как называется атрибут с
ResNet. Возможные варианты: `vision_encoder`, `image_encoder`, `cnn_backbone`.
Добавь обработку этого имени в `_build_act_head`.

### `RuntimeError: shape mismatch` на первом forward

Размерности encoder output / projector / ACT input не совпадают. Запусти
encoder отдельно и проверь shape:

```python
from lerobot_policy_visualprior_act.encoders import VAEEncoder
import torch

enc = VAEEncoder(latent_dim=32)
x = torch.randn(2, 3, 224, 224)
print(enc(x).shape)  # должно быть (2, 32)
print(enc.output_dim, enc.num_spatial_tokens)  # 32, 1
```

Если shape другой — баг в encoder. Если как ожидается — проблема в projector
или в том, как ACT принимает tokens.

---

## VAE pretraining

### Loss не убывает

- Проверь lr (по умолчанию 1e-3 норм для VAE/β-VAE)
- Проверь что изображения в [0,1] range и mse_loss работает корректно
- Если кадры почти одинаковые (мало вариации) — увеличь датасет

### VQ-VAE: `codebook_usage` <10%

**Codebook collapse.** Возможные причины:
- Слишком большой codebook для маленького датасета — попробуй меньше (128, 256)
- commitment_weight слишком большой — попробуй 0.1 вместо 0.25
- Слишком маленький latent_dim — попробуй 64

Эмпирическое правило: codebook_size <= sqrt(num_frames) × 10 для стабильности.

### Reconstruction плохой

Это **не критично** для нашей задачи — главное чтобы encoder извлекал полезные
features для policy. Но если reconstruction совсем не работает, скорее всего
encoder тоже плохо учится — увеличь capacity backbone или epochs.

---

## Pretrained weights

### `Missing keys when loading pretrained encoder`

Это норма для большинства missing — мы загружаем только encoder, decoder
отброшен. Тревожно только если missing keys содержат encoder-специфичные слои
(не `decoder.*`).

### `RuntimeError: size mismatch for fc_mu.weight`

Размерность latent_dim при pretraining != при policy training. Должны
совпадать строго. Перепроверь, что одинаковый `--latent-dim` использовался
при pretraining и в config policy.

---

## YOLO

### `YOLO('yolov8n.pt')` качает модель долго / падает

Ultralytics кэширует модели в `~/.config/Ultralytics/`. Проверь интернет,
свободное место. Можно скачать .pt вручную с GitHub releases и положить в
рабочую директорию.

### `feature_level=4 exceeds available layers`

В новых версиях ultralytics структура `model.model` могла поменяться. Запусти:

```python
from ultralytics import YOLO
y = YOLO('yolov8n.pt')
print(list(y.model.model.children()))
```

Посмотри, сколько блоков. P4 обычно блок 4-5. Поправь `feature_level` в config.

### YOLO не детектит куб (для `yolo_bbox`)

COCO не содержит "cube". Опции:
- Использовать YOLO-World с text prompt "wooden cube"
- Дотюнить YOLO на 50-100 размеченных кадрах твоего датасета (~1 час разметки)
- Использовать generic класс (например "sports ball" хорошо детектит блочные объекты)

---

## SAM2 / DINOv2

### `AutoModel.from_pretrained` падает на SAM2

Твоя версия transformers не поддерживает SAM2. Нужна `transformers >= 4.42`
с правильной интеграцией. Проверь:

```bash
pip show transformers
```

Если меньше — `pip install -U transformers`.

### Out of memory с SAM2/DINOv2

Foundation модели большие. Опции:
- Перейти на меньшую модель: `facebook/sam2-hiera-tiny`, `facebook/dinov2-small`
- Уменьшить batch_size до 4 или 2
- Использовать torch.amp.autocast (bfloat16) для inference

### Странная нормализация

SAM2 и DINOv2 ожидают свои mean/std. Наш processor применяет ImageNet stats —
для DINOv2 это корректно, для SAM2 может быть не идеально. Если результаты
слабые, проверь документацию модели на HF и поправь normalization внутри
encoder класса.

---

## Real-robot eval

### `RuntimeError: select_action returns wrong shape`

Чек:
1. `policy.reset()` вызывается перед каждым новым эпизодом?
2. `self._action_queue` правильно деаллоцируется?
3. action_dim в датасете и в policy совпадает?

### Action chunk пустой после первого вызова

Bug: chunks не пополняются. В `select_action` проверь, что когда queue пуст,
вызывается `predict_action_chunk` и список заполняется.

---

## Производительность

### Тренировка медленная

- Batch size слишком маленький → увеличь до 16-32 если VRAM позволяет
- Foundation models (SAM2/DINOv2) тяжёлые → frozen + cache features оффлайн
- DataLoader bottleneck → увеличь `num_workers`

### Inference latency высокая (>50ms)

Для real-time контроля на 30 Hz нужно <33ms на forward. Тяжёлые varианты:
- SAM2: ~80ms на CPU, ~30ms на GPU → ок только с GPU и no batching
- DINOv2 ViT-S: ~20ms на GPU
- VAE / VQ-VAE: <5ms на GPU
- ResNet baseline: <10ms на GPU

Используй `--policy.use_amp=true` если lerobot версия поддерживает.

---

## Когда ничего не помогает

1. Проверь все TODO(integration) комментарии в коде — они помечают места,
   требующие адаптации.
2. Запусти tests/test_encoders.py — там видно, какие encoders работают.
3. Минимизируй: запусти на dummy data forward pass, не на реальном датасете.
4. Открой issue с конкретным error message и `git log -1` для lerobot.
