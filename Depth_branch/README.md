# Stage 2.5 Rollout: RGB feature -> Depth feature

Bundle này nối 2 script gốc lại thành 1 pipeline chạy trực tiếp trong vòng lặp
rollout (không cần ghi JSONL/`.pt` ra đĩa giữa 2 bước như pipeline batch cũ):

```
rgb_image + instruction ─[LAPA]─> z_rgb_feature ─┐
                                                   ├─[Model4]─> z_depth_feature_pred
                        depth_image ──────────────┘
```

1. `latent_pretraining/inference_update_jsonl_train.py` (LAPA, JAX):
   `rgb_image + instruction -> z_rgb_feature`
2. `laq/test_ssv2_25_model4_no_gt.py` (Model4, PyTorch):
   `depth_image + z_rgb_feature -> z_depth_feature_pred`

File mới `laq/rollout_stage25_model4.py` **import trực tiếp 2 file trên**
(không copy/sửa logic bên trong), chỉ thêm:
- `build_lapa(...)` / `build_model4(...)`: dựng lại đúng 2 model như trong
  `main()` của 2 file gốc.
- `Stage25RolloutFeatureExtractor`: 1 class với `.step(rgb_image, instruction,
  depth_image) -> dict` — gọi 1 lần cho mỗi timestep của rollout.

## Cấu trúc thư mục (giữ nguyên khi giải nén)

```
stage25_rollout_bundle/
├── requirements.txt
├── laq/
│   ├── rollout_stage25_model4.py      <- entry point mới (merge)
│   ├── test_ssv2_25_model4_no_gt.py   <- nguyên bản, không sửa
│   └── laq_model/
│       ├── __init__.py
│       ├── attention.py
│       └── latent_action_quantization_stage25_feature_model4.py
└── latent_pretraining/
    ├── __init__.py
    ├── inference_update_jsonl_train.py <- nguyên bản, không sửa
    ├── sampler_latent_pretrain.py
    ├── delta_llama.py
    ├── llama.py
    ├── ring_attention.py
    └── vqgan.py
```

Quan trọng: `laq/` và `latent_pretraining/` phải là 2 package **cùng cấp**
(sibling). `rollout_stage25_model4.py` tự thêm thư mục cha (repo root) và
thư mục `laq/` vào `sys.path` khi import, nên chạy từ thư mục nào cũng được,
miễn giữ đúng cấu trúc cây thư mục ở trên.

## Cài đặt

```bash
pip install -r requirements.txt
# + cần cài jax[cuda12] phù hợp bản CUDA của máy, xem comment đầu requirements.txt
```

Đây là 2 stack dependency khác nhau chạy chung 1 process:
- LAPA: JAX/Flax/transformers==4.29.2/tux/albumentations...
- Model4: torch/opencv-python/einops/beartype/tqdm (không cần transformers)

Không cần `pip install` theo `laq/setup.py` — file đó kéo theo 1 stack lớn
hơn nhiều (tensorflow, transformers==4.40.1, dlimp...) dùng cho các script
train/test khác trong `laq/`, không cần cho việc chạy Model4 inference.

## Checkpoint cần có (không nằm trong bundle — tự copy riêng, dung lượng lớn)

- LAPA: `--vqgan_checkpoint`, `--vocab_file` (tokenizer.model), `--load_checkpoint`
  (streaming_params, dùng tiền tố `params::` như script gốc)
- Model4: `--model4_checkpoint`

## Cách 1 — test nhanh qua CLI (1 sample)

```bash
cd laq
python rollout_stage25_model4.py \
  --rgb_image /path/to/frame.jpg \
  --depth_image /path/to/depth.png \
  --instruction "pick up the cup" \
  --vqgan_checkpoint /path/lapa_checkpoints/vqgan \
  --vocab_file /path/lapa_checkpoints/tokenizer.model \
  --load_checkpoint params::/path/lapa_checkpoints/streaming_params_22485 \
  --model4_checkpoint /path/model4.65000.pt \
  --output_pt /tmp/step_output.pt
```

In ra shape của `z_rgb_feature` và `z_depth_feature_pred`, và lưu cả 2 +
`latent_action` vào `--output_pt` nếu cần kiểm tra lại.

## Cách 2 — dùng trong vòng lặp rollout (LIBERO hoặc simulator khác)

```python
import sys
sys.path.insert(0, "laq")  # hoặc chạy script của bạn từ trong thư mục laq/

from rollout_stage25_model4 import (
    build_lapa,
    build_model4,
    Stage25RolloutFeatureExtractor,
)

lapa = build_lapa(
    tokens_per_delta=4,
    vqgan_checkpoint=".../vqgan",
    vocab_file=".../tokenizer.model",
    load_checkpoint="params::.../streaming_params_22485",
)
model4 = build_model4(checkpoint=".../model4.65000.pt")

extractor = Stage25RolloutFeatureExtractor(lapa, model4)

obs = env.reset()
instruction = env.get_instruction()

for t in range(max_steps):
    rgb = obs["agentview_rgb"]      # HxWx3 uint8
    depth = obs["agentview_depth"]  # HxW uint16/float — mảng trong RAM, không cần ghi file

    out = extractor.step(rgb, instruction, depth)
    z_depth_feature = out["z_depth_feature_pred"]  # dùng cho policy / bước tiếp theo

    action = policy(z_depth_feature, ...)
    obs, reward, done, info = env.step(action)
    if done:
        break
```

`depth_image` trong `.step()` nhận cả 2 dạng: đường dẫn file (giống script
gốc), hoặc mảng numpy đã có sẵn trong bộ nhớ (tiện cho rollout online).

## Không đổi hành vi 2 file gốc

`inference_update_jsonl_train.py` và `test_ssv2_25_model4_no_gt.py` được copy
y nguyên, không sửa dòng nào. Pipeline batch cũ (quét folder ảnh -> JSONL ->
`.pt` shard -> manifest) vẫn chạy độc lập bình thường nếu vẫn cần dùng để
tạo dataset offline.
