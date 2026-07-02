# Usage — Fine-tune & Evaluate LAPA on LIBERO

> ⚠️ **STATUS — EVALUATION IN PROGRESS / BEING FIXED.**
> The closed-loop **success rate is not available yet**. The eval pipeline is still being brought
> up: a checkpoint-loading bug in `eval_libero.sh` (`action_vocab_size`) has been fixed, but the
> GPU + singularity rollout has **not been run** yet. The numbers in §6 are **training metrics only**,
> from a run that stopped at step 1545/2000 (saved checkpoint = **step 1000**).

Requires 2 venvs: `lapa-depth` (JAX — data / fine-tune / deploy server) and `LIBERO`
(mujoco — rollout eval).

---

## Quick start

### 1. Prepare data

Convert LIBERO HDF5 demos → JPEG images + JSONL under `datasets/lapa_libero/`.

```bash
/scratch/users/create/smrvmdo/venvs/lapa-depth/bin/python data/process_libero.py \
    --libero_root datasets/libero_raw \
    --output_dir  datasets/lapa_libero
```

### 2. Fine-tune  (GPU, 4×)

Fine-tune LAPA-7B on `all_train.jsonl`; checkpoint saved to `outputs/finetune_libero_full/streaming_params`.

```bash
cd /scratch/users/create/smrvmdo/projects/lapa_finetune/LAPA
bash scripts/finetune_libero_full.sh
```

### 3. Evaluate success rate  (GPU + singularity)

Starts the deploy server + runs the LIBERO rollout; writes one video per episode and
`outputs/eval_libero/results.json`.

```bash
cd /scratch/users/create/smrvmdo/projects/lapa_finetune/LAPA

# Quick smoke test (1 suite, 1 episode/task)
SUITES="libero_spatial" N_EVAL_PER_TASK=1 bash scripts/eval_libero.sh

# Full evaluation (4 suites × 5 episodes/task)
bash scripts/eval_libero.sh
```

---

## 1. Environments (two separate venvs — conflicting deps)

| venv | Interpreter | Used for |
|---|---|---|
| `lapa-depth` (JAX/Flax) | `/scratch/users/create/smrvmdo/venvs/lapa-depth/bin/python` | data processing, fine-tune, deploy server |
| `LIBERO` (mujoco/robosuite) | `/scratch/users/create/smrvmdo/venvs/LIBERO/bin/python` | rollout client during eval |

Every script needs `absolute_path` = project root
(`/scratch/users/create/smrvmdo/projects/lapa_finetune/LAPA`). `eval_libero.sh` derives it
automatically; `finetune_libero_full.sh` **hard-codes it at line 7** — edit it for another machine.

---

## 2. Data — `data/process_libero.py`

Key args:
- `--discretize_bins 256` — number of action discretization bins; `pd.qcut` drops duplicate edges ⇒ up to **219** bins in practice.
- `--camera agentview_rgb` — third-person camera (128×128).
- `--suites` — default: all 5 suites.

Produced under `datasets/lapa_libero/`:
- `images/{suite}/{task}/demo_{i}/step_{t}.jpg`
- `{suite}_train.jsonl`, `{suite}_test.jsonl`
- `all_train.jsonl` — **849,278** records (shuffled); `all_test.jsonl` — **158,340** records.
- `action_bins.csv` — bin edges the deploy server uses to decode token → action (**219 columns**).

Train records per source: `libero_90` 669,043 · `libero_object` 67,053 · `libero_goal` 57,201 ·
`libero_spatial` 55,981. Splits: spatial/object/goal use demos 0–44 (train) / 45–49 (test);
`libero_90` all = train; `libero_10` all = test.

---

## 3. Fine-tune config — `scripts/finetune_libero_full.sh`

**3a. Default — same as `finetune_real.sh`:**

| Group | Param | Value |
|---|---|---|
| Init | `--load_checkpoint` | `params::lapa_checkpoints/params` (LAPA-7B-openx) |
| Modality | `--modality` | `vision,action,delta` |
| Parallelism | `--mesh_dim` | `!-1,4,1,1` (dp,fsdp,sp,tp — fsdp = 4 GPU) |
| Optim | AdamW `lr = end_lr` / warmup / decay / wd | `2e-5` / 0 / 100 / 0 |
| Sequence | `seq_length` | 384 |
| Tokens | `n_tokens_per_action` / `n_tokens_per_delta` | 7 / 4 |
| Image | `img_aug` / `max_n_frames` | True / 1 |
| Vocab | `--llama.delta_vocab_size` | 8 |
| Schedule | `total_steps` / `save_model_freq` | 2000 / 0 |

**3b. Modify — changed for LIBERO:**

| Param | Base (`finetune_real`) | LIBERO | Reason |
|---|---|---|---|
| `action_vocab_size` | 256 | **219** | actual number of LIBERO action bins (`qcut` drops duplicates) |
| `batch_size` | 128 | **400** | use the 4 GPUs |
| `dataset path` + `image_absolute_path` | `data/` | `datasets/lapa_libero/` | LIBERO data |
| `save_milestone_freq` | 2000 | **1000** | save an extra mid-run milestone |
| `logger.online` | True | False | WandB offline |

**Files produced** under `outputs/finetune_libero_full/`:
- `streaming_params` — model params (no optimizer state), latest overwritable copy → used for eval/deploy.
- `streaming_params_{step}` — permanent snapshot at that step (`_1000` = real milestone).
- `metadata*.pkl` — `step` + all FLAGS + `llama_config` (incl. `action_vocab_size=219`).
- `dataset*.pkl` — data-loader state for resume.
- `wandb/` — offline logs.

> `save_optimizer_state=False` ⇒ **no** `streaming_train_state`; to continue training you must reload params.

---

## 4. Evaluation config — `scripts/eval_libero.sh`

Closed-loop: the script **starts the deploy server** (`lapa-depth` venv, JAX) then **runs the rollout
client** (`LIBERO` venv, mujoco) inside a singularity `--nv` container for EGL rendering.
Needs **GPU + singularity**.

Overridable env vars:

| Var | Default | Meaning |
|---|---|---|
| `FINETUNED_CHECKPOINT` | `params::outputs/finetune_libero_full/streaming_params` | checkpoint to evaluate |
| `ACTION_SCALE_FILE` | `datasets/lapa_libero/action_bins.csv` | bin table to decode actions |
| `ACTION_VOCAB_SIZE` | derived from `action_bins.csv` = **219** | **must match training** (see §5) |
| `SUITES` | `libero_spatial libero_object libero_goal libero_10` | suites to evaluate |
| `N_EVAL_PER_TASK` | 5 | rollouts per task |
| `OUTPUT_DIR` | `outputs/eval_libero` | where results are written |
| `PORT` | 32820 | deploy server port |
| `USE_SINGULARITY` | 1 | 1 = render in a `--nv` container (recommended) |
| `CUDA_VISIBLE_DEVICES` | 0 | GPU for the deploy server |

**Outputs:**
- `outputs/eval_libero/{suite}/{task}/ep{i}_{success|fail}.mp4` — one video per episode.
- `outputs/eval_libero/results.json` — success rate per task / per suite / overall.

> Test split is chosen automatically: spatial/object/goal use init states 45–49 (held out);
> libero_10 uses all (tasks never trained on).

---

## 5. Fixes in `eval_libero.sh` / `process_libero.py`

(Fine-tune config changes: see §3b.)

| File | Change | Base → New | Reason |
|---|---|---|---|
| `eval_libero.sh` | add `--update_llama_config` (`action_vocab_size=219`) | (missing) → present | FIX checkpoint load (see below) |
| `data/process_libero.py` | prompt + gripper | — | prompt matches the inference sampler; LIBERO gripper `[-1,1]` binarized to `[-1.5,0,1.5]` |

**Why `action_vocab_size` is critical:** it sets the size of the action-token embedding & head
(`delta_llama_action.py:320,439`). Training used 219 but `deploy.py` defaults to 256 ⇒ the server
crashes at load, and the client retries ~10 min then reports "server unreachable". Fixed in
`eval_libero.sh` (derives 219 from `action_bins.csv`); `deploy.py` left unchanged since 256 is still
correct for `finetune_real`/`finetune_simpler`.

---

## 6. Current training results

> Training metrics only — **not** the success rate (see status banner at top).

Training metrics (offline WandB, main run):

| Step | action_loss | action_acc (top-1 / 219 bins) | grad_norm |
|---:|---:|---:|---:|
| 0 | 5.4225 | 0.25 % | 4.50 |
| 100 | 3.6647 | 35.2 % | 4.26 |
| 500 | 3.1650 | 37.4 % | 3.52 |
| **1000** (checkpoint) | **3.0410** | **38.5 %** | 2.60 |
| 1545 (last log) | 3.0378 | 37.1 % | 2.08 |

Metric meaning & formulas (source: `tux/loss.py`, `tux/stats.py`, `train.py:356-411`):

- **`action_loss`** = cross-entropy (natural log, *nats*) over action tokens only, normalized per
  sequence then averaged over the batch:
  `-(1/B)·Σ_i ( Σ_t valid[i,t]·log p[i,t](y[i,t]) ) / L[i]`.
  This is the **entire** training loss (`loss = action_loss`); delta/vision/text losses are logged
  only, not backpropagated. Random baseline `ln(219) ≈ 5.39` (≈ the step-0 value).
- **`action_acc` (top-1 / 219 bins)** = fraction of action tokens whose `argmax` over the 219 classes
  matches the target bin:
  `(1/B)·Σ_i ( Σ_t valid[i,t]·1{argmax=y} ) / L[i]`. Random baseline `1/219 ≈ 0.46 %`. This is
  per-token top-1 — **not** a 7-token exact match and **not** the success rate.
- **`grad_norm` / `param_norm`** = global L2 norm `sqrt(Σ_p Σ_j x²)` over gradients / params.
  grad 4.5→2.1 = stable convergence (no clipping); param ≈ 3552 stable (no blow-up/collapse).

**Assessment:**
- The main run stopped at **step 1545/2000**; the only completed milestone is **step 1000** ⇒
  `streaming_params` = model at **step 1000** (~0.47 epoch over 849k records — **less than 1 epoch**).
- Loss plateaued (5.42 → 3.04 from ~step 500–1000); training is stable.
- **Real success rate not available yet** — run Quick start step 3 / §4 (`eval_libero.sh`). Since the
  checkpoint is at step 1000 (< 1 epoch), the success rate may still be low; if needed, re-run
  fine-tune to 2000 steps (or more epochs) then re-evaluate.
