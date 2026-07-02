- [x] setup envs (/scratch/users/create/smrvmdo/venvs/lapa-depth/)
    - [x] LAPA
    - [x] LIBERO 
        - issue 0: Running setup.py install for egl-probe did not run successfully
        - fix 0: python -m pip install "cmake<4" ninja
        - issue 1: failed-wheel-build-for-install
        - fix 1: 
            ```
            python -m pip install -U pip setuptools wheel
            python -m pip uninstall -y cmake
            python -m pip install cmake==3.22.0 ninja scikit-build

            which cmake
            cmake --version

            python -m pip install egl-probe --no-build-isolation --no-cache-dir -v
            ```
        - issue 2: ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts
        - fix 2: create 2 venvs


- [x] download pretrain LAPA 7B openx (/home/users/create/smrvmdo/scratch/projects/lapa_finetune/LAPA/lapa_checkpoints)

- [x] setup raw datasets (/home/users/create/smrvmdo/scratch/projects/lapa_finetune/LAPA/datasets/LIBERO)
    - [x] download repo LIBERO
    - [x] download all LIBERO subdata
        - [x] libero_object     7.0G
        - [x] libero_spatial    5.9G
        - [x] libero_goal       6.0G
        - [x] libero_10
        - [x] libero_90

- [x] processing LIBERO for training LAPA (/home/users/create/smrvmdo/scratch/projects/lapa_finetune/LAPA/datasets/lapa_libero)
    - script: data/process_libero.py
    - [x] split data for training and testing (handled inside process_libero.py)
    ---
    | Suite          |              Train |                Test |            Số task |               Train episodes |               Test episodes |
    | -------------- | -----------------: | ------------------: | -----------------: | ---------------------------: | --------------------------: |
    | LIBERO-Spatial | demo 0–44 mỗi task | demo 45–49 mỗi task |                 10 |                          450 |                          50 |
    | LIBERO-Object  | demo 0–44 mỗi task | demo 45–49 mỗi task |                 10 |                          450 |                          50 |
    | LIBERO-Goal    | demo 0–44 mỗi task | demo 45–49 mỗi task |                 10 |                          450 |                          50 |
    | LIBERO-100     |  toàn bộ LIBERO-90 |   toàn bộ LIBERO-10 | 90 train / 10 test | khoảng 4500 nếu 50 demo/task | khoảng 500 nếu 50 demo/task |
    ---
    - [x] (GPU) Run: python data/process_libero.py --libero_root datasets/libero_raw --output_dir datasets/lapa_libero

- [x] Write lapa_finetune/LAPA/scripts/finetune_libero_full.sh for training with 4 suite (shuffle data) and testing on test set using config following LAPA/scripts/finetune_real.sh
    - [ ] (GPU) Set absolute_path and run: ./scripts/finetune_libero_full.sh
        - issue 0: AttributeError: module 'pyarrow' has no attribute 'PyExtensionType'. Did you mean: 'ExtensionType'?
        - fix 0: pip install -U datasets pyarrow
        - issue 1: jaxlib.xla_extension.XlaRuntimeError: FAILED_PRECONDITION: DNN library initialization failed. Look at the errors above for more details.
        - fix 1: 
        ```
        pip uninstall -y jax jaxlib jax-cuda12-plugin jax-cuda12-pjrt

        pip install "jax==0.4.23" \
        "jaxlib==0.4.23+cuda12.cudnn89" \
        -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
        ```

- [x] Create mocktest for lapa_finetune/LAPA/scripts/finetune_libero_full.sh

- [x] Rollout eval on test set with video + success rate per suite and overall
    - client/server design (2 venvs): model server = latent_pretraining.deploy (lapa-depth),
      sim rollout client = eval/eval_libero_rollout.py (LIBERO venv)
    - orchestrated by scripts/eval_libero.sh; videos saved under outputs/eval_libero/<suite>/<task>/
    - results.json holds per-task, per-suite, and overall success rate
    - NOTE: also fixed data/process_libero.py instruction prompt to match the inference sampler
      ("What action should the robot take to `{task}`") -> data must be regenerated + retrained
    - [ ] (GPU) After a full training run, set FINETUNED_CHECKPOINT and run: ./scripts/eval_libero.sh
        - needs EGL offscreen rendering: MUJOCO_GL=egl PYOPENGL_PLATFORM=egl (set in script)
        - libero editable install is broken -> repo added to PYTHONPATH in the script
        - checkpoint saved as outputs/finetune_libero_full/streaming_params (+ _5/_10/... milestones);
          eval default (params::.../streaming_params) is correct
        - issue 0: ModuleNotFoundError: No module named 'json_numpy' (deploy.py server, lapa-depth)
        - fix 0: pip install json_numpy draccus  (added to requirements.txt)
        - issue 1: torch>=2.6 torch.load weights_only=True breaks libero get_task_init_states
        - fix 1: eval_libero_rollout.py patches torch.load to default weights_only=False
        - issue 2: EGL_NOT_INITIALIZED -> this node has NO GL/EGL/OSMesa rendering stack
          (empty /usr/share/glvnd/egl_vendor.d, no libEGL_nvidia, no libOSMesa). Infra, not code.
        - render options (pick one):
          a) singularity/4.1.5 module -> run client in a container with `--nv` (mounts NVIDIA
             GL libs -> GPU EGL). RECOMMENDED on this NVIDIA HPC.
          b) OSMesa software render: need libOSMesa.so on RENDER_LD_LIBRARY_PATH, then
             MUJOCO_GL=osmesa ./scripts/eval_libero.sh  (conda-forge mesa 26 dropped OSMesa;
             need the right mesa build). eval_libero.sh already supports this via env vars.
          c) ask sysadmin to expose NVIDIA EGL vendor ICD or a mesa module.
        - NOTE: $HOME is 100% full (50G/50G) -> move caches (conda/pip/wandb) to /scratch
        - issue 3: server crashed at checkpoint load -> shape mismatch on the action embedding/head.
          deploy.py defaults update_llama_config action_vocab_size=256, but finetune_libero_full.sh
          trained with 219 (= # action bins = action_bins.csv column count). Client then hung on
          retries (60 x 10s) and failed with "server unreachable".
        - fix 3 (done, no GPU): eval_libero.sh now passes --update_llama_config with
          action_vocab_size derived from the action_bins.csv header (219), overridable via
          $ACTION_VOCAB_SIZE / $UPDATE_LLAMA_CONFIG.
        - [ ] (GPU) Smoke eval to confirm fix 3: SUITES="libero_spatial" N_EVAL_PER_TASK=1 ./scripts/eval_libero.sh
            - server must reach uvicorn startup + print "image_path: ..." (no shape-mismatch traceback)
            - client must print per-episode success|fail lines and write outputs/eval_libero/results.json
            - open one outputs/eval_libero/libero_spatial/<task>/ep0_*.mp4: arm should move purposefully
              toward the target (libero_spatial is in-distribution -> expect some successes)
        - [ ] (GPU) Full eval (defaults: 4 suites x 5 eps/task): ./scripts/eval_libero.sh
            - check SUMMARY + results.json per-suite / overall success rates
            - if ~0 everywhere despite server loading + arm moving: try --flip_for_model (image
              orientation). NOT expected: training images are upside-down and the client already matches.
