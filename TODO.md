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
    - [ ] (GPU) Run: python data/process_libero.py --libero_root datasets/libero_raw --output_dir datasets/lapa_libero

- [x] Write lapa_finetune/LAPA/scripts/finetune_libero_full.sh for training with 4 suite (shuffle data) and testing on test set using config following LAPA/scripts/finetune_real.sh
    - [ ] (GPU) Set absolute_path and run: ./scripts/finetune_libero_full.sh

- [ ] Create mocktest for lapa_finetune/LAPA/scripts/finetune_libero_full.sh
