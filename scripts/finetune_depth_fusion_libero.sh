export SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
export PROJECT_DIR="$( cd -- "$( dirname -- "$SCRIPT_DIR" )" &> /dev/null && pwd )"
cd "$PROJECT_DIR"
export PYTHONPATH="$PYTHONPATH:$PROJECT_DIR"

# Set this to the directory containing:
#   z_depth_train_shard0_model4_manifest.json
#   z_depth_train_shard0_model4_part00000.pt
#   ...
export LIBERO_DEPTH_FUSION_DIR="${LIBERO_DEPTH_FUSION_DIR:-$PROJECT_DIR/data/libero_depth_fusion}"
export LIBERO_DEPTH_FUSION_MANIFEST="${LIBERO_DEPTH_FUSION_MANIFEST:-$LIBERO_DEPTH_FUSION_DIR/z_depth_train_shard0_model4_manifest.json}"
export OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/outputs/depth_fusion_libero}"

python3 -u -m latent_pretraining.depth_fusion.train_depth_fusion \
    --data_dir "$LIBERO_DEPTH_FUSION_DIR" \
    --manifest "$LIBERO_DEPTH_FUSION_MANIFEST" \
    --output_dir "$OUTPUT_DIR" \
    --rgb_feature_key "auto" \
    --depth_feature_key "auto" \
    --action_key "auto" \
    --epochs 20 \
    --batch_size 256 \
    --lr 1e-4 \
    --weight_decay 1e-4 \
    --val_fraction 0.05 \
    --hidden_dim 2048 \
    --dropout 0.1
