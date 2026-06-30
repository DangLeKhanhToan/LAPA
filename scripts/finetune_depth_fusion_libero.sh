export SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
export PROJECT_DIR="$( cd -- "$( dirname -- "$SCRIPT_DIR" )" &> /dev/null && pwd )"
cd "$PROJECT_DIR"
export PYTHONPATH="$PYTHONPATH:$PROJECT_DIR"

# Set this to the directory containing:
#   all_models_train_libero10_manifest.json
#   all_models_train_libero10_part00000.pt
#   ...
export LIBERO_DEPTH_FUSION_DIR="${LIBERO_DEPTH_FUSION_DIR:-$PROJECT_DIR/data/libero_depth_fusion}"
export OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/outputs/depth_fusion_libero}"

python3 -u -m latent_pretraining.depth_fusion.train_depth_fusion \
    --data_dir "$LIBERO_DEPTH_FUSION_DIR" \
    --manifest "$LIBERO_DEPTH_FUSION_DIR/all_models_train_libero10_manifest.json" \
    --output_dir "$OUTPUT_DIR" \
    --rgb_feature_key "z_rgb_feature_input" \
    --depth_feature_key "z_depth_feature_pred_model7_1" \
    --action_key "action_vector" \
    --epochs 20 \
    --batch_size 256 \
    --lr 1e-4 \
    --weight_decay 1e-4 \
    --val_fraction 0.05 \
    --hidden_dim 2048 \
    --dropout 0.1
