#!/usr/bin/env python3
"""HTTP server for online Stage-2.5 depth feature extraction.

This process intentionally imports the Stage-2.5 bundle from an external
directory first. The bundle contains its own ``latent_pretraining`` package, so
it should run as a separate process from the LAPA-Depth policy server.
"""

import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from PIL import Image


def import_stage25_bundle(bundle_dir: str):
    bundle = Path(bundle_dir).resolve()
    laq_dir = bundle / "laq"
    if not (laq_dir / "rollout_stage25_model4.py").exists():
        raise FileNotFoundError(f"Cannot find laq/rollout_stage25_model4.py under {bundle}")
    for path in (str(bundle), str(laq_dir)):
        if path in sys.path:
            sys.path.remove(path)
        sys.path.insert(0, path)

    from rollout_stage25_model4 import (  # noqa: E402
        Stage25RolloutFeatureExtractor,
        build_lapa,
        build_model4,
    )

    return Stage25RolloutFeatureExtractor, build_lapa, build_model4


def load_rgb(path: str, image_size: int) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB").resize((image_size, image_size)), dtype=np.uint8)


def load_depth(depth_value: Any) -> Any:
    if isinstance(depth_value, str) and depth_value.lower().endswith(".npy"):
        return np.load(depth_value)
    return depth_value


class DepthAnythingV2Runner:
    """Small adapter for a DepthAnythingV2 checkpoint trained on Sth2Sth."""

    MODEL_CONFIGS = {
        "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
        "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
        "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
        "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
    }

    def __init__(self, repo_dir: str, checkpoint: str, encoder: str, input_size: int, device: str):
        import torch

        repo = Path(repo_dir).resolve()
        if not (repo / "depth_anything_v2").exists():
            raise FileNotFoundError(
                f"Cannot find depth_anything_v2 package under {repo}. "
                "Set --depth_anything_repo_dir to the cloned DepthAnythingV2 repo."
            )
        if encoder not in self.MODEL_CONFIGS:
            raise ValueError(f"Unknown DepthAnythingV2 encoder {encoder!r}; choose one of {sorted(self.MODEL_CONFIGS)}")
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))

        from depth_anything_v2.dpt import DepthAnythingV2  # noqa: E402

        self.input_size = input_size
        self.device = torch.device(device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = DepthAnythingV2(**self.MODEL_CONFIGS[encoder])
        state = torch.load(checkpoint, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        self.model.load_state_dict(state, strict=True)
        self.model.to(self.device).eval()

    @staticmethod
    def _to_uint16_depth(depth: np.ndarray) -> np.ndarray:
        depth = np.asarray(depth, dtype=np.float32)
        finite = np.isfinite(depth)
        if not finite.any():
            return np.zeros(depth.shape, dtype=np.uint16)
        lo = float(depth[finite].min())
        hi = float(depth[finite].max())
        if hi <= lo:
            return np.zeros(depth.shape, dtype=np.uint16)
        depth = (depth - lo) / (hi - lo)
        depth = np.clip(depth, 0.0, 1.0)
        return (depth * 65535.0).astype(np.uint16)

    def infer(self, rgb: np.ndarray) -> np.ndarray:
        with __import__("torch").no_grad():
            depth = self.model.infer_image(rgb, self.input_size)
        return self._to_uint16_depth(depth)


class Stage25FeatureServer:
    def __init__(self, args):
        Stage25RolloutFeatureExtractor, build_lapa, build_model = import_stage25_bundle(args.stage25_bundle_dir)

        self.image_size = args.image_size
        self.model_name = args.model_name
        self.depth_anything: Optional[DepthAnythingV2Runner] = None
        if args.depth_anything_checkpoint:
            self.depth_anything = DepthAnythingV2Runner(
                repo_dir=args.depth_anything_repo_dir,
                checkpoint=args.depth_anything_checkpoint,
                encoder=args.depth_anything_encoder,
                input_size=args.depth_anything_input_size,
                device=args.depth_anything_device,
            )
        print(
            json.dumps(
                {
                    "stage25_server": {
                        "bundle": str(Path(args.stage25_bundle_dir).resolve()),
                        "model_name": args.model_name,
                        "model_checkpoint": args.model_checkpoint,
                        "original_lapa_checkpoint": args.original_lapa_checkpoint,
                        "depth_anything_checkpoint": args.depth_anything_checkpoint,
                        "depth_anything_encoder": args.depth_anything_encoder,
                    }
                }
            ),
            flush=True,
        )

        lapa = build_lapa(
            tokens_per_delta=args.tokens_per_delta,
            vqgan_checkpoint=args.vqgan_checkpoint,
            vocab_file=args.vocab_file,
            load_checkpoint=args.original_lapa_checkpoint,
            image_size=args.image_size,
            multi_image=args.multi_image,
            seed=args.seed,
            mesh_dim=args.mesh_dim,
            dtype=args.dtype,
            load_llama_config=args.load_llama_config,
        )
        model = build_model(
            checkpoint=args.model_checkpoint,
            dim=args.dim,
            image_size=args.image_size,
            patch_size=args.patch_size,
            spatial_depth=args.spatial_depth,
            dim_head=args.dim_head,
            heads=args.heads,
            code_seq_len=args.code_seq_len,
            z_rgb_feature_dim=args.z_rgb_feature_dim,
            z_depth_feature_dim=args.z_depth_feature_dim,
            predict_token_features=bool(args.predict_token_features),
            strict=bool(args.strict),
        )
        self.extractor = Stage25RolloutFeatureExtractor(
            lapa=lapa,
            model4=model,
            image_size=args.image_size,
            depth_scale=args.depth_scale,
            repeat_depth_to_3ch=bool(args.repeat_depth_to_3ch),
        )

    def feature(self, payload: Dict[str, Any]):
        try:
            rgb_path = payload["image"]
            instruction = payload["instruction"]

            rgb = load_rgb(rgb_path, self.image_size)
            if payload.get("depth_image") is not None:
                depth = load_depth(payload["depth_image"])
                depth_source = "payload"
            elif self.depth_anything is not None:
                depth = self.depth_anything.infer(rgb)
                depth_source = "depth_anything_v2"
            else:
                raise ValueError(
                    "No depth_image was provided and DepthAnythingV2 is not configured. "
                    "Set --depth_anything_checkpoint for rollout-time depth generation."
                )
            out = self.extractor.step(rgb, instruction, depth)
            z_depth = out["z_depth_feature_pred"]
            z_rgb = out["z_rgb_feature"]

            result = {
                "z_depth_feature_pred": z_depth.detach().cpu().float().numpy().tolist(),
                "z_depth_shape": list(z_depth.shape),
                "model_name": self.model_name,
                "depth_source": depth_source,
            }
            if payload.get("return_debug", False):
                result["z_rgb_shape"] = list(z_rgb.shape)
                result["depth_shape"] = list(np.asarray(depth).shape)
            return JSONResponse(result)
        except Exception:
            traceback.print_exc()
            return JSONResponse({"error": traceback.format_exc()}, status_code=500)


def parse_args():
    parser = argparse.ArgumentParser(description="Online Stage-2.5 feature server.")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=32821)
    parser.add_argument("--stage25_bundle_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="model4", choices=("model2", "model4"))
    parser.add_argument("--model_checkpoint", type=str, required=True)

    parser.add_argument("--original_lapa_checkpoint", type=str, required=True)
    parser.add_argument("--vqgan_checkpoint", type=str, required=True)
    parser.add_argument("--vocab_file", type=str, required=True)
    parser.add_argument("--tokens_per_delta", type=int, default=4)
    parser.add_argument("--multi_image", type=int, default=1)
    parser.add_argument("--mesh_dim", type=str, default="1,1,1,1")
    parser.add_argument("--dtype", type=str, default="bf16")
    parser.add_argument("--load_llama_config", type=str, default="7b")
    parser.add_argument("--seed", type=int, default=1234)

    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--dim", type=int, default=1024)
    parser.add_argument("--patch_size", type=int, default=32)
    parser.add_argument("--spatial_depth", type=int, default=8)
    parser.add_argument("--dim_head", type=int, default=64)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--code_seq_len", type=int, default=4)
    parser.add_argument("--z_rgb_feature_dim", type=int, default=4096)
    parser.add_argument("--z_depth_feature_dim", type=int, default=1024)
    parser.add_argument("--predict_token_features", type=int, default=0, choices=(0, 1))
    parser.add_argument("--strict", type=int, default=1, choices=(0, 1))
    parser.add_argument("--repeat_depth_to_3ch", type=int, default=1, choices=(0, 1))
    parser.add_argument("--depth_scale", type=float, default=65535.0)

    parser.add_argument("--depth_anything_repo_dir", type=str, default="")
    parser.add_argument("--depth_anything_checkpoint", type=str, default="")
    parser.add_argument("--depth_anything_encoder", type=str, default="vitl", choices=("vits", "vitb", "vitl", "vitg"))
    parser.add_argument("--depth_anything_input_size", type=int, default=518)
    parser.add_argument("--depth_anything_device", type=str, default="auto")
    return parser.parse_args()


def main():
    args = parse_args()
    server = Stage25FeatureServer(args)
    app = FastAPI()
    app.post("/feature")(server.feature)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
