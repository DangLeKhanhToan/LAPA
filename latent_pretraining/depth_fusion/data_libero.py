import bisect
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import torch
from torch.utils.data import Dataset


def _load_manifest_parts(manifest_path: Path, data_dir: Path) -> List[Path]:
    manifest = json.loads(manifest_path.read_text())
    parts = []
    for part in manifest.get("parts", []):
        source_path = Path(part["path"])
        local_path = data_dir / source_path.name
        parts.append(local_path if local_path.exists() else source_path)
    return parts


def discover_part_files(data_dir: Path, manifest_path: Optional[Path] = None) -> List[Path]:
    if manifest_path is not None:
        return _load_manifest_parts(manifest_path, data_dir)
    return sorted(data_dir.glob("*_part*.pt"))


class LiberoDepthFusionDataset(Dataset):
    """Dataset for colleague-provided LIBERO feature shards.

    Each .pt shard is expected to contain:
      - z_rgb_feature_input: Tensor[N, 4096]
      - one selected depth feature tensor: Tensor[N, 1024]
      - action_vector: Tensor[N, 7]
    """

    def __init__(
        self,
        part_files: Iterable[Path],
        rgb_feature_key: str = "z_rgb_feature_input",
        depth_feature_key: str = "z_depth_feature_pred_model7_1",
        action_key: str = "action_vector",
        preload: bool = True,
    ):
        self.part_files = [Path(p) for p in part_files]
        if not self.part_files:
            raise ValueError("No .pt shard files were found.")

        self.rgb_feature_key = rgb_feature_key
        self.depth_feature_key = depth_feature_key
        self.action_key = action_key
        self.preload = preload

        self._lengths = []
        rgb_features = []
        depth_features = []
        actions = []
        for part_file in self.part_files:
            shard = torch.load(part_file, map_location="cpu")
            self._validate_keys(shard, part_file)
            self._lengths.append(int(shard[self.action_key].shape[0]))
            if self.preload:
                rgb_features.append(shard[self.rgb_feature_key].float().clone())
                depth_features.append(shard[self.depth_feature_key].float().clone())
                actions.append(shard[self.action_key].float().clone())

        self._cumulative = []
        running = 0
        for length in self._lengths:
            running += length
            self._cumulative.append(running)

        self._cached_part_index = None
        self._cached_shard = None
        self._preloaded = None
        if self.preload:
            self._preloaded = {
                "rgb_feature": torch.cat(rgb_features, dim=0),
                "depth_feature": torch.cat(depth_features, dim=0),
                "action": torch.cat(actions, dim=0),
            }

    def _validate_keys(self, shard: Dict[str, object], part_file: Path) -> None:
        missing = [
            key
            for key in (self.rgb_feature_key, self.depth_feature_key, self.action_key)
            if key not in shard
        ]
        if missing:
            raise KeyError(f"{part_file} is missing required key(s): {missing}")

    def __len__(self) -> int:
        return self._cumulative[-1]

    def _load_shard(self, part_index: int) -> Dict[str, object]:
        if self._cached_part_index != part_index:
            self._cached_shard = torch.load(self.part_files[part_index], map_location="cpu")
            self._cached_part_index = part_index
        return self._cached_shard

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        if index < 0 or index >= len(self):
            raise IndexError(index)

        if self._preloaded is not None:
            return {
                "rgb_feature": self._preloaded["rgb_feature"][index],
                "depth_feature": self._preloaded["depth_feature"][index],
                "action": self._preloaded["action"][index],
            }

        part_index = bisect.bisect_right(self._cumulative, index)
        previous = 0 if part_index == 0 else self._cumulative[part_index - 1]
        local_index = index - previous
        shard = self._load_shard(part_index)

        return {
            "rgb_feature": shard[self.rgb_feature_key][local_index].float(),
            "depth_feature": shard[self.depth_feature_key][local_index].float(),
            "action": shard[self.action_key][local_index].float(),
        }
