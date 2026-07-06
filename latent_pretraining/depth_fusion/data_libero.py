import bisect
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset


RGB_FEATURE_CANDIDATES = (
    "z_rgb_feature_input",
    "z_rgb_feature",
    "rgb_feature",
)
DEPTH_FEATURE_CANDIDATES = (
    "z_depth_feature_pred",
    "z_depth_feature_pred_model7_1",
    "z_depth_feature_gt",
    "z_depth_feature",
    "depth_feature",
)
ACTION_CANDIDATES = (
    "action_vector",
    "raw_actions",
    "action",
    "actions",
)
IMAGE_KEY_CANDIDATES = (
    "image",
    "image_path",
    "image_paths",
    "rgb_path",
    "rgb_paths",
    "file_name",
    "file_names",
    "id",
    "ids",
)


def load_manifest(manifest_path: Optional[Path]) -> Dict[str, object]:
    if manifest_path is None:
        return {}
    return json.loads(Path(manifest_path).read_text())


def _load_manifest_parts(manifest_path: Path, data_dir: Path) -> List[Path]:
    manifest = load_manifest(manifest_path)
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


def _first_present_key(shard: Dict[str, object], candidates: Sequence[str]) -> Optional[str]:
    for key in candidates:
        if key in shard:
            return key
    return None


def _manifest_key(manifest: Dict[str, object], candidates: Sequence[str]) -> Optional[str]:
    for key in candidates:
        value = manifest.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def resolve_feature_keys(
    shard: Dict[str, object],
    manifest: Optional[Dict[str, object]] = None,
    rgb_feature_key: str = "auto",
    depth_feature_key: str = "auto",
    action_key: str = "auto",
    image_key: str = "auto",
) -> Tuple[str, str, str, Optional[str]]:
    manifest = manifest or {}

    if rgb_feature_key == "auto":
        rgb_feature_key = (
            _manifest_key(manifest, ("rgb_feature_key_input", "rgb_feature_key"))
            or _first_present_key(shard, RGB_FEATURE_CANDIDATES)
        )
    if depth_feature_key == "auto":
        depth_feature_key = (
            _manifest_key(manifest, ("feature_key", "feature_key_pred", "depth_feature_key"))
            or _first_present_key(shard, DEPTH_FEATURE_CANDIDATES)
        )
    if action_key == "auto":
        action_key = _manifest_key(manifest, ("action_key",)) or _first_present_key(shard, ACTION_CANDIDATES)
    if image_key == "auto":
        image_key = _manifest_key(manifest, ("image_key", "image_path_key")) or _first_present_key(shard, IMAGE_KEY_CANDIDATES)

    missing = []
    if not rgb_feature_key or rgb_feature_key not in shard:
        missing.append(("rgb_feature_key", rgb_feature_key, RGB_FEATURE_CANDIDATES))
    if not depth_feature_key or depth_feature_key not in shard:
        missing.append(("depth_feature_key", depth_feature_key, DEPTH_FEATURE_CANDIDATES))
    if not action_key or action_key not in shard:
        missing.append(("action_key", action_key, ACTION_CANDIDATES))
    if image_key not in (None, "") and image_key not in shard:
        image_key = None

    if missing:
        available = ", ".join(sorted(shard.keys()))
        details = "; ".join(
            f"{name}={value!r}, tried {list(candidates)}"
            for name, value, candidates in missing
        )
        raise KeyError(f"Could not resolve required shard keys: {details}. Available keys: {available}")

    return rgb_feature_key, depth_feature_key, action_key, image_key


def _value_at(value: object, index: int) -> object:
    if torch.is_tensor(value):
        return value[index]
    if isinstance(value, (list, tuple)):
        return value[index]
    return value


class LiberoDepthFusionDataset(Dataset):
    """Dataset for colleague-provided LIBERO feature shards.

    Each .pt shard must contain RGB features, depth features, and actions. Keys can
    be supplied explicitly or discovered from the manifest/shard with "auto".
    """

    def __init__(
        self,
        part_files: Iterable[Path],
        manifest: Optional[Dict[str, object]] = None,
        rgb_feature_key: str = "auto",
        depth_feature_key: str = "auto",
        action_key: str = "auto",
        image_key: str = "auto",
        preload: bool = True,
        return_metadata: bool = False,
    ):
        self.part_files = [Path(p) for p in part_files]
        if not self.part_files:
            raise ValueError("No .pt shard files were found.")

        first_shard = torch.load(self.part_files[0], map_location="cpu")
        self.rgb_feature_key, self.depth_feature_key, self.action_key, self.image_key = resolve_feature_keys(
            first_shard,
            manifest=manifest,
            rgb_feature_key=rgb_feature_key,
            depth_feature_key=depth_feature_key,
            action_key=action_key,
            image_key=image_key,
        )
        self.preload = preload
        self.return_metadata = return_metadata

        self._lengths = []
        rgb_features = []
        depth_features = []
        actions = []
        image_names = []
        for part_file in self.part_files:
            shard = torch.load(part_file, map_location="cpu")
            self._validate_keys(shard, part_file)
            self._lengths.append(int(shard[self.action_key].shape[0]))
            if self.preload:
                rgb_features.append(shard[self.rgb_feature_key].float().clone())
                depth_features.append(shard[self.depth_feature_key].float().clone())
                actions.append(shard[self.action_key].float().clone())
                if self.return_metadata and self.image_key is not None:
                    image_names.extend(list(shard[self.image_key]))

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
            if self.return_metadata and self.image_key is not None:
                self._preloaded["image_name"] = image_names

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
            item = {
                "rgb_feature": self._preloaded["rgb_feature"][index],
                "depth_feature": self._preloaded["depth_feature"][index],
                "action": self._preloaded["action"][index],
            }
            if self.return_metadata and "image_name" in self._preloaded:
                item["image_name"] = self._preloaded["image_name"][index]
            return item

        part_index = bisect.bisect_right(self._cumulative, index)
        previous = 0 if part_index == 0 else self._cumulative[part_index - 1]
        local_index = index - previous
        shard = self._load_shard(part_index)

        item = {
            "rgb_feature": shard[self.rgb_feature_key][local_index].float(),
            "depth_feature": shard[self.depth_feature_key][local_index].float(),
            "action": shard[self.action_key][local_index].float(),
        }
        if self.return_metadata and self.image_key is not None:
            item["image_name"] = _value_at(shard[self.image_key], local_index)
        return item
