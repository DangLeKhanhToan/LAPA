"""Diagnostic telemetry for LIBERO rollouts.

The helpers use public observations where possible and fall back to the
robosuite task environment for object poses and exact two-finger grasp checks.
All fallbacks are recorded so downstream analysis can reject uncertain rows.
"""

import gzip
import json
import os

import numpy as np


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    return value


def unwrap_env(vector_env):
    """Return (ControlEnv, underlying LIBERO task env)."""
    wrapped = vector_env.envs[0] if hasattr(vector_env, "envs") else vector_env
    underlying = getattr(wrapped, "env", wrapped)
    return wrapped, underlying


def object_names(vector_env):
    _, underlying = unwrap_env(vector_env)
    names = []
    for attr in ("objects_dict", "obj_body_id"):
        value = getattr(underlying, attr, None)
        if isinstance(value, dict):
            names.extend(value.keys())
    return sorted(set(str(name) for name in names))


def object_positions(vector_env):
    _, underlying = unwrap_env(vector_env)
    body_ids = getattr(underlying, "obj_body_id", {})
    positions = {}
    for name in object_names(vector_env):
        try:
            body_id = body_ids[name]
            positions[name] = np.asarray(
                underlying.sim.data.body_xpos[body_id], dtype=np.float64
            ).copy()
        except Exception:
            continue
    return positions


def objects_of_interest(vector_env):
    wrapped, underlying = unwrap_env(vector_env)
    for owner in (wrapped, underlying):
        try:
            values = list(owner.obj_of_interest)
            if values:
                return [str(value) for value in values]
        except Exception:
            pass
    return []


def select_target_object(vector_env):
    """Select the source object; retain candidates and selection provenance."""
    positions = object_positions(vector_env)
    interests = objects_of_interest(vector_env)
    for name in interests:
        if name in positions:
            return name, interests, "first_obj_of_interest_with_body"
    if positions:
        name = sorted(positions)[0]
        return name, interests, "fallback_first_movable_object"
    return None, interests, "unresolved"


def grasped_objects(vector_env):
    """Return exact grasp contacts when the installed robosuite exposes them."""
    _, underlying = unwrap_env(vector_env)
    checker = getattr(underlying, "_check_grasp", None)
    robots = getattr(underlying, "robots", [])
    if checker is None or not robots:
        return [], False
    gripper = getattr(robots[0], "gripper", None)
    if gripper is None:
        return [], False

    grasped = []
    attempted = False
    for name in object_names(vector_env):
        try:
            obj = underlying.get_object(name)
            geoms = getattr(obj, "contact_geoms", None)
            if geoms is None:
                continue
            attempted = True
            if bool(checker(gripper=gripper, object_geoms=geoms)):
                grasped.append(name)
        except TypeError:
            try:
                attempted = True
                if bool(checker(gripper, geoms)):
                    grasped.append(name)
            except Exception:
                continue
        except Exception:
            continue
    return sorted(set(grasped)), attempted


def eef_position(obs):
    for key in ("robot0_eef_pos", "eef_pos"):
        if key in obs:
            value = np.asarray(obs[key], dtype=np.float64).reshape(-1)
            if value.size >= 3:
                return value[:3].copy()
    return None


class EpisodeDiagnostics:
    def __init__(
        self,
        vector_env,
        suite,
        task_id,
        episode,
        init_index,
        instruction,
        reference_suite=None,
        reference_positions=None,
        approach_threshold=0.10,
    ):
        self.env = vector_env
        self.approach_threshold = float(approach_threshold)
        self.initial_positions = object_positions(vector_env)
        self.reference_positions = reference_positions or {}
        target, candidates, method = select_target_object(vector_env)
        self.target = target
        self.previous_grasped = set()
        self.ever_grasped = set()
        self.drop_events = []
        self.min_current_target_distance = None
        self.min_original_target_distance = None
        self.grasp_check_available = False
        self.steps = []
        self.meta = {
            "schema_version": 1,
            "suite": suite,
            "reference_suite": reference_suite,
            "task_id": int(task_id),
            "episode": int(episode),
            "init_index": int(init_index),
            "instruction": instruction,
            "target_object": target,
            "target_candidates": candidates,
            "target_selection_method": method,
            "approach_threshold_m": self.approach_threshold,
            "initial_object_positions": self.initial_positions,
            "reference_object_positions": self.reference_positions,
        }

    @staticmethod
    def _distance(a, b):
        if a is None or b is None:
            return None
        return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))

    def observe(self, step, obs, action=None, success=False):
        positions = object_positions(self.env)
        eef = eef_position(obs)
        grasped, available = grasped_objects(self.env)
        self.grasp_check_available = self.grasp_check_available or available
        grasped_set = set(grasped)
        self.ever_grasped.update(grasped_set)

        current_target_position = self.initial_positions.get(self.target)
        original_target_position = self.reference_positions.get(self.target)
        current_distance = self._distance(eef, current_target_position)
        original_distance = self._distance(eef, original_target_position)
        if current_distance is not None:
            self.min_current_target_distance = (
                current_distance
                if self.min_current_target_distance is None
                else min(self.min_current_target_distance, current_distance)
            )
        if original_distance is not None:
            self.min_original_target_distance = (
                original_distance
                if self.min_original_target_distance is None
                else min(self.min_original_target_distance, original_distance)
            )

        released = sorted(self.previous_grasped - grasped_set)
        if released and not success:
            self.drop_events.append({"step": int(step), "objects": released})
        self.previous_grasped = grasped_set

        self.steps.append(
            {
                "step": int(step),
                "eef_position": eef,
                "action": action,
                "object_positions": positions,
                "grasped_objects": grasped,
                "success": bool(success),
            }
        )

    def finalize(self, success):
        target_grasped = self.target is not None and self.target in self.ever_grasped
        wrong_objects = sorted(name for name in self.ever_grasped if name != self.target)
        current_target_position = self.initial_positions.get(self.target)
        original_target_position = self.reference_positions.get(self.target)
        original_current_separation = self._distance(
            current_target_position, original_target_position
        )
        original_location_is_distinct = (
            original_current_separation is not None
            and original_current_separation > self.approach_threshold
        )
        approached_current = (
            self.min_current_target_distance is not None
            and self.min_current_target_distance <= self.approach_threshold
        )
        approached_original = (
            original_location_is_distinct
            and
            self.min_original_target_distance is not None
            and self.min_original_target_distance <= self.approach_threshold
        )
        summary = {
            "success": bool(success),
            "n_steps": len(self.steps),
            "grasp_check_available": self.grasp_check_available,
            "approached_current_target_location": bool(approached_current),
            "approached_original_target_location": bool(approached_original),
            "original_location_is_distinct": bool(original_location_is_distinct),
            "original_current_target_separation_m": original_current_separation,
            "min_current_target_distance_m": self.min_current_target_distance,
            "min_original_target_distance_m": self.min_original_target_distance,
            "grasped_any_object": bool(self.ever_grasped),
            "grasped_correct_object": bool(target_grasped),
            "grasped_wrong_object": bool(wrong_objects),
            "grasped_objects": sorted(self.ever_grasped),
            "wrong_grasped_objects": wrong_objects,
            "dropped_before_success": bool(self.drop_events),
            "drop_events": self.drop_events,
        }
        return {"meta": self.meta, "summary": summary, "steps": self.steps}


def save_episode_diagnostics(payload, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(_jsonable(payload), handle, separators=(",", ":"))


def aggregate_episode_summaries(episodes):
    valid = [episode["summary"] for episode in episodes]
    n = len(valid)
    if not n:
        return {"n_episodes": 0}

    def rate(key, denominator=None):
        rows = valid if denominator is None else [row for row in valid if row[denominator]]
        return None if not rows else sum(bool(row[key]) for row in rows) / len(rows)

    return {
        "n_episodes": n,
        "success_rate": rate("success"),
        "approached_original_target_location_rate": rate(
            "approached_original_target_location", denominator="original_location_is_distinct"
        ),
        "approached_current_target_location_rate": rate(
            "approached_current_target_location"
        ),
        "grasped_any_object_rate": rate("grasped_any_object"),
        "grasped_correct_object_rate": rate("grasped_correct_object"),
        "grasped_wrong_object_rate": rate("grasped_wrong_object"),
        "drop_rate_among_episodes_with_grasp": rate(
            "dropped_before_success", denominator="grasped_any_object"
        ),
        "n_episodes_with_grasp": sum(row["grasped_any_object"] for row in valid),
        "n_episodes_with_distinct_original_location": sum(
            row["original_location_is_distinct"] for row in valid
        ),
        "n_episodes_with_exact_grasp_check": sum(
            row["grasp_check_available"] for row in valid
        ),
    }
