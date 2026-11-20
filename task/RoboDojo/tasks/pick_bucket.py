"""Five ordered stages of the registered real G1-D food/basket task."""

import json
from pathlib import Path

import numpy as np
from pxr import Gf, Usd, UsdGeom
from scipy.spatial.transform import Rotation

from env.environment.task_env import TaskEnv
from env.reward_manager.pick_bucket_checks import Progress
from env.reward_manager.reward_manager import RewardManager


class PickBucketCommon:
    def __init__(self, config, app, **kwargs):
        if config.eval_cfg.config_name != "g1d_pick_bucket":
            raise ValueError("pick_bucket recorded initialization requires --env_cfg_type g1d_pick_bucket")
        super().__init__(config, app, **kwargs)
        from scripts.g1d.pick_bucket_alignment import configure_camera

        configure_camera(self.camera_manager.init_config)
        self.reward_manager = RewardManager(self.num_envs)
        self.step_lim = 1800
        self.progress = [Progress() for _ in range(self.num_envs)]
        self.trace = [[] for _ in range(self.num_envs)]
        self.food_surfaces = {}

    def _post_setup_scene(self, sim):
        super()._post_setup_scene(sim)
        from scripts.g1d.pick_bucket_alignment import apply_lens

        apply_lens(self.camera_manager)
        self.reward_manager.initialize(self)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        from scripts.g1d.pick_bucket_alignment import apply_lens

        apply_lens(self.camera_manager)
        from scripts.g1d.pick_bucket_alignment import add_hatch

        add_hatch(self.scene_manager)
        self.reward_manager.reset()
        self.progress = [Progress() for _ in range(self.num_envs)]
        self.trace = [[] for _ in range(self.num_envs)]
        self.food_surfaces = {}

    def food_surface(self, i):
        """Actual visible mesh vertices in the rigid root frame; a mango is not its enclosing box."""
        if i not in self.food_surfaces:
            lm = self.scene_manager.layout_manager
            obj = lm.get_scene_object(i, lm.get_instance_name(i, "food"))
            root = obj.stage.GetPrimAtPath(obj.usd_prim_path)
            cache, vertices = UsdGeom.XformCache(), []
            for prim in Usd.PrimRange(root):
                if not prim.IsA(UsdGeom.Mesh) or UsdGeom.Imageable(prim).ComputeVisibility() == "invisible":
                    continue
                transform, _ = cache.ComputeRelativeTransform(prim, root)
                vertices.extend(
                    transform.Transform(Gf.Vec3d(*map(float, v))) for v in UsdGeom.Mesh(prim).GetPointsAttr().Get()
                )
            if not vertices:
                raise ValueError("Food asset has no visible mesh for containment checks")
            self.food_surfaces[i] = np.asarray(vertices) * np.asarray(obj.scale)
        return self.food_surfaces[i]

    def pick_bucket_complete(self, i):
        lm = self.scene_manager.layout_manager
        positions, rotations, boxes = {}, {}, {}
        for label in ("food", "basket"):
            pos, quat = lm.get_instance_pose(i, label=label)
            positions[label] = np.asarray(pos, float)
            rotations[label] = Rotation.from_quat(np.asarray(quat)[[1, 2, 3, 0]])
            boxes[label] = lm.get_instance_bbox_vertices(lm.get_instance_name(i, label), i)
        food_bottom = (positions["food"] + rotations["food"].apply(self.food_surface(i)))[:, 2].min()
        basket_bottom = (positions["basket"] + rotations["basket"].apply(boxes["basket"]))[:, 2].min()
        # Registered 00014: origin at full bbox centre; base=-126 mm, rim=-13 mm.
        food_corners = (
            rotations["basket"]
            .inv()
            .apply(positions["food"] + rotations["food"].apply(self.food_surface(i)) - positions["basket"])
        )
        radius = 0.0725 + (food_corners[:, 2] + 0.126) / 0.113 * (0.12 - 0.0725)
        inside = bool(
            np.all(np.linalg.norm(food_corners[:, :2], axis=1) < radius - 0.004)
            and np.all(food_corners[:, 2] > -0.130)
            and np.all(food_corners[:, 2] < -0.013)
        )
        rm = self.robot_manager
        right = next(r for r in rm.robot_list if r.type == "target" and r.arm_name.startswith("right"))
        val = rm.get_end_effector_real_val(robot=right, env_idx_list=[i])[i]
        lo, hi = right.gripper_scale
        opening = float((hi - np.mean(val)) / (hi - lo))
        entry = self.scene_manager.config[f"env{i}"]["Rigid"]["mango_prop"][0]
        table = entry["default_pos"][2] + boxes["food"][:, 2].min() - 0.001
        state = self.progress[i]
        hand_pose = rm.get_real_endpose(right, env_idx_list=[i], is_relative=True)[i]
        hand_rot = Rotation.from_quat(hand_pose[[4, 5, 6, 3]])
        pads = hand_pose[:3] + hand_rot.apply([0.12, 0, 0])
        handle = positions["basket"] + rotations["basket"].apply([0, 0, 0.122])
        food_near = bool(np.linalg.norm(pads - positions["food"]) < 0.070)
        handle_near = bool(np.linalg.norm(pads - handle) < 0.045)
        upright = bool(rotations["basket"].apply([0, 0, 1])[2] > np.cos(np.deg2rad(10)))
        done = state.update(
            food_bottom - table,
            basket_bottom - table,
            inside,
            opening,
            positions["basket"],
            food_near=food_near,
            handle_near=handle_near,
            upright=upright,
        )
        self.trace[i].append(
            dict(
                stage=state.stage,
                food_height=float(food_bottom - table),
                basket_height=float(basket_bottom - table),
                inside=inside,
                opening=opening,
                food_near=food_near,
                handle_near=handle_near,
                upright=upright,
                pads=pads.tolist(),
                food=positions["food"].tolist(),
                food_quat=rotations["food"].as_quat().tolist(),
                basket_quat=rotations["basket"].as_quat().tolist(),
                basket=positions["basket"].tolist(),
            )
        )
        return done

    def run_reward(self):
        self.reward_manager.check([("is_pick_bucket_complete", {})])

    def gen_instruction(self, env_idx):
        path = Path(__file__).resolve().parents[3] / "real2sim/tasks/pick_bucket/instructions.json"
        data = json.loads(path.read_text())
        return data["en"] + data["zh"]


class pick_bucket(PickBucketCommon, TaskEnv):
    pass
