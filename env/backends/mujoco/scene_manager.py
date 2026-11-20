"""SceneManager of the MuJoCo backend: the same LayoutManager, layout files and reload / saved-pose flow as
env/scene_manager/scene_manager.py, building each env's MuJoCo model (world.py) instead of a USD stage."""

from collections.abc import Sequence
import logging
from typing import Any, Dict, List

import mujoco
from omegaconf import DictConfig, OmegaConf
import torch

from env.backends.mujoco import usd_assets
from env.backends.mujoco.objects import (
    MjArticulationObject,
    MjDynamicObject,
    MjGeometryObject,
    MjRigidObject,
    MjTable,
    light_kwargs,
)
from env.scene_manager.layout_manager import LayoutManager
from env.seeding import seed_everywhere
from utils.path import deep_resolve_paths

logger = logging.getLogger(__name__)

LAYOUT_METADATA = (
    *("Light", "Room", "Ground", "Background", "Table"),
    *("Robot", "Instruction", "Camera", "Imaging", "Generalization"),
)
GLOBAL_CONFIG_KEYS = (
    "common",
    "deformable_material",
    "visual_material",
    "deformable_config",
    "particle_system",
    "particle_material",
    "garment_config",
    "fem_config",
)
UNSUPPORTED_TYPES = ("garment", "fluid", "granular")  # PhysX particles / FEM cloth
DEFAULT_LIGHTS = (
    {
        "type": mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
        "pos": [0, -0.5, 3],
        "dir": [0.2, 0.3, -1],
        "diffuse": [0.45] * 3,
    },
    {"type": mujoco.mjtLightType.mjLIGHT_DIRECTIONAL, "pos": [0, 1, 3], "dir": [-0.2, -0.6, -1], "diffuse": [0.25] * 3},
)


def _plain(cfg):
    if isinstance(cfg, DictConfig):
        return OmegaConf.to_container(cfg, resolve=True)
    return cfg if cfg is not None else {}


class MujocoSceneManager:
    def __init__(
        self,
        num_envs: int,
        device,
        env_spacing: float,
        scene_config: DictConfig,
        task_config: DictConfig,
        use_fabric: bool = False,
        seeds_per_env: Sequence[int] | None = None,
    ):
        self.config = DictConfig({})
        self.background_config = DictConfig({})
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.env_spacing = env_spacing
        self.use_fabric = use_fabric
        self.sim = None
        self.layout_manager = LayoutManager(
            num_envs=num_envs,
            env_spacing=env_spacing,
            seeds_per_env=seeds_per_env,
            scene_config=scene_config,
            task_config=task_config,
        )
        self.layout_manager.scene_manager = self
        self._seeds_per_env = None
        if seeds_per_env is not None:
            self.update_env_seeds(seeds_per_env)
        self.env_roots = [f"/World/envs/env_{env_id}" for env_id in range(num_envs)]
        self._rigid_and_dynamic_objects: List[Dict[str, Any]] = [{} for _ in range(num_envs)]
        self._articulation_objects: List[Dict[str, Any]] = [{} for _ in range(num_envs)]
        self._garment_objects: List[Dict[str, Any]] = [{} for _ in range(num_envs)]
        self._geometry_objects: List[Dict[str, Any]] = [{} for _ in range(num_envs)]
        self._fluid_objects: List[Dict[str, Any]] = [{} for _ in range(num_envs)]
        self._tables: List[MjTable | None] = [None] * num_envs
        self._taken_paths: List[set] = [set() for _ in range(num_envs)]
        self.pending_initialization = []
        self.setup_scene = False

    def is_path_taken(self, prim_path: str) -> bool:
        return any(prim_path in taken for taken in self._taken_paths)

    def update_env_seeds(self, seeds: Sequence[int] | None):
        if seeds is None:
            self._seeds_per_env = None
            return
        seed_list = [int(s) for s in seeds]
        if len(seed_list) != self.num_envs:
            raise ValueError(f"seed list length {len(seed_list)} does not match num_envs {self.num_envs}.")
        self._seeds_per_env = seed_list
        self.layout_manager.update_env_seeds(seeds)

    def _set_env_seed(self, env_id: int):
        if self._seeds_per_env is None:
            return
        if env_id >= len(self._seeds_per_env):
            raise IndexError(f"Requested env_id {env_id} exceeds configured seeds (len={len(self._seeds_per_env)}).")
        seed_everywhere(self._seeds_per_env[env_id])

    def _world(self, env_id):
        return self.sim.worlds[env_id]

    def initialize(self, sim):
        self.sim = sim
        self.env_origins = self.sim.scene.env_origins
        self.background_config = self.layout_manager.select_background()
        for env_idx in range(self.num_envs):
            self.config[f"env{env_idx}"] = self.layout_manager.load_saved_layout(env_idx)
        for env_idx in range(self.num_envs):
            self._set_env_seed(env_idx)
            self.spawn_scene_objects(env_id=env_idx, exclude_types=["articulation"])
        self.setup_scene = True

    def post_init(self):
        for env_id in range(self.num_envs):
            for obj_dict in (self._rigid_and_dynamic_objects[env_id], self._articulation_objects[env_id]):
                for obj in obj_dict.values():
                    obj.initialize()

    def reload_scene(self):
        logger.info("Resetting all environments")
        self.post_init()
        self.reload_envs(list(range(self.num_envs)))
        self.setup_scene = False

    def reload_envs(self, env_ids: Sequence[int]):
        if self.background_config is not None and not self.setup_scene:
            self.background_config = self.layout_manager.select_background()
        env_ids = [int(e) for e in env_ids]
        for env_id in env_ids:
            self._set_env_seed(env_id)
            self.reload_env_scene(env_id)
        self.sim.compile(env_ids)
        for obj in self.pending_initialization:
            obj.initialize()
        self.pending_initialization.clear()
        self.sim.sim_step()
        for env_idx in range(self.num_envs):
            if self._tables[env_idx] is not None:
                self._tables[env_idx].relocate_offscreen()

    def apply_saved_poses(self, env_idx_list: List[int]):
        """Tables first, settle, then every object at its saved pose, settle (as the Isaac scene manager)."""
        for env_idx in env_idx_list:
            if self._tables[env_idx] is not None:
                self._tables[env_idx].apply_saved_pose()
        for _ in range(20):
            self.sim.sim_step(render=False)
        for env_idx in env_idx_list:
            for obj_dict in self._collections(env_idx):
                for obj in obj_dict.values():
                    obj.apply_saved_pose()
        for _ in range(20):
            self.sim.sim_step(render=False)

    def _collections(self, env_id):
        return (
            self._rigid_and_dynamic_objects[env_id],
            self._articulation_objects[env_id],
            self._geometry_objects[env_id],
        )

    def reload_env_scene(self, env_id: int, object_types: List[str] = None):
        if not self.setup_scene:
            self.config[f"env{env_id}"] = self.layout_manager.load_saved_layout(env_id)
        world = self._world(env_id)
        env_cfg = self.config.get(f"env{env_id}") or {}
        self._reload_lights(world, env_cfg.get("Light"))
        self._reload_fixture(world, "Ground", env_cfg.get("Ground"), self._ground_template)
        self._reload_fixture(world, "Room", env_cfg.get("Room"), self._room_template)
        self._reload_table(env_id, world, env_cfg.get("Table"))
        self.clear_scene_objects(env_id)
        self.spawn_scene_objects(env_id)

    def _reload_lights(self, world, light_cfg):
        world.lights = []
        ambient = None
        if light_cfg:
            for light in _plain(light_cfg).values():
                kwargs = light_kwargs(light["types"], light)
                if "ambient" in kwargs:
                    ambient = kwargs["ambient"]
                else:
                    world.lights.append(kwargs)
        if not world.lights:
            world.lights = [dict(light) for light in DEFAULT_LIGHTS]
        world.ambient = ambient

    def _reload_fixture(self, world, key, cfg, build):
        world.remove(key)
        if cfg is None:
            return
        cfg = _plain(cfg)
        template = build(cfg)
        if template is not None:
            world.add(key, template, cfg.get("default_pos", [0, 0, 0]), cfg.get("default_ori", [1, 0, 0, 0]))

    def _ground_template(self, cfg):
        thickness = cfg.get("thickness", 0.1)
        materials = cfg.get("materials") or {}
        image = None
        if isinstance(materials, dict) and materials.get("default"):
            image = usd_assets.material_base_color(f"{_assets_path()}/Material/{materials['default']}")
        friction = (cfg.get("physics_material") or {}).get("static_friction", 1.0)
        size = [self.env_spacing, self.env_spacing, thickness]
        template = usd_assets.box_template(size, rgba=(0.55, 0.55, 0.55, 1), texture=image, friction=friction)
        template.geoms[0].pos = [0, 0, -thickness / 2]  # the top face is at default_pos (Ground.create)
        return template

    def _room_template(self, cfg):
        if cfg.get("appearance") or cfg.get("table_lift"):
            logger.warning("MuJoCo backend: room appearance / table_lift overrides are not applied")
        return usd_assets.static_template(cfg["usd_path"], scale=cfg.get("scale", (1, 1, 1)), collide=True)

    def _reload_table(self, env_id, world, cfg):
        world.remove("Table")
        self._tables[env_id] = None
        if cfg is None:
            return
        cfg = _plain(cfg)
        image = usd_assets.material_base_color(cfg.get("mdl_path"))
        template = usd_assets.box_template(cfg.get("scale", [1, 1, 1]), rgba=(0.7, 0.7, 0.7, 1), texture=image)
        self._tables[env_id] = MjTable(world, "Table", template, cfg)

    def spawn_scene_objects(self, env_id: int, exclude_types: List[str] | None = None):
        if not self.config:
            return
        objects_cfg = self.config.get(f"env{env_id}", None)
        if objects_cfg is None:
            return
        exclude_types = exclude_types or []
        for physics_type in objects_cfg:
            if physics_type in LAYOUT_METADATA:
                continue
            for cat_name in objects_cfg[physics_type]:
                if cat_name in GLOBAL_CONFIG_KEYS:
                    continue
                self.spawn_category_objects(env_id, cat_name, objects_cfg[physics_type][cat_name], exclude_types)

    def spawn_category_objects(self, env_id, cat_name, cat_cfg, exclude_types=None):
        exclude_types = exclude_types or []
        for inst_cfg in cat_cfg:
            deep_resolve_paths(inst_cfg)
            obj_type = inst_cfg.get("physics", {}).get("type", "rigid")
            if obj_type in exclude_types:
                continue
            obj = self.create_scene_object(env_id, inst_cfg, obj_type)
            if obj is not None:
                self.register_scene_object(inst_cfg.get("inst_name"), env_id, obj, obj_type)
                self.pending_initialization.append(obj)

    def create_scene_object(self, env_id, inst_cfg, obj_type):
        if obj_type in UNSUPPORTED_TYPES:
            raise NotImplementedError(
                f"MuJoCo backend: '{obj_type}' objects (PhysX particles / cloth) are not supported; "
                f"run this task with --sim_backend isaacsim"
            )
        usd_path = inst_cfg.get("usd_path")
        if not usd_path:
            raise NotImplementedError(f"MuJoCo backend: primitive '{inst_cfg.get('inst_name')}' has no USD asset")
        cfg = _plain(inst_cfg)
        physics, visual = cfg.get("physics", {}) or {}, cfg.get("visual", {}) or {}
        scale = cfg.get("scale", (1, 1, 1))
        if obj_type in ("rigid", "dynamic"):
            from env.scene_manager.objects.asset_defaults import with_asset_defaults

            cfg = with_asset_defaults(usd_path, cfg)
            physics = cfg.get("physics", {}) or {}
        if obj_type == "rigid":
            mass = min(physics.get("mass", 0.5), physics.get("mass_cap", 0.5))
            template = usd_assets.rigid_template(
                usd_path,
                scale,
                mass=mass if mass > 0 else 0.05,
                friction=(physics.get("static_friction", 0.6), physics.get("dynamic_friction", 1.5)),
                color=visual.get("color"),
                color_parts=visual.get("color_parts"),
            )
            cls = MjRigidObject
        elif obj_type == "dynamic":
            template, cls = usd_assets.articulation_template(usd_path, scale), MjDynamicObject
        elif obj_type == "geometry":
            template = usd_assets.static_template(
                usd_path,
                scale,
                collide=True,
                color=visual.get("color"),
                friction=physics.get("static_friction"),
            )
            cls = MjGeometryObject
        elif obj_type == "articulation":
            template = usd_assets.articulation_template(
                usd_path, scale, fixed_base=physics.get("fixed_base", False), color=visual.get("color")
            )
            cls = MjArticulationObject
        else:
            raise ValueError(f"Invalid object type: {obj_type}")
        prim_path = cfg["prim_path"]
        self._taken_paths[env_id].add(prim_path)
        return cls(
            self._world(env_id),
            prim_path,
            usd_path,
            DictConfig(cfg),
            self.env_origins[env_id],
            cfg.get("default_pos", (0.0, 0.0, 0.0)),
            cfg.get("default_ori", (1.0, 0.0, 0.0, 0.0)),
            scale,
            template,
        )

    def register_scene_object(self, obj_key, env_id, obj, obj_type):
        if obj_type in ("rigid", "dynamic"):
            self._rigid_and_dynamic_objects[env_id][obj_key] = obj
        elif obj_type == "articulation":
            self._articulation_objects[env_id][obj_key] = obj
        elif obj_type == "geometry":
            self._geometry_objects[env_id][obj_key] = obj

    def clear_scene_objects(self, env_id: int, exclude_types: List[str] | None = None):
        for obj_dict in self._collections(env_id):
            for obj in obj_dict.values():
                obj.destroy()
            obj_dict.clear()

    def relocate_stale_objects(self, env_id: int, obj_types: List[str] = None):
        for obj_dict in self._collections(env_id):
            for obj in obj_dict.values():
                obj.relocate_offscreen()

    def is_global_config_key(self, key: str) -> bool:
        return key in GLOBAL_CONFIG_KEYS

    def get_objects(self, env_ids=None, object_name=None, object_type=None) -> Dict[str, Any]:
        if env_ids is None:
            env_ids = list(range(self.num_envs))
        result = {}
        for collection_type, env_dict_list in self._get_object_collections_by_type(object_type).items():
            for env_id in env_ids:
                if env_id >= len(env_dict_list):
                    continue
                for obj_key, obj in env_dict_list[env_id].items():
                    if object_name is None or obj_key == object_name:
                        result[f"env{env_id}_{collection_type}_{obj_key}"] = obj
        return result

    def _get_object_collections_by_type(self, object_type):
        all_collections = {
            "rigid": self._rigid_and_dynamic_objects,
            "dynamic": self._rigid_and_dynamic_objects,
            "articulation": self._articulation_objects,
            "garment": self._garment_objects,
            "geometry": self._geometry_objects,
            "fluid": self._fluid_objects,
        }
        if object_type is None:
            return all_collections
        if object_type in all_collections:
            return {object_type: all_collections[object_type]}
        return {}

    def close(self):
        self.layout_manager.close()
        for env_id in range(self.num_envs):
            for obj_dict in self._collections(env_id):
                obj_dict.clear()
            self._tables[env_id] = None


def _assets_path():
    from utils.path import ASSETS_PATH

    return ASSETS_PATH
