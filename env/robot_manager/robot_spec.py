"""Robot physics shared by the simulation backends, without simulator imports.

A robot config module (env/robot_manager/robot_config/<name>.py) builds the IsaacLab ArticulationCfg from it; the MuJoCo
backend builds its model from the same spec (env/backends/mujoco/robots.py), so both drive the joints alike.
"""

from dataclasses import dataclass, field, fields, replace


@dataclass(frozen=True)
class ActuatorGroup:
    """Joint-space PD drive of the joints matching `joint_names_expr` (IsaacLab ImplicitActuatorCfg fields)."""

    joint_names_expr: list[str]
    stiffness: float
    damping: float
    effort_limit_sim: float | None = None
    velocity_limit_sim: float | None = None
    armature: float | None = None

    def replace(self, **changes):
        return replace(self, **changes)

    def as_kwargs(self):
        """Fields for ImplicitActuatorCfg; unset optional fields are left to its defaults."""
        values = {f.name: getattr(self, f.name) for f in fields(self)}
        return {k: v for k, v in values.items() if v is not None}


@dataclass(frozen=True)
class RobotSpec:
    urdf_path: str
    actuators: dict[str, ActuatorGroup]
    init_joint_pos: dict[str, float]  # {joint name pattern: value}
    fix_base: bool = True
    disable_gravity: bool = False
    self_collisions: bool = True
    filtered_pairs: list[tuple[str, str]] = field(default_factory=list)  # links that never collide
    finger_pads: dict[str, str] | None = None  # {finger joint: pad link} for the grip force
    hidden_visuals: list[str] = field(default_factory=list)  # regexes on URDF <visual name>: not rendered
    # {link name regex: PreviewSurface fields (diffuse_color, metallic, roughness)} replacing the URDF colours
    link_finishes: dict[str, dict] = field(default_factory=dict)

    def replace(self, **changes):
        return replace(self, **changes)
