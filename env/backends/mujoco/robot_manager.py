from env.backends.mujoco.robots import get_robot_spec
from env.robot_manager.robot_manager import RobotManager, merge_init_joint_pos, override_actuators


class MujocoRobotManager(RobotManager):
    """RobotManager on the MuJoCo backend: the same robot classes, control and state code over MjArticulation; each
    robot's model comes from its RobotSpec (env/backends/mujoco/robots.py) with the robot yml's overrides."""

    def _attach_scene_cfg(self, robot_name, robots, idx, cfg):
        if not isinstance(robots, tuple):
            robots = (robots,)
        spec = get_robot_spec(robot_name)
        if cfg is not None and "enabled_self_collisions" in cfg:
            spec = spec.replace(self_collisions=cfg["enabled_self_collisions"])
        if cfg is not None and "actuators" in cfg:
            actuators = override_actuators(spec.actuators, cfg["actuators"], lambda group, **kw: group.replace(**kw))
            spec = spec.replace(actuators=actuators)
        spec = spec.replace(init_joint_pos=merge_init_joint_pos(spec.init_joint_pos, cfg))
        for robot in robots:
            robot.SceneCfg = None
            robot.robot_spec = spec
        return robots[0] if len(robots) == 1 else robots

    def _get_SceneCfg(self, num_envs, env_spacing, replicate_physics=False):
        """[(scene key, RobotSpec, env-local root pose)] of the robots MjSimEnv builds (the first arm of a coupled
        robot carries the articulation)."""
        entries = []
        for idx, robot in enumerate(self.robot_list):
            robot.finger_pads = robot.robot_spec.finger_pads
            robot.tactile_contact_links = None
            if self.use_scene_cfg[idx]:
                entries.append((f"robot{idx}", robot.robot_spec, list(robot.entity_origin_pose)))
        return entries
