"""MuJoCo backend on the CPU: drives, Jacobian, grasp and camera on a small test robot, and (with the G1-D assets and a
sorting layout under Assets/) the eval client running the sorting expert's first pick on mujoco."""

import json
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
torch = pytest.importorskip("torch")

from env import sim_backend  # noqa: E402
from env.backends.mujoco.articulation import grid_origins  # noqa: E402
from env.backends.mujoco.sim_env import MAX_TIMESTEP, MjSimEnv  # noqa: E402
from env.backends.mujoco.world import ROOT, EnvWorld  # noqa: E402
from env.robot_manager.robot_spec import ActuatorGroup, RobotSpec  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
DT = 1 / 300  # the G1-D sim config's physics step

# Column (prismatic z) -> yaw -> pitch -> hand with two pads closing along y: q = 0 open (55 mm gap), +q closes.
URDF = """<robot name="pincher">
  <link name="base"><inertial><mass value="5"/><inertia ixx="0.1" iyy="0.1" izz="0.1" ixy="0" ixz="0" iyz="0"/>
  </inertial></link>
  <joint name="lift" type="prismatic"><parent link="base"/><child link="column"/><origin xyz="0 0 0.5"/>
    <axis xyz="0 0 1"/><limit lower="-0.3" upper="0.3" effort="1000" velocity="1"/></joint>
  <link name="column"><inertial><mass value="1"/><inertia ixx="0.01" iyy="0.01" izz="0.01" ixy="0" ixz="0" iyz="0"/>
  </inertial></link>
  <joint name="yaw" type="revolute"><parent link="column"/><child link="upper"/><axis xyz="0 0 1"/>
    <limit lower="-3" upper="3" effort="100" velocity="5"/></joint>
  <link name="upper"><inertial><origin xyz="0.15 0 0"/><mass value="1"/>
    <inertia ixx="0.005" iyy="0.01" izz="0.01" ixy="0" ixz="0" iyz="0"/></inertial></link>
  <joint name="pitch" type="revolute"><parent link="upper"/><child link="hand"/><origin xyz="0.3 0 0"/>
    <axis xyz="0 1 0"/><limit lower="-2" upper="2" effort="100" velocity="5"/></joint>
  <link name="hand"><inertial><mass value="0.3"/><inertia ixx="0.001" iyy="0.001" izz="0.001" ixy="0" ixz="0" iyz="0"/>
  </inertial></link>
  <joint name="finger_1" type="prismatic"><parent link="hand"/><child link="pad_1"/><origin xyz="0 0.03 -0.05"/>
    <axis xyz="0 -1 0"/><limit lower="0" upper="0.025" effort="55" velocity="0.064"/></joint>
  <link name="pad_1"><inertial><mass value="0.04"/><inertia ixx="1e-5" iyy="1e-5" izz="1e-5" ixy="0" ixz="0" iyz="0"/>
    </inertial><collision><geometry><box size="0.02 0.005 0.04"/></geometry></collision></link>
  <joint name="finger_2" type="prismatic"><parent link="hand"/><child link="pad_2"/><origin xyz="0 -0.03 -0.05"/>
    <axis xyz="0 1 0"/><limit lower="0" upper="0.025" effort="55" velocity="0.064"/></joint>
  <link name="pad_2"><inertial><mass value="0.04"/><inertia ixx="1e-5" iyy="1e-5" izz="1e-5" ixy="0" ixz="0" iyz="0"/>
    </inertial><collision><geometry><box size="0.02 0.005 0.04"/></geometry></collision></link>
</robot>
"""
PADS = [0.3, 0.0, 0.5 - 0.05]  # between the pads at lift, yaw and pitch 0
FINGER_KP, FINGER_EFFORT, FINGER_VMAX = 3.6e4, 55.0, 0.064  # the G1-D's Dex1 drive (g1d_spec.py)


@pytest.fixture
def robot_spec(tmp_path):
    urdf = tmp_path / "pincher.urdf"
    urdf.write_text(URDF)
    return RobotSpec(
        urdf_path=str(urdf),
        actuators={
            "body": ActuatorGroup(["lift", "yaw", "pitch"], stiffness=4400.0, damping=40.0, velocity_limit_sim=0.05),
            "fingers": ActuatorGroup(
                ["finger_.*"],
                stiffness=FINGER_KP,
                damping=100.0,
                effort_limit_sim=FINGER_EFFORT,
                velocity_limit_sim=FINGER_VMAX,
            ),
        },
        init_joint_pos={},
        disable_gravity=True,
        self_collisions=False,
        finger_pads={"finger_1": "pad_1", "finger_2": "pad_2"},
    )


def make_env(robot_spec):
    return MjSimEnv(1, 2.0, DT, 10, 10, "cpu", [("robot0", robot_spec, [0, 0, 0, 1, 0, 0, 0])])


def free_box(size, mass, friction=0.6):
    spec = mujoco.MjSpec()
    root = spec.worldbody.add_body(name=ROOT)
    root.add_freejoint()
    root.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX, size=[s / 2 for s in size], mass=mass, friction=[friction, 0.005, 0.0001]
    )
    return spec


def joint_ids(art, *names):
    return art.find_joints(list(names), preserve_order=True)[0]


def test_backend_precedence(monkeypatch):
    monkeypatch.delenv(sim_backend.ENV_VAR, raising=False)
    assert sim_backend.select(None, {}) == sim_backend.ISAACSIM
    monkeypatch.delenv(sim_backend.ENV_VAR)
    assert sim_backend.select(None, {"backend": "mujoco"}) == "mujoco"
    monkeypatch.setenv(sim_backend.ENV_VAR, "isaaclab_newton")
    assert sim_backend.select(None, {"backend": "mujoco"}) == "isaaclab_newton"
    assert sim_backend.select("mujoco", {}) == "mujoco" and os.environ[sim_backend.ENV_VAR] == "mujoco"
    assert not sim_backend.uses_kit("mujoco") and sim_backend.uses_kit("isaaclab_newton")
    with pytest.raises(ValueError):
        sim_backend.select("physx")


def test_substeps_cap_the_mujoco_step(robot_spec):
    env = make_env(robot_spec)
    assert env.physics_dt == DT and env.substeps * env.worlds[0].model.opt.timestep == pytest.approx(DT)
    assert env.worlds[0].model.opt.timestep <= MAX_TIMESTEP


def test_stiff_drive_is_implicit(robot_spec):
    """PhysX takes the drive spring at the next step's position. sqrt(kp / m) * dt is ~3 for the 36 kN/m pads at
    1/300 s (no substeps): undamped, an explicit spring diverges; the implicit one settles."""
    fingers_group = robot_spec.actuators["fingers"].replace(damping=1.0, effort_limit_sim=None)
    spec = robot_spec.replace(actuators={**robot_spec.actuators, "fingers": fingers_group})
    world = EnvWorld(0, DT)
    world.add_robot("robot0/", spec, [0, 0, 0], [1, 0, 0, 0])
    world.compile()
    drive = world.drives["robot0/"]
    fingers = [drive.joint_names.index(n) for n in ("finger_1", "finger_2")]
    drive.q_target[fingers] = 0.01
    for _ in range(150):
        world.step()
    q, qd = world.data.qpos[drive.qadr[fingers]], world.data.qvel[drive.dadr[fingers]]
    assert np.allclose(q, 0.01, atol=1e-4) and np.abs(qd).max() < 1e-3


def test_velocity_limit_ramps_the_target(robot_spec):
    env = make_env(robot_spec)
    art = env.scene["robot0"]
    lift = joint_ids(art, "lift")
    art.set_joint_position_target(torch.tensor([[0.2]]), joint_ids=lift)
    for _ in range(30):  # 1 s at 0.05 m/s
        env.sim_step()
    assert float(art.data.joint_pos[0, lift[0]]) == pytest.approx(0.05, abs=0.005)


def test_jacobian_matches_finite_differences(robot_spec):
    env = make_env(robot_spec)
    art = env.scene["robot0"]
    q0 = torch.tensor([[0.1, 0.4, -0.7, 0.0, 0.0]])
    names = ["lift", "yaw", "pitch", "finger_1", "finger_2"]
    ids = joint_ids(art, *names)
    art.write_joint_state_to_sim(q0, torch.zeros_like(q0), joint_ids=ids)
    hand = art.body_names.index("hand")
    jac = art.root_physx_view.get_jacobians()[0, hand - 1].numpy().astype(float)
    assert jac.shape == (6, art.num_joints)

    def pose(q):
        art.write_joint_state_to_sim(q, torch.zeros_like(q), joint_ids=ids)
        world = env.worlds[0]
        body = world.model.body("robot0/hand").id
        return world.data.xpos[body].copy(), world.data.xquat[body].copy()

    p0, r0 = pose(q0)
    eps = 1e-6
    for k, joint in enumerate(ids[:3]):
        q = q0.clone().double()
        q[0, k] += eps
        p, r = pose(q)
        local, dr = np.zeros(3), np.zeros(3)
        mujoco.mju_subQuat(local, r, r0)  # in the body frame
        mujoco.mju_rotVecQuat(dr, local, r0)
        assert np.allclose((p - p0) / eps, jac[:3, joint], atol=1e-4), names[k]
        assert np.allclose(dr / eps, jac[3:, joint], atol=1e-4), names[k]


def test_pinch_and_lift_a_light_block(robot_spec):
    """An 8 g block pinched with the full 55 N follows the hand up (soft contacts let a light block be squeezed
    through or out unless they are stiffened)."""
    env = make_env(robot_spec)
    world = env.worlds[0]
    world.add("block", free_box([0.03, 0.03, 0.015], 0.008), PADS, [1, 0, 0, 0])
    env.compile()
    art = env.scene["robot0"]
    fingers = joint_ids(art, "finger_1", "finger_2")
    lift = joint_ids(art, "lift")
    gravity = world.model.opt.gravity.copy()
    world.model.opt.gravity[:] = 0  # the block floats between the pads until they close
    art.set_joint_position_target(torch.tensor([[0.025, 0.025]]), joint_ids=fingers)
    for _ in range(30):
        env.sim_step()
    world.model.opt.gravity[:] = gravity
    for _ in range(10):
        env.sim_step()
    q = art.data.joint_pos[0, fingers].numpy()
    assert np.allclose(q, 0.03 - 0.0025 - 0.015, atol=0.001)  # pads stop on the block's faces
    forces = env.scene["robot0_pads"].data.net_forces_w[0].numpy()
    assert np.allclose(np.abs(forces[:, 1]), FINGER_EFFORT, rtol=0.1) and forces[0, 1] * forces[1, 1] < 0
    art.set_joint_position_target(torch.tensor([[0.05]]), joint_ids=lift)
    for _ in range(45):
        env.sim_step()
    z = world.item_pose("block")[0][2]
    assert float(art.data.joint_pos[0, lift[0]]) == pytest.approx(0.05, abs=0.002)
    assert z == pytest.approx(PADS[2] + 0.05, abs=0.003)


def test_grid_origins_are_centred():
    origins = grid_origins(4, 2.0)
    assert np.allclose(origins.mean(axis=0), 0) and len({tuple(o) for o in origins}) == 4


def test_camera_projection_matches_intrinsics():
    """A point rendered by a camera built from a template lands where the template's K projects it."""
    from env.backends.mujoco.cameras import _fovy
    from env.camera_manager.camera_pose import REAL_MAP, intrinsics

    camera_type = next(t for t, info in REAL_MAP.items() if "focal_length" in info and "horizontal_aperture" in info)
    width, height = REAL_MAP[camera_type]["resolution"]
    world = EnvWorld(0, DT, offscreen_size=(width, height))
    marker = mujoco.MjSpec()
    marker.worldbody.add_body(name=ROOT).add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.01], rgba=[1, 0, 0, 1])
    world.add("marker", marker, [0.1, 0.05, 0.0])
    world.cameras["top"] = {"body": None, "pos": [0, 0, 1.0], "quat": [1, 0, 0, 0], "fovy": _fovy(camera_type)}
    world.compile()
    try:
        renderer = mujoco.Renderer(world.model, height, width)
    except Exception as err:  # no OpenGL context (e.g. a Linux runner without EGL/OSMesa)
        pytest.skip(f"no offscreen OpenGL: {err}")
    renderer.update_scene(world.data, camera="top")
    rgb = renderer.render().astype(int)
    renderer.close()
    red = (rgb[..., 0] > 150) & (rgb[..., 1] < 80) & (rgb[..., 2] < 80)
    v, u = np.nonzero(red)
    k = intrinsics(camera_type)
    # the camera looks down -z with +y up: image x = +x, image y = -y, depth 1 m
    expected = [k[0, 2] + k[0, 0] * 0.1, k[1, 2] - k[1, 1] * 0.05]
    assert red.sum() > 10
    assert np.allclose([u.mean(), v.mean()], expected, atol=1.5)
    assert math.isclose(_fovy(camera_type), math.degrees(2 * math.atan(height / 2 / k[1, 1])))


# ---- end to end: the eval client on mujoco ----------------------------------------------------------------------------

SORT_TASK = "g1d_sort_blocks_by_color"
SORT_LAYOUT = REPO / "Assets/Eval_Layout/RoboDojo/g1d/0" / f"{SORT_TASK}_0.json"
G1D_URDF = REPO / "Assets/Robots/g1d"
FIRST_PICK = 136  # actions until the sorting expert has lifted its first block


def eval_one_episode(TASK_ENV, model_client):
    """Policy module of the end-to-end test: the sorting expert's first pick, then how far the blocks rose."""
    from scripts.g1d import scripted_demo as demo

    TASK_ENV.get_obs()
    lm = TASK_ENV.scene_manager.layout_manager
    blocks = [b for box in TASK_ENV.BOXES for b in TASK_ENV.groups[0][box]]

    def heights():
        return np.array([float(lm.get_instance_pose(0, label=b)[0][2]) for b in blocks])

    start = heights()
    for i, action in enumerate(demo.block_sort_actions(TASK_ENV)):
        TASK_ENV.take_action(action)
        if i + 1 == FIRST_PICK:
            break
    obs = TASK_ENV.get_obs()
    frames = {f"{cam}/{ann}": list(np.shape(v)) for cam, data in obs["vision"].items() for ann, v in data.items()}
    result = {"rise": float((heights() - start).max()), "frames": frames}

    # a diverged solver (MuJoCo itself resets a bad qpos, so the pose is faked) ends the episode as unstable
    rm = TASK_ENV.robot_manager
    real_endpose = rm.get_real_endpose
    rm.get_real_endpose = lambda robot, env_idx_list=None, is_relative=True: {0: np.full(7, np.nan)}
    result["diverged_ends"] = bool(TASK_ENV.take_action(action) is None and TASK_ENV.is_episode_end())
    result["diverged_unstable"] = 0 in TASK_ENV.unstable_envs
    rm.get_real_endpose = real_endpose
    TASK_ENV.unstable_envs.discard(0)

    Path(os.environ["ROBODOJO_TEST_RESULT"]).write_text(json.dumps(result))
    TASK_ENV.success[0], TASK_ENV.end_flag[0] = False, True


@pytest.mark.skipif(not (SORT_LAYOUT.exists() and G1D_URDF.exists()), reason="needs the G1-D and a sorting layout")
def test_eval_client_runs_the_sorting_expert(tmp_path):
    xpolicylab = REPO / "XPolicyLab"
    if not (xpolicylab / "client_server").exists():
        pytest.skip("needs the XPolicyLab submodule")
    env = dict(
        os.environ,
        ROBODOJO_SIM_BACKEND="mujoco",
        ROBODOJO_TEST_RESULT=str(tmp_path / "result.json"),
        EVAL_NUM="1",
        PYTHONPATH=os.pathsep.join([str(REPO), str(xpolicylab), os.environ.get("PYTHONPATH", "")]),
    )
    code = (
        "import sys; from scripts.g1d import scripted_demo; "
        f"sys.argv = ['x', '--env_cfg_type', 'g1d', '--task_name', '{SORT_TASK}', '--seed', '0', "
        "'--layout_ids', '0', '--allow_train_layouts']; "
        "scripted_demo.main(policy_module='env.backends.mujoco.test_mujoco_backend')"
    )
    run = subprocess.run([sys.executable, "-c", code], cwd=REPO, env=env, capture_output=True, text=True, timeout=600)
    assert run.returncode == 0, run.stdout[-3000:] + run.stderr[-3000:]
    result = json.loads((tmp_path / "result.json").read_text())
    assert result["rise"] > 0.03, result
    assert result["frames"], result
    assert result["diverged_ends"] and result["diverged_unstable"], result
