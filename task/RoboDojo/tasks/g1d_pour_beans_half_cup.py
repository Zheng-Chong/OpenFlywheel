import json
from pathlib import Path

from env.environment.task_env import TaskEnv
from task.RoboDojo.tasks.g1d_pour_beans_cup_to_cup import G1dPourBeansCupToCupCommon

INSTRUCTIONS = Path(__file__).resolve().parents[3] / "real2sim/tasks/g1d_pour_beans_half_cup/instructions.json"
# how the real data and the instructions name each target cup
TARGET_NAMES = {
    ("plastic_cup", 9): {"zh": "灰色的粗口杯", "en": "the grey cup"},
    ("plastic_cup", 8): {"zh": "蓝色杯子", "en": "the blue cup"},
}
# the locked platform definition: the target cup 30-75% full (of its capacity, round_cup_shape), at most 5% of what
# left the tall cup spilled; the datagen plugin's half_cup_v1 counts the same beans
FILL, SPILL = (0.30, 0.75), 0.05


class G1dPourBeansHalfCupCommon(G1dPourBeansCupToCupCommon):
    """倒半杯 (Pour Half a Cup of Beans): the right hand picks the tall square-mouth cup of red beans, the left hand
    the grey or blue cup; held up over the desk, the tall cup pours into the other one until it is about half full;
    both cups go back upright on the desk and the hands let go. The twin of the real G1D-PourBeansEps380 episodes
    (real2sim/tasks/g1d_pour_beans_half_cup/: the locked platform definition and the real episodes it was made from).
    """

    def __init__(self, config, app, **kwargs):
        super().__init__(config, app, **kwargs)
        self.step_lim = 1200  # the real episodes take 12-32 s to pick, pour and put back both cups

    def pour_check(self):
        return self.reward_manager.is_fluid_fill_in_range(
            "beans", "target_cup", fill=FILL, label_C="source_cup", spill_max=SPILL
        )

    def run_reward(self):
        # all at once: about half a cup poured, both cups back on the desk upright, both hands open. Not trigger_check:
        # in its "level" mode the conditions are not looked at, so a fill reached mid-pour, cups still held up, passed
        self.reward_manager.check(
            [
                self.pour_check(),
                self.reward_manager.is_all_gripper_open(),
                self.reward_manager.is_axis_up(label="source_cup", axis=[0, 0, 1], threshold=15),
                self.reward_manager.is_axis_up(label="target_cup", axis=[0, 0, 1], threshold=15),
                self.reward_manager.is_not_lift(label="source_cup", z_threshold=0.01),
                self.reward_manager.is_not_lift(label="target_cup", z_threshold=0.01),
            ]
        )

    def gen_instruction(self, env_idx):
        """The layout's own instruction when it has one (an L0 twin layout keeps its real episode's), else the
        platform's instruction set with the target cup named as the real data names it."""
        layout = self.scene_manager.layout_manager.saved_layouts[env_idx] or {}
        if layout.get("Instruction"):
            return [layout["Instruction"]]
        lm = self.scene_manager.layout_manager
        name = lm.get_instance_name(env_idx, "target_cup")
        names = TARGET_NAMES.get(
            lm.decode_inst_name(name) if name else None, {"zh": "另一个杯子", "en": "the other cup"}
        )
        sets = json.loads(INSTRUCTIONS.read_text(encoding="utf-8"))
        return [s.replace("<target_cup>", names[lang]) for lang in ("zh", "en") for s in sets.get(lang, [])]


class g1d_pour_beans_half_cup(G1dPourBeansHalfCupCommon, TaskEnv):
    pass
