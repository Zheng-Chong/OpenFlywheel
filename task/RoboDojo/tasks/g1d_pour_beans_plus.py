import json
from pathlib import Path

from env.environment.task_env import TaskEnv
from task.RoboDojo.tasks.g1d_pour_beans_cup_to_cup import G1dPourBeansCupToCupCommon

INSTRUCTIONS = Path(__file__).resolve().parents[3] / "real2sim/tasks/g1d_pour_beans_plus/instructions.json"
IN_CUP = 0.95  # of the beans in the target cup's cavity (the platform definition)
UPRIGHT_DEG, REST_M = 15, 0.01


class G1dPourBeansPlusCommon(G1dPourBeansCupToCupCommon):
    """倒豆子 Plus (Pour Beans Plus): two cups of any shape on the desk, beans in cup A (source_cup). Each hand picks
    one cup, A is tipped over B (target_cup) in mid-air until all its beans are in B, and both cups go back upright
    on the desk with the hands open. The cups come from the pool of every cup that stands and holds beans
    (real2sim/tasks/g1d_pour_beans_plus/cups.json); each layout entry carries its cup's cavity profile
    (scripts/objects/cup_profile.py), which the success check counts the beans against.

    Layouts (and their instructions, naming each cup by a random one of its names or by its side) come from
    scripts/datagen/pour_beans_plus_layouts.py.
    """

    def __init__(self, config, app, **kwargs):
        super().__init__(config, app, **kwargs)
        self.step_lim = 1500  # picking both cups, pouring a cup of any size empty, putting both back

    def pour_check(self):
        return self.reward_manager.is_fluid_in_cup_profile("beans", "target_cup", share=IN_CUP)

    def run_reward(self):
        rm = self.reward_manager
        rm.check(
            [
                self.pour_check(),
                rm.is_all_gripper_open(),
                *(
                    rm.is_axis_up(label=cup, axis=[0, 0, 1], threshold=UPRIGHT_DEG)
                    for cup in ("source_cup", "target_cup")
                ),
                *(rm.is_not_lift(label=cup, z_threshold=REST_M) for cup in ("source_cup", "target_cup")),
            ]
        )

    def gen_instruction(self, env_idx):
        """The layout's own instruction; without one, every template with the cups' simplest names."""
        layout = self.scene_manager.layout_manager.saved_layouts[env_idx] or {}
        if layout.get("Instruction"):
            return [layout["Instruction"]]
        lm = self.scene_manager.layout_manager
        sets = json.loads(INSTRUCTIONS.read_text(encoding="utf-8"))
        bean = next(iter(layout.get("Fluid", {}).values()), [{}])[0].get("bean") or {"zh": "红豆", "en": "red beans"}
        out = []
        for lang, beans in (("zh", bean["zh"]), ("en", bean["en"])):
            names = {}
            for label in ("source_cup", "target_cup"):
                found = lm.get_label_descriptions(env_idx, label=label, lang=lang) or ["cup"]
                names[label] = found[1] if len(found) > 1 else found[0]  # the simplest name after the category
            for s in sets[lang]:
                s = s.replace("<source_cup>", names["source_cup"]).replace("<target_cup>", names["target_cup"])
                out.append(s.replace("<beans>", beans))
        return out


class g1d_pour_beans_plus(G1dPourBeansPlusCommon, TaskEnv):
    pass
